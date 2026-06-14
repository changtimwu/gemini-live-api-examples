"""End-to-end ASR glossary-rephrasing pipeline for ONE video — the unit of work that
run_parallel.py fans out across a corpus.

The four steps the user asked for, chained:
  1. fetch        — download the video's audio (16k mono PCM) + ground-truth caption.
  2. glossary_llm — read the caption with an LLM, surface companies + jargon, build a glossary SI.
  3. rephrase     — transcribe the audio with a *plain* recognizer prompt (no glossary baked in),
                    then post-correct that raw transcript using the glossary SI.
  4. score        — CER + domain-term recall of raw vs rephrased, both against the caption.

The A/B is deliberately the live app's general-vs-glossary split (gemini-live-asr-glossary-compare):
both transcripts come from the *same* glossary-free transcription, so the only difference is the
rephrase pass — which is exactly the effect we want to measure. (eval_full.py instead bakes the
glossary into the recognizer; this measures the post-transcribe correction the user found works.)

Self-contained so it can be spawned as an isolated subprocess. Writes results/<id>_pipeline.json
incrementally with a status field (running | complete | interrupted | error), so a crash mid-run
keeps the expensive transcripts collected so far.

  python pipeline.py --url https://youtu.be/... [--start 0 --end 300] [--trials 1]
                     [--analyzer-model gemini-3.1-flash-lite] [--rephrase-model gemini-3.1-flash-lite]
                     [--add-tickers] [--chunk 480] [--out path.json]
"""
import argparse
import asyncio
import json
import os
import statistics
import sys
import time

import fetch
import glossary_llm
from rephrase import (build_rephrase_instruction, rephrase_transcript,
                      strip_ticker_annotations)
import transcribe_batch
from score import cer, term_recall
from transcribe import SAMPLE_RATE, transcribe_pcm
ASR_MODEL_DEFAULT = transcribe_batch.DEFAULT_BATCH_MODEL   # batch gemini-3.1 ASR (cheap; see transcribe_batch)

# Plain recognizer prompt — identical to the live app's BASE_INSTRUCTION, used for BOTH arms so the
# transcription is glossary-free and the rephrase pass is the only variable.
BASE_INSTRUCTION = (
    "這是一段台灣股市分析的廣播，內容可能會提到台灣上市櫃公司的名稱與股票代號，"
    "請盡量正確辨識並轉寫所有內容。"
)
CHUNK_SECS = 480  # 8 min — safely under the Live API audio-only session limit


def _agg(xs):
    xs = [x for x in xs if x is not None and x == x]  # drop None (no-term recall) + nan (empty ref)
    if not xs:
        return None
    return {"mean": round(statistics.mean(xs), 4),
            "min": round(min(xs), 4), "max": round(max(xs), 4)}


def _chunks(pcm: bytes, secs: int) -> list[bytes]:
    n = SAMPLE_RATE * 2 * secs
    return [pcm[i:i + n] for i in range(0, len(pcm), n)]


async def _transcribe_retry(ch: bytes, *, api_key: str, exp_secs: float, model: str,
                            retries: int = 4) -> str | None:
    """Transcribe one audio chunk with the plain prompt; None on persistent failure so a bad chunk
    invalidates the trial instead of poisoning the average with empty text. (Mirrors eval_full.)"""
    floor = max(30, exp_secs * 0.5)
    for attempt in range(retries + 1):
        try:
            res = await transcribe_pcm(ch, api_key=api_key, model=model, system_instruction=BASE_INSTRUCTION)
            if len(res["source_zh"]) >= floor:
                return res["source_zh"]
            print(f"      short/empty ({len(res['source_zh'])} chars); retry {attempt+1}/{retries}", flush=True)
        except Exception as e:
            print(f"      transcribe error ({type(e).__name__}: {str(e)[:60]}); retry {attempt+1}/{retries}", flush=True)
        if attempt < retries:
            await asyncio.sleep(min(60, 5 * 2 ** attempt))
    return None


async def _transcribe_full(pcm: bytes, *, api_key: str, chunk_secs: int, model: str) -> tuple[str | None, bool]:
    """Transcribe the whole (possibly long) track with the plain prompt.
    Returns (full_transcript, ok); ok is False if transcription failed/came back too short."""
    total_secs = len(pcm) / (SAMPLE_RATE * 2)
    if transcribe_batch.is_batch_model(model):
        # Batch (non-Live) ASR: one call set, splits internally; no per-chunk Live streaming.
        res = await transcribe_batch.batch_transcribe_pcm(
            pcm, api_key=api_key, model=model, system_instruction=BASE_INSTRUCTION)
        zh = res["source_zh"]
        ok = len(zh) >= max(100, total_secs * 0.3)   # batch normalizes/compresses; lenient floor
        print(f"    batch {model} -> {len(zh)} chars" + ("" if ok else "  (too short -> trial invalid)"), flush=True)
        return (zh if ok else None), ok
    chs = _chunks(pcm, chunk_secs)
    parts, ok = [], True
    for ci, ch in enumerate(chs):
        exp = len(ch) / (SAMPLE_RATE * 2)
        hyp = await _transcribe_retry(ch, api_key=api_key, exp_secs=exp, model=model)
        if hyp is None:
            ok, hyp = False, ""
            print(f"    chunk {ci+1}/{len(chs)} -> FAILED (trial invalid)", flush=True)
        else:
            print(f"    chunk {ci+1}/{len(chs)} -> {len(hyp)} chars", flush=True)
        parts.append(hyp)
    return (" ".join(parts).strip() if ok else None), ok


def build_glossary(gt_text: str, *, api_key: str, analyzer_model: str) -> tuple[str, dict, dict]:
    """Step 2: LLM glossary from the caption. Returns (si_text, recall_terms, glossary_dict).
    recall_terms is {ticker: name} for the stocks that carry a confirmed ticker (what score wants)."""
    g = glossary_llm.analyze(gt_text, api_key=api_key, model=analyzer_model)
    g.stocks, report = glossary_llm.validate_stocks(g.stocks)
    for s in g.stocks:                         # the SI builder + English map expect a non-empty name
        if not s.english.strip():
            s.english = s.name
    si = glossary_llm.build_system_instruction(g)
    terms = {s.ticker: s.name for s in g.stocks if s.ticker}
    return si, terms, {"glossary": g.model_dump(), "ticker_report": report}


async def run(url: str, *, api_key: str, t0: float | None, t1: float | None, trials: int,
              chunk_secs: int, asr_model: str, analyzer_model: str, rephrase_model: str,
              add_tickers: bool, out_path: str) -> dict:
    t_start = time.monotonic()
    vid = fetch._video_id(url)

    # --- step 1: fetch audio + ground-truth caption -------------------------------------------
    pcm, gt = fetch.fetch(url, t0, t1)
    audio_secs = len(pcm) / (SAMPLE_RATE * 2)
    if not gt.strip():
        sys.exit(f"{vid}: empty caption — no Chinese auto-captions for this video/window")
    print(f"[{vid}] audio {audio_secs:.0f}s | caption {len(gt)} chars", flush=True)

    # --- step 2: LLM glossary -> rephrase system instruction -----------------------------------
    si, terms, glossary_info = build_glossary(gt, api_key=api_key, analyzer_model=analyzer_model)
    instr = build_rephrase_instruction(si, add_tickers=add_tickers)
    print(f"[{vid}] glossary: {len(terms)} stocks w/ ticker -> {[f'{n}({t})' for t, n in terms.items()]}", flush=True)

    out = {
        "url": url, "video_id": vid, "window": [t0, t1], "audio_secs": round(audio_secs, 1),
        "trials": trials, "add_tickers": add_tickers,
        "asr_model": asr_model, "analyzer_model": analyzer_model, "rephrase_model": rephrase_model,
        "glossary_terms": terms, "system_instruction": si, **glossary_info,
        "gt_text": gt, "status": "running",
        "arms": {}, "trial_data": [],
    }

    def flush(status: str | None = None):
        if status:
            out["status"] = status
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        json.dump(out, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    cers_raw, cers_reph, nrec_raw, nrec_reph, trec_raw, trec_reph = [], [], [], [], [], []
    # per-term name-hit counts across valid trials, per arm (which terms the rephrase rescues)
    hits = {a: {f"{n}({t})": 0 for t, n in terms.items()} for a in ("raw", "rephrased")}
    out["per_stock_hitrate"] = hits

    try:
        for k in range(trials):
            print(f"[{vid}] trial {k+1}/{trials}: transcribing with {asr_model}…", flush=True)
            raw, ok = await _transcribe_full(pcm, api_key=api_key, chunk_secs=chunk_secs, model=asr_model)
            if not ok:
                out["trial_data"].append({"valid": False, "raw": raw or ""})
                flush()
                continue
            print(f"[{vid}] trial {k+1}/{trials}: rephrasing {len(raw)} chars…", flush=True)
            reph, reph_failed, reph_total = await rephrase_transcript(
                raw, instr, api_key=api_key, model=rephrase_model)
            if reph_failed:
                # A failed rephrase falls back to raw text, which would otherwise look like
                # "rephrasing made no difference". Flag it so the result isn't mistaken for a clean
                # measurement (this is what a depleted-credit / quota run looks like).
                out["rephrase_degraded"] = True
                print(f"[{vid}] ⚠ rephrase degraded: {reph_failed}/{reph_total} chunks fell back "
                      f"to raw (API failure) — this video's rephrased arm is not trustworthy", flush=True)

            reph_for_cer = strip_ticker_annotations(reph)   # fair CER vs ticker-free caption
            c_raw, c_reph = cer(gt, raw), cer(gt, reph_for_cer)
            r_raw, r_reph = term_recall(terms, raw), term_recall(terms, reph)
            cers_raw.append(c_raw); cers_reph.append(c_reph)
            nrec_raw.append(r_raw["name_recall"]); nrec_reph.append(r_reph["name_recall"])
            trec_raw.append(r_raw["ticker_recall"]); trec_reph.append(r_reph["ticker_recall"])
            for term, v in r_raw["per_term"].items():
                hits["raw"][term] += int(v["name_hit"])
            for term, v in r_reph["per_term"].items():
                hits["rephrased"][term] += int(v["name_hit"])

            out["trial_data"].append({"valid": True, "raw": raw, "rephrased": reph,
                                      "rephrase_failed_chunks": reph_failed, "rephrase_chunks": reph_total,
                                      "cer_raw": c_raw, "cer_rephrased": c_reph,
                                      "name_recall_raw": r_raw["name_recall"],
                                      "name_recall_rephrased": r_reph["name_recall"]})
            print(f"[{vid}] trial {k+1}/{trials}: CER {c_raw:.3f} -> {c_reph:.3f}   "
                  f"name_recall {r_raw['name_recall']} -> {r_reph['name_recall']}", flush=True)
            out["arms"] = {
                "raw": {"cer": _agg(cers_raw), "name_recall": _agg(nrec_raw), "ticker_recall": _agg(trec_raw)},
                "rephrased": {"cer": _agg(cers_reph), "name_recall": _agg(nrec_reph), "ticker_recall": _agg(trec_reph)},
            }
            flush()
    except KeyboardInterrupt:
        flush("interrupted"); print(f"\n⚠ interrupted — partial saved to {out_path}", flush=True); raise
    except Exception:
        flush("error"); print(f"\n⚠ error — partial saved to {out_path}", flush=True); raise

    valid = sum(1 for t in out["trial_data"] if t["valid"])
    out["valid_trials"] = valid
    out["elapsed_secs"] = round(time.monotonic() - t_start, 1)
    if valid == 0:
        flush("no_valid_trials")
        print(f"\n[{vid}] ⚠ no valid trials ({trials} attempted) — network/API failures", flush=True)
        return out

    A, B = out["arms"]["raw"], out["arms"]["rephrased"]
    m = lambda agg: agg["mean"] if agg else None     # a recall agg is None when there are no terms
    out["delta"] = {"cer": round(A["cer"]["mean"] - B["cer"]["mean"], 4),
                    "name_recall": round((m(B["name_recall"]) or 0) - (m(A["name_recall"]) or 0), 4)}
    print(f"\n[{vid}] ====== SUMMARY ({valid}/{trials} valid) ======", flush=True)
    print(f"  raw        CER {A['cer']['mean']:.3f}  name_recall {m(A['name_recall'])}", flush=True)
    print(f"  rephrased  CER {B['cer']['mean']:.3f}  name_recall {m(B['name_recall'])}", flush=True)
    print(f"  rephrase effect:  ΔCER {out['delta']['cer']:+.3f} (lower=better)  "
          f"Δname_recall {out['delta']['name_recall']:+.3f} (higher=better)", flush=True)
    rescued = [t for t in hits["raw"] if hits["rephrased"][t] > hits["raw"][t]]
    if rescued:
        print(f"  terms rescued by rephrase: {rescued}", flush=True)
    # Stocks STILL missed after rephrasing (name never found in any valid rephrased trial) — the
    # list to inspect when improving the glossary/rephrase. Also keep the raw-arm misses for diff.
    out["missed_stocks_rephrased"] = sorted(t for t in hits["rephrased"] if hits["rephrased"][t] == 0)
    out["missed_stocks_raw"] = sorted(t for t in hits["raw"] if hits["raw"][t] == 0)
    if out["missed_stocks_rephrased"]:
        print(f"  still missed after rephrase ({len(out['missed_stocks_rephrased'])}): "
              f"{out['missed_stocks_rephrased']}", flush=True)
    # Dump the transcripts as plain .txt next to the JSON so they're easy to eyeball / grep.
    v = next((t for t in out["trial_data"] if t.get("valid")), None)
    if v:
        base = os.path.splitext(out_path)[0]
        with open(base + ".raw.txt", "w", encoding="utf-8") as f:
            f.write(v["raw"])
        with open(base + ".rephrased.txt", "w", encoding="utf-8") as f:
            f.write(v["rephrased"])
        out["transcript_files"] = [base + ".raw.txt", base + ".rephrased.txt"]
    flush("complete")
    print(f"\n[{vid}] wrote {out_path}  ({out['elapsed_secs']}s)", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True, help="YouTube video URL")
    ap.add_argument("--start", type=float, help="clip start (sec); omit for whole video")
    ap.add_argument("--end", type=float, help="clip end (sec); omit for whole video")
    ap.add_argument("--trials", type=int, default=1, help="transcribe+rephrase N times (default 1)")
    ap.add_argument("--chunk", type=int, default=CHUNK_SECS, help="audio chunk seconds for the Live API")
    ap.add_argument("--asr-model", default=ASR_MODEL_DEFAULT,
                    help=f"ASR model (default {ASR_MODEL_DEFAULT}, cheap batch; "
                         f"gemini-3.5-live-translate-preview is the higher-accuracy Live model)")
    ap.add_argument("--analyzer-model", default=glossary_llm.DEFAULT_MODEL, help="LLM for glossary extraction")
    ap.add_argument("--rephrase-model", default="gemini-3.1-flash-lite", help="model for the rephrase pass")
    ap.add_argument("--add-tickers", action="store_true",
                    help="rephrase also appends '(代號)' (live-app behaviour; stripped before CER)")
    ap.add_argument("--out", help="result JSON path (default results/<id>_pipeline.json)")
    args = ap.parse_args()

    api_key = glossary_llm.load_api_key()
    out_path = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "results", f"{fetch._video_id(args.url)}_pipeline.json")
    try:
        asyncio.run(run(args.url, api_key=api_key, t0=args.start, t1=args.end, trials=args.trials,
                        chunk_secs=args.chunk, asr_model=args.asr_model,
                        analyzer_model=args.analyzer_model, rephrase_model=args.rephrase_model,
                        add_tickers=args.add_tickers, out_path=out_path))
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()

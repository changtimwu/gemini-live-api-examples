"""Full-video A/B evaluation: chunk long audio for the Live API, build the glossary from the
WHOLE caption, apply it to every chunk's B run, N trials per arm.

  python eval_full.py --url https://youtu.be/... --trials 3 [--chunk 480]

Aggregation: per trial, concatenate the per-chunk transcriptions into a full-video hypothesis,
then CER + name/ticker recall vs the full caption. Report mean over trials + per-stock A→B
hit-rate. Incremental JSON is written so a crash mid-run keeps partial data."""
import argparse
import asyncio
import json
import os
import statistics
import sys

import fetch
from glossary import build_system_instruction, extract_terms
from score import cer, term_recall
from transcribe import SAMPLE_RATE, transcribe_pcm

CHUNK_SECS = 480  # 8 min — safely under the Live API audio-only session limit


def _key():
    k = os.environ.get("GEMINI_API_KEY")
    if not k:
        sys.exit("set GEMINI_API_KEY")
    return k


def _agg(xs):
    xs = [x for x in xs if x == x]
    return {"mean": round(statistics.mean(xs), 4), "min": round(min(xs), 4),
            "max": round(max(xs), 4)} if xs else None


def _chunks(pcm, secs):
    n = SAMPLE_RATE * 2 * secs
    return [pcm[i:i + n] for i in range(0, len(pcm), n)]


async def _transcribe_retry(ch, key, sysi, exp_secs, retries=4):
    """Return transcription, or None on persistent failure (so a network outage marks the
    trial invalid instead of poisoning averages with empty 0-recall data). Backs off on errors."""
    floor = max(30, exp_secs * 0.5)
    for attempt in range(retries + 1):
        try:
            res = await transcribe_pcm(ch, api_key=key, system_instruction=sysi)
            if len(res["source_zh"]) >= floor:
                return res["source_zh"]
            print(f"      short/empty ({len(res['source_zh'])} chars); retry {attempt+1}/{retries}", flush=True)
        except Exception as e:
            print(f"      transcribe error ({type(e).__name__}: {str(e)[:60]}); retry {attempt+1}/{retries}", flush=True)
        if attempt < retries:
            await asyncio.sleep(min(60, 5 * 2 ** attempt))  # backoff: 5,10,20,40,60s — ride out blips
    return None  # persistent failure


async def run(url, trials, chunk_secs, out_path):
    key = _key()
    pcm, full_gt = fetch.fetch(url)
    segs = fetch.parse_vtt(fetch.download_subtitle(url))
    terms = extract_terms(full_gt)
    si = build_system_instruction(terms)
    chs = _chunks(pcm, chunk_secs)
    total = len(pcm) / (SAMPLE_RATE * 2)
    print(f"full audio {total:.0f}s -> {len(chs)} chunks x {chunk_secs}s | "
          f"glossary {len(terms)} stocks | trials={trials}", flush=True)
    print("glossary:", [f"{n}({t})" for t, n in terms.items()], flush=True)

    raw = {"A_no_si": {}, "B_with_si": {}}     # arm -> trial -> [chunk hyps]
    valid = {"A_no_si": {}, "B_with_si": {}}   # arm -> trial -> bool (False if any chunk failed)
    for arm, sysi in (("A_no_si", None), ("B_with_si", si)):
        for k in range(trials):
            parts, ok = [], True
            for ci, ch in enumerate(chs):
                exp = len(ch) / (SAMPLE_RATE * 2)
                hyp = await _transcribe_retry(ch, key, sysi, exp)
                if hyp is None:
                    ok, hyp = False, ""
                    print(f"  {arm} trial{k+1} chunk{ci+1}/{len(chs)} -> FAILED (trial invalid)", flush=True)
                else:
                    print(f"  {arm} trial{k+1}/{trials} chunk{ci+1}/{len(chs)} -> {len(hyp)} chars", flush=True)
                parts.append(hyp)
            raw[arm][k] = parts; valid[arm][k] = ok
            json.dump({"glossary": terms, "raw": raw, "valid": valid},
                      open(out_path, "w", encoding="utf-8"), ensure_ascii=False)  # incremental (keeps raw)

    # aggregate VALID trials only; per trial full hyp = concat chunks
    summary = {"url": url, "trials": trials, "chunks": len(chs), "glossary": terms,
               "arms": {}, "raw": raw, "valid": valid}
    hitrate = {a: {f"{n}({t})": 0 for t, n in terms.items()} for a in ("A_no_si", "B_with_si")}
    for arm in ("A_no_si", "B_with_si"):
        cers, nrec, trec, nv = [], [], [], 0
        for k in range(trials):
            if not valid[arm][k]:
                continue
            nv += 1
            full_hyp = " ".join(raw[arm][k])
            cers.append(cer(full_gt, full_hyp))
            tr = term_recall(terms, full_hyp)
            nrec.append(tr["name_recall"]); trec.append(tr["ticker_recall"])
            for term, v in tr["per_term"].items():
                hitrate[arm][term] += int(v["name_hit"])
        summary["arms"][arm] = {"valid_trials": nv, "cer": _agg(cers),
                                "name_recall": _agg(nrec), "ticker_recall": _agg(trec)}

    print("\n================ FULL-VIDEO SUMMARY ================", flush=True)
    for arm in ("A_no_si", "B_with_si"):
        a = summary["arms"][arm]
        if a["cer"] is None:
            print(f"{arm:11} NO VALID TRIALS ({trials} attempted) — network/API failures", flush=True)
        else:
            print(f"{arm:11} [{a['valid_trials']}/{trials} valid] CER {a['cer']['mean']:.3f}  "
                  f"name_recall {a['name_recall']['mean']:.3f}  ticker_recall {a['ticker_recall']['mean']:.3f}", flush=True)
    A, B = summary["arms"]["A_no_si"], summary["arms"]["B_with_si"]
    if A["cer"] and B["cer"]:
        print(f"\nGlossary effect (B vs A): ΔCER {A['cer']['mean']-B['cer']['mean']:+.3f} (lower better)  "
              f"Δname_recall {B['name_recall']['mean']-A['name_recall']['mean']:+.3f} (higher better)", flush=True)
        av, bv = A["valid_trials"], B["valid_trials"]
        print("\nper-stock name hit-rate (A → B), where different:", flush=True)
        for term in hitrate["A_no_si"]:
            x, y = hitrate["A_no_si"][term], hitrate["B_with_si"][term]
            if x != y:
                print(f"   {term:18} {x}/{av} → {y}/{bv}", flush=True)
    else:
        print("\n⚠ cannot compare arms — an arm had 0 valid trials (re-run on a stable network).", flush=True)

    summary["per_stock_hitrate"] = hitrate
    json.dump(summary, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\nwrote", out_path, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--chunk", type=int, default=CHUNK_SECS)
    ap.add_argument("--out")
    args = ap.parse_args()
    out = args.out or f"results/{fetch._video_id(args.url)}_full.json"
    asyncio.run(run(args.url, args.trials, args.chunk, out))


if __name__ == "__main__":
    main()

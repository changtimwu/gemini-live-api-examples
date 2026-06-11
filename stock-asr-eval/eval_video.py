"""A/B evaluation orchestrator (per video), with N-trial averaging.

Real mode:   python eval_video.py --url https://youtu.be/... --start 0 --end 300 --trials 3
Local mode:  python eval_video.py --pcm x.pcm --gt x.txt --trials 3

For each arm (A=no SI, B=glossary SI) we run `trials` transcriptions, then report mean CER,
mean name/ticker recall, and per-stock hit-rate across trials (A vs B)."""
import argparse
import asyncio
import json
import os
import statistics
import sys

from glossary import build_system_instruction, extract_terms
from score import cer, normalize, term_recall
from transcribe import SAMPLE_RATE, transcribe_pcm


def _key() -> str:
    k = os.environ.get("GEMINI_API_KEY")
    if not k:
        sys.exit("set GEMINI_API_KEY")
    return k


def _agg(xs):
    xs = [x for x in xs if x == x]  # drop nan
    if not xs:
        return None
    return {"mean": round(statistics.mean(xs), 4),
            "min": round(min(xs), 4), "max": round(max(xs), 4)}


async def evaluate(pcm: bytes, gt_text: str, api_key: str, trials: int = 3) -> dict:
    terms = extract_terms(gt_text)
    si = build_system_instruction(terms)
    print(f"audio {len(pcm)/(SAMPLE_RATE*2):.1f}s | glossary {len(terms)} stocks: "
          f"{[f'{n}({t})' for t,n in terms.items()]}\n")

    out = {"glossary": terms, "trials": trials, "arms": {}}
    # per-stock hit counts across trials, per arm
    hit = {lbl: {f"{n}({t})": 0 for t, n in terms.items()} for lbl in ("A_no_si", "B_with_si")}

    for label, sysi in (("A_no_si", None), ("B_with_si", si)):
        cers, nrec, trec, hyps = [], [], [], []
        for k in range(trials):
            res = await transcribe_pcm(pcm, api_key=api_key, system_instruction=sysi)
            h = res["source_zh"]
            tr = term_recall(terms, h)
            cers.append(cer(gt_text, h)); nrec.append(tr["name_recall"]); trec.append(tr["ticker_recall"])
            hyps.append(h)
            for term, v in tr["per_term"].items():
                hit[label][term] += int(v["name_hit"])
            print(f"  {label} trial {k+1}/{trials}: CER={cers[-1]:.3f} name_recall={nrec[-1]}")
        out["arms"][label] = {"cer": _agg(cers), "name_recall": _agg(nrec),
                              "ticker_recall": _agg(trec), "hyps": hyps}

    print("\n================ SUMMARY ================")
    for label in ("A_no_si", "B_with_si"):
        a = out["arms"][label]
        print(f"{label:11} CER {a['cer']['mean']:.3f} (min {a['cer']['min']} max {a['cer']['max']})  "
              f"name_recall {a['name_recall']['mean']:.3f}  ticker_recall {a['ticker_recall']['mean']:.3f}")
    dc = out["arms"]["A_no_si"]["cer"]["mean"] - out["arms"]["B_with_si"]["cer"]["mean"]
    dr = out["arms"]["B_with_si"]["name_recall"]["mean"] - out["arms"]["A_no_si"]["name_recall"]["mean"]
    print(f"\nGlossary effect (B vs A):  ΔCER {dc:+.3f} (lower=better)   Δname_recall {dr:+.3f} (higher=better)")
    print("\nper-stock name hit-rate (A → B), only where they differ:")
    for term in hit["A_no_si"]:
        a, b = hit["A_no_si"][term], hit["B_with_si"][term]
        if a != b:
            print(f"   {term:18} {a}/{trials} → {b}/{trials}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pcm"); ap.add_argument("--gt")
    ap.add_argument("--url"); ap.add_argument("--start", type=float); ap.add_argument("--end", type=float)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--out")
    args = ap.parse_args()

    if args.url:
        from fetch import fetch
        pcm, gt_text = fetch(args.url, args.start, args.end)
    else:
        pcm = open(args.pcm, "rb").read()
        gt_text = open(args.gt, encoding="utf-8").read()

    result = asyncio.run(evaluate(pcm, gt_text, _key(), trials=args.trials))
    if args.out:
        json.dump(result, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print("\nwrote", args.out)


if __name__ == "__main__":
    main()

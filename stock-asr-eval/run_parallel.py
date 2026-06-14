"""Fan the pipeline.py worker out across a corpus of videos, N at a time.

Each video runs in its own pipeline.py subprocess — a fully isolated "agent": separate Live API
sessions, independent crash domain, its own results/<id>_pipeline.json. --concurrency caps how
many run at once (the real limit is the Gemini Live API's concurrent-session quota, so keep it
modest). When every worker has finished, the per-video results are rolled up into a corpus summary
showing whether post-transcribe rephrasing helps on average.

Inputs (any combination; videos are de-duplicated by id):
  python run_parallel.py <url> [<url> ...]
  python run_parallel.py --file urls.txt                 # one URL per line, # for comments
  python run_parallel.py --playlist <playlist_url> --recent 10

Common options: --concurrency 3, --trials 1, --start/--end (applied to every video, e.g. for a
quick smoke test), --analyzer-model, --rephrase-model, --add-tickers, --chunk, --out-dir.
"""
import argparse
import asyncio
import json
import os
import statistics
import sys

import fetch
import playlist

HERE = os.path.dirname(os.path.abspath(__file__))
PIPELINE = os.path.join(HERE, "pipeline.py")


def _collect_urls(args) -> list[str]:
    urls: list[str] = list(args.urls)
    if args.file:
        for line in open(args.file, encoding="utf-8"):
            line = line.split("#", 1)[0].strip()
            if line:
                urls.append(line)
    if args.playlist:
        urls += [f"https://www.youtube.com/watch?v={e['id']}"
                 for e in playlist.iter_playlist(args.playlist, args.recent)]
    # de-dup by video id, preserve order
    seen, out = set(), []
    for u in urls:
        vid = fetch._video_id(u)
        if vid not in seen:
            seen.add(vid); out.append(u)
    return out


async def _run_one(url: str, args, out_dir: str, sem: asyncio.Semaphore) -> dict:
    """Spawn one pipeline.py subprocess; stream its output prefixed; return a compact result."""
    vid = fetch._video_id(url)
    out_path = os.path.join(out_dir, f"{vid}_pipeline.json")
    cmd = [sys.executable, PIPELINE, "--url", url, "--trials", str(args.trials),
           "--chunk", str(args.chunk), "--asr-model", args.asr_model,
           "--analyzer-model", args.analyzer_model,
           "--rephrase-model", args.rephrase_model, "--out", out_path]
    if args.start is not None:
        cmd += ["--start", str(args.start)]
    if args.end is not None:
        cmd += ["--end", str(args.end)]
    if args.add_tickers:
        cmd += ["--add-tickers"]

    async with sem:
        print(f"▶ start {vid}", flush=True)
        proc = await asyncio.create_subprocess_exec(
            *cmd, cwd=HERE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        tag = vid[:6]
        async for raw in proc.stdout:                       # live, attributed progress
            print(f"  {tag}| {raw.decode(errors='replace').rstrip()}", flush=True)
        rc = await proc.wait()

    res = {"video_id": vid, "url": url, "returncode": rc, "out": out_path}
    if os.path.exists(out_path):
        try:
            data = json.load(open(out_path, encoding="utf-8"))
            res.update({"status": data.get("status"), "valid_trials": data.get("valid_trials", 0),
                        "arms": data.get("arms", {}), "delta": data.get("delta"),
                        "rephrase_degraded": data.get("rephrase_degraded", False)})
        except Exception as e:
            res["status"] = f"unreadable: {e}"
    else:
        res["status"] = "no_output"
    mark = "✔" if rc == 0 and res.get("status") == "complete" else "✗"
    d = res.get("delta") or {}
    print(f"{mark} done {vid}  status={res.get('status')}  "
          f"ΔCER {d.get('cer', '—')}  Δname_recall {d.get('name_recall', '—')}", flush=True)
    return res


def _mean(xs):
    xs = [x for x in xs if x is not None and x == x]
    return round(statistics.mean(xs), 4) if xs else None


def _summarize(results: list[dict]) -> dict:
    """Macro-average the per-video means across the corpus (each video weighted equally).

    Only videos whose rephrase pass ran fully (not degraded by API failures) feed the arm
    comparison — otherwise a quota/credit failure that silently kept the raw text would pollute the
    'rephrased' arm and read as 'rephrasing had no effect'."""
    complete = [r for r in results if r.get("status") == "complete" and r.get("arms")]
    ok = [r for r in complete if not r.get("rephrase_degraded")]
    def arm(metric, which):
        return _mean([r["arms"].get(which, {}).get(metric, {}).get("mean")
                      if r["arms"].get(which, {}).get(metric) else None for r in ok])
    summary = {
        "videos_total": len(results), "videos_complete": len(complete),
        "videos_clean": len(ok), "videos_degraded": len(complete) - len(ok),
        "raw": {"cer": arm("cer", "raw"), "name_recall": arm("name_recall", "raw"),
                "ticker_recall": arm("ticker_recall", "raw")},
        "rephrased": {"cer": arm("cer", "rephrased"), "name_recall": arm("name_recall", "rephrased"),
                      "ticker_recall": arm("ticker_recall", "rephrased")},
        "per_video": [{"video_id": r["video_id"], "status": r.get("status"),
                       "rephrase_degraded": r.get("rephrase_degraded", False), "delta": r.get("delta")}
                      for r in results],
    }
    R, P = summary["raw"], summary["rephrased"]
    if R["cer"] is not None and P["cer"] is not None:
        summary["delta"] = {"cer": round(R["cer"] - P["cer"], 4),
                            "name_recall": round((P["name_recall"] or 0) - (R["name_recall"] or 0), 4)}
    return summary


async def _main(args):
    urls = _collect_urls(args)
    if not urls:
        sys.exit("no videos: pass URLs, --file, or --playlist")
    out_dir = args.out_dir or os.path.join(HERE, "results")
    os.makedirs(out_dir, exist_ok=True)
    print(f"corpus: {len(urls)} videos | concurrency {args.concurrency} | trials {args.trials}\n", flush=True)

    sem = asyncio.Semaphore(args.concurrency)
    results = await asyncio.gather(*(_run_one(u, args, out_dir, sem) for u in urls))

    summary = _summarize(results)
    summary_path = os.path.join(out_dir, "parallel_summary.json")
    json.dump(summary, open(summary_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print("\n================ CORPUS SUMMARY ================", flush=True)
    print(f"{summary['videos_complete']}/{summary['videos_total']} videos completed "
          f"({summary['videos_clean']} clean, {summary['videos_degraded']} rephrase-degraded "
          f"and excluded from the comparison)", flush=True)
    R, P = summary["raw"], summary["rephrased"]
    if R["cer"] is not None:
        print(f"  raw        CER {R['cer']}  name_recall {R['name_recall']}  ticker_recall {R['ticker_recall']}", flush=True)
        print(f"  rephrased  CER {P['cer']}  name_recall {P['name_recall']}  ticker_recall {P['ticker_recall']}", flush=True)
        d = summary.get("delta", {})
        print(f"  rephrase effect (macro):  ΔCER {d.get('cer'):+.3f} (lower=better)  "
              f"Δname_recall {d.get('name_recall'):+.3f} (higher=better)", flush=True)
    else:
        print("  no videos completed — check the per-video logs above.", flush=True)
    print(f"\nwrote {summary_path}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("urls", nargs="*", help="YouTube video URLs")
    ap.add_argument("--file", help="text file of URLs (one per line, # comments)")
    ap.add_argument("--playlist", help="YouTube playlist URL")
    ap.add_argument("--recent", type=int, help="with --playlist: only the N newest videos")
    ap.add_argument("--concurrency", type=int, default=3, help="max videos in flight (Live API quota)")
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--start", type=float, help="clip start (sec), applied to every video")
    ap.add_argument("--end", type=float, help="clip end (sec), applied to every video")
    ap.add_argument("--chunk", type=int, default=480)
    ap.add_argument("--asr-model", default="gemini-3.5-live-translate-preview",
                    help="Live recognizer model (flash-live is non-functional here; see transcribe.py)")
    ap.add_argument("--analyzer-model", default="gemini-3.1-flash-lite")
    ap.add_argument("--rephrase-model", default="gemini-3.1-flash-lite")
    ap.add_argument("--add-tickers", action="store_true")
    ap.add_argument("--out-dir", help="where per-video + summary JSON land (default ./results)")
    args = ap.parse_args()
    try:
        asyncio.run(_main(args))
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()

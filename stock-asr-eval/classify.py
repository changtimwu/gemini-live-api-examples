"""Classify which 錢線百分百 episodes are actually individual-stock discussion (vs macro/policy/
personal-finance), so the ASR eval corpus isn't diluted by episodes that name no Taiwan stocks.

The signal is exactly what the corpus is for: **density of distinct TWSE company names** in the
caption. A stock-discussion episode names many listed companies, repeatedly; a "區域經濟 / 升息 /
退休理財" episode names few or none. We scan the caption against the TWSE/TPEx name dictionary
(stockdict, same one the glossary uses) and score by how many distinct companies are mentioned
≥2 times plus the per-1000-char mention density. This is free (no API) and aligned with the goal.

`--llm` adds a gemini-3.1-flash-lite categorizer (topic label + is-stock-discussion + reason) as a
nuanced cross-check; `--llm-borderline` only spends it on the heuristic's uncertain middle band.

Reads the playlist cache populated by `playlist.py --what subs`
(data/playlists/<id>/<vid>.{info.json,<lang>.vtt}) and writes:
  results/classification.json   — every video, ranked, with features + label(s)
  results/stock_urls.txt        — URLs labelled stock-related, for `run_parallel.py --file`

  python classify.py <playlist_url> [--min-distinct2 5] [--min-density 3] [--llm|--llm-borderline]
"""
import argparse
import json
import os
import re
import sys
from collections import Counter

import fetch
import playlist
import stockdict
from glossary import STOPLIST

# Company-name candidates to scan for: dictionary names ≥2 chars, minus the common-word stoplist.
# (Single-char and stoplisted names are everyday words — they'd false-positive on any show.)
_NAMES = sorted((n for n in stockdict.BY_NAME
                 if len(n) >= 2 and n not in STOPLIST),
                key=len, reverse=True)                      # longest-first → longest match wins
_SCAN = re.compile("|".join(re.escape(n) for n in _NAMES))


def scan_stocks(text: str) -> Counter:
    """Count non-overlapping stock-name occurrences in the caption (聯發科 beats 聯發 at a position)."""
    return Counter(_SCAN.findall(text))


def features(text: str) -> dict:
    counts = scan_stocks(text)
    chars = len(text)
    n_match = sum(counts.values())
    distinct2 = [n for n, c in counts.items() if c >= 2]
    return {
        "chars": chars,
        "n_match": n_match,
        "distinct1": len(counts),
        "distinct2": len(distinct2),
        "density": round(n_match / (chars / 1000), 2) if chars else 0.0,
        "top": [f"{n}×{c}" for n, c in counts.most_common(8)],
    }


def heuristic_label(f: dict, min_distinct2: int, min_density: float, rescue_density: float) -> str:
    """stock if EITHER:
      - it names enough distinct companies repeatedly AND densely (long episodes), OR
      - it's very stock-dense with ≥3 distinct companies (short clips can't clear distinct2≥5,
        but density ≥ rescue_density has no false positives against macro/policy/finance episodes).
    """
    long_form = f["distinct2"] >= min_distinct2 and f["density"] >= min_density
    dense_clip = f["density"] >= rescue_density and f["distinct1"] >= 3
    return "stock" if (long_form or dense_clip) else "non_stock"


# ---- optional LLM categorizer (gemini-3.1-flash-lite) ----------------------------------------

LLM_SYSTEM = ("你是財經節目分類助理。判斷一集節目是否以『台灣個股討論』為主軸——"
              "也就是大量提到具體的台灣上市櫃公司、個股技術分析、籌碼、產業類股輪動等。"
              "若主軸是總體經濟、利率政策、區域經濟、個人理財／退休規劃、房地產、保險等，"
              "即使偶爾提到一兩家公司，也不算個股討論。只依據提供的字幕內容判斷。")
LLM_PROMPT = ("以下是一集節目的字幕（可能截斷）。請分類：\n"
              "- category：individual_stocks / macro_or_policy / personal_finance / other 之一\n"
              "- is_stock_discussion：true/false（是否以台灣個股討論為主軸）\n"
              "- confidence：0~1\n"
              "- reason：12 字內中文理由\n\n字幕：\n---\n{sub}\n---")


def llm_classify(text: str, *, api_key: str, model: str = "gemini-3.1-flash-lite") -> dict:
    from pydantic import BaseModel
    from google import genai
    from google.genai import types

    class Verdict(BaseModel):
        category: str
        is_stock_discussion: bool
        confidence: float = 0.0
        reason: str = ""

    # Sample the caption: head + middle (cheap, and topic is usually clear from a slice).
    sample = text if len(text) <= 6000 else text[:4000] + "\n…\n" + text[len(text)//2: len(text)//2 + 2000]
    client = genai.Client(api_key=api_key)
    resp = client.models.generate_content(
        model=model, contents=LLM_PROMPT.format(sub=sample),
        config=types.GenerateContentConfig(
            system_instruction=LLM_SYSTEM, response_mime_type="application/json",
            response_schema=Verdict, temperature=0.0,
            thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW)))
    v = resp.parsed if resp.parsed is not None else Verdict(**json.loads(resp.text))
    return v.model_dump()


def _load_videos(url: str) -> list[dict]:
    """Read the playlist cache: each video's id, title, duration, upload_date, and subtitle path."""
    plid = playlist._playlist_id(url)
    pdir = os.path.join(playlist.DATA, "playlists", plid)
    index_path = os.path.join(pdir, "index.json")
    if not os.path.exists(index_path):
        sys.exit(f"no cache at {pdir} — run: python playlist.py '{url}' --recent 100 --what subs")
    entries = json.load(open(index_path, encoding="utf-8"))["entries"]
    out = []
    for e in entries:
        vid = e["id"]
        info_path = os.path.join(pdir, f"{vid}.info.json")
        title, date, dur = e.get("title"), None, e.get("duration")
        if os.path.exists(info_path):
            info = json.load(open(info_path, encoding="utf-8"))
            title = info.get("title") or title
            date = info.get("upload_date")
            dur = info.get("duration") or dur
        sub = playlist.cached_subtitle(vid, data_dir=playlist.DATA)
        out.append({"video_id": vid, "title": title, "upload_date": date,
                    "duration": dur, "sub_path": sub})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("url", help="playlist URL (must already be cached via playlist.py --what subs)")
    ap.add_argument("--min-distinct2", type=int, default=5,
                    help="stock if ≥ this many companies are each mentioned ≥2× (default 5)")
    ap.add_argument("--min-density", type=float, default=3.0,
                    help="…AND ≥ this many stock-name mentions per 1000 chars (default 3.0)")
    ap.add_argument("--rescue-density", type=float, default=6.0,
                    help="short clips: stock if density ≥ this with ≥3 companies (default 6.0)")
    ap.add_argument("--llm", action="store_true", help="also run the flash-lite categorizer on every video")
    ap.add_argument("--llm-borderline", action="store_true",
                    help="run the categorizer only on videos near the heuristic threshold (cheaper)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "results", "classification.json"))
    args = ap.parse_args()

    vids = _load_videos(args.url)
    api_key = None
    if args.llm or args.llm_borderline:
        from glossary_llm import load_api_key
        api_key = load_api_key()

    rows = []
    for v in vids:
        if not v["sub_path"] or not os.path.exists(v["sub_path"]):
            rows.append({**v, "label": "no_subtitle"}); continue
        text = fetch.subtitle_text(fetch.parse_vtt(v["sub_path"]))
        f = features(text)
        label = heuristic_label(f, args.min_distinct2, args.min_density, args.rescue_density)
        kind = "full" if "完整版" in (v.get("title") or "") else "segment"
        rows.append({**v, **f, "kind": kind, "label": label})

    # LLM pass (all, or just the borderline band around the threshold)
    if api_key:
        def borderline(r):
            return r.get("label") in ("stock", "non_stock") and \
                args.min_distinct2 - 2 <= r.get("distinct2", 0) <= args.min_distinct2 + 2
        targets = [r for r in rows if r.get("label") in ("stock", "non_stock")
                   and (args.llm or borderline(r))]
        for i, r in enumerate(targets, 1):
            text = fetch.subtitle_text(fetch.parse_vtt(r["sub_path"]))
            print(f"  llm {i}/{len(targets)} {r['video_id']} …", flush=True)
            try:
                r["llm"] = llm_classify(text, api_key=api_key)
            except Exception as e:
                r["llm"] = {"error": f"{type(e).__name__}: {str(e)[:60]}"}

    # rank: stock first, then by distinct2 desc, then density desc
    rows.sort(key=lambda r: (r.get("label") == "stock", r.get("distinct2", -1), r.get("density", -1)),
              reverse=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(rows, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    urls_path = os.path.join(os.path.dirname(args.out), "stock_urls.txt")
    stock = [r for r in rows if r.get("label") == "stock"]
    with open(urls_path, "w", encoding="utf-8") as fh:
        for r in stock:
            fh.write(f"https://www.youtube.com/watch?v={r['video_id']}  # {(r.get('title') or '')[:50]}\n")

    n_sub = sum(1 for r in rows if r.get("label") != "no_subtitle")
    print(f"\n{len(rows)} videos | {n_sub} with subtitles | "
          f"{len(stock)} stock / {n_sub - len(stock)} non-stock "
          f"(threshold: distinct2≥{args.min_distinct2} & density≥{args.min_density})")
    print(f"{'label':10} {'kind':>7} {'d2':>3} {'dens':>5} {'min':>4}  title")
    for r in rows:
        if r.get("label") == "no_subtitle":
            print(f"{'NO_SUB':10} {'—':>7} {'—':>3} {'—':>5} {'—':>4}  {(r.get('title') or '')[:44]}"); continue
        mins = f"{int(r.get('duration') or 0)//60}m"
        llm = r.get("llm", {})
        tag = f"  [llm:{llm.get('category','?')}/{'Y' if llm.get('is_stock_discussion') else 'N'}]" if llm else ""
        print(f"{r['label']:10} {r.get('kind',''):>7} {r['distinct2']:>3} {r['density']:>5} {mins:>4}  "
              f"{(r.get('title') or '')[:44]}{tag}")
    print(f"\nwrote {args.out}\nwrote {urls_path}  ({len(stock)} stock URLs for run_parallel --file)")


if __name__ == "__main__":
    main()

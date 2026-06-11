"""Build an ASR glossary system-instruction from a stock show's YouTube subtitle, using an LLM.

Unlike glossary.py (which scans the caption against a fixed TWSE dictionary), this reads the
WHOLE subtitle once with a Gemini text model and asks it to surface anything worth pinning for
ASR: Taiwan-listed company names (+ ticker) and stock-market / technical jargon that a general
recognizer is likely to mis-hear. Extracted tickers are then cross-checked against the TWSE/TPEx
dictionary (stockdict.py): a wrong code is repaired from the spoken name when the dict knows it,
and an unconfirmable code is dropped so the instruction never asserts a bogus ticker. The result
is written as a ready-to-use Chinese system instruction text file — the same shape
transcribe.py / asr-config.ts consume.

The Gemini API key is read from .env.local (GEMINI_API_KEY or GOOGLE_API_KEY); the env var of
the same name also works. .env.local is searched in this dir, then the repo root, then the cwd.

CLI:  python glossary_llm.py <youtube_url> [--model gemini-31-flash-lite] [--out path.txt]
"""
import argparse
import json
import os
import subprocess
import sys

from pydantic import BaseModel

from google import genai
from google.genai import types

import fetch
import stockdict

# Analyzer model. The API id is gemini-3.1-flash-lite (note the dot). Override with --model.
DEFAULT_MODEL = "gemini-3.1-flash-lite"

ANALYZER_SYSTEM = (
    "你是台灣股市節目字幕的術語擷取助理。你的輸出會被用來當作語音辨識（ASR）模型的提示詞，"
    "幫助它正確辨識容易聽錯的專有名詞。請只根據實際出現在字幕中的內容作答，不要臆測或補充字幕沒提到的標的。"
)

EXTRACT_PROMPT = """以下是一段台灣股市分析節目的完整字幕（YouTube 自動字幕，可能有辨識錯誤）。
請通讀整段字幕，找出值得放進 ASR 詞彙表（glossary）的項目，分兩類：

1. stocks：字幕中提到的台灣上市／上櫃公司。
   - name：該公司最常被口語稱呼的標準中文名稱（例如「台積電」「聯發科」）。
   - ticker：對應的台股代號（4–6 位數字字串，例如 "2330"）。若無法確定就留空字串。
   - english：國際通用英文名稱（若知道，例如 "TSMC"），不確定就留空字串。

2. terms：字幕中提到、且一般辨識模型容易聽錯的股市／財經／技術專有名詞
   （例如指標、籌碼術語、產業／技術名詞、外資機構名等）。
   - term：標準中文寫法。
   - explanation：一句話簡短說明（10–25 字）。

規則：
- 只收錄字幕中真的有提到的項目；寧缺勿濫。
- 同一標的／名詞只列一次，使用最標準的寫法。
- 一般常用詞（如「股票」「上漲」「公司」）不要收錄。

字幕內容：
---
{subtitle}
---
"""


class Stock(BaseModel):
    name: str
    ticker: str = ""
    english: str = ""


class Term(BaseModel):
    term: str
    explanation: str = ""


class Glossary(BaseModel):
    stocks: list[Stock]
    terms: list[Term]


# Preferred zh auto-caption codes, best first. fetch.download_subtitle hardcodes zh-Hant, but
# many Taiwan channels publish the source track as zh-TW (already Traditional); the
# "<lang>-zh-TW" entries are machine re-translations, so they rank below the real source.
ZH_SUB_LANGS = ["zh-Hant", "zh-TW", "zh-Hant-zh-TW", "zh-Hans", "zh"]


def download_subtitle(url: str) -> str:
    """Download the best available Chinese auto-caption to data/, returning its path.

    Like fetch.download_subtitle but tries several zh codes and picks the highest-priority track
    that actually exists for this video (not every channel exposes zh-Hant)."""
    vid = fetch._video_id(url)

    def first_present():
        for lang in ZH_SUB_LANGS:
            p = os.path.join(fetch.DATA, f"{vid}.{lang}.vtt")
            if os.path.exists(p):
                return p
        return None

    hit = first_present()
    if hit:
        return hit
    os.makedirs(fetch.DATA, exist_ok=True)
    # One language at a time, best first: stop at the first track that downloads. Requesting all
    # langs at once makes yt-dlp fetch every matching (incl. machine-translated) track and trip
    # YouTube's 429 rate limit.
    for lang in ZH_SUB_LANGS:
        subprocess.run(["yt-dlp", "-q", "--no-warnings", "--skip-download", "--write-auto-subs",
                        "--sub-langs", lang, "--sub-format", "vtt",
                        "-o", os.path.join(fetch.DATA, "%(id)s.%(ext)s"), url],
                       capture_output=True)  # tolerate per-lang failure (missing track / 429)
        p = os.path.join(fetch.DATA, f"{vid}.{lang}.vtt")
        if os.path.exists(p):
            return p
    sys.exit(f"no Chinese auto-captions found (tried {', '.join(ZH_SUB_LANGS)}). "
             f"Check `yt-dlp --list-subs {url}`.")


def load_api_key() -> str:
    """GEMINI_API_KEY / GOOGLE_API_KEY from the environment, else from the nearest .env.local."""
    for var in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        if os.environ.get(var):
            return os.environ[var]
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [os.path.join(here, ".env.local"),
                  os.path.join(here, "..", ".env.local"),
                  os.path.join(os.getcwd(), ".env.local")]
    for path in candidates:
        if not os.path.exists(path):
            continue
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            k = k.replace("export ", "").strip()
            v = v.strip().strip('"').strip("'")
            if k in ("GEMINI_API_KEY", "GOOGLE_API_KEY") and v:
                return v
    sys.exit("no API key: set GEMINI_API_KEY or put it in .env.local")


def analyze(subtitle: str, *, api_key: str, model: str = DEFAULT_MODEL) -> Glossary:
    """Single pass over the whole subtitle -> structured glossary."""
    client = genai.Client(api_key=api_key)
    resp = client.models.generate_content(
        model=model,
        contents=EXTRACT_PROMPT.format(subtitle=subtitle),
        config=types.GenerateContentConfig(
            system_instruction=ANALYZER_SYSTEM,
            response_mime_type="application/json",
            response_schema=Glossary,
            temperature=0.0,
        ),
    )
    if resp.parsed is not None:
        return resp.parsed
    return Glossary(**json.loads(resp.text))  # fallback if SDK didn't auto-parse


def validate_stocks(stocks: list[Stock]) -> tuple[list[Stock], dict[str, list[str]]]:
    """Cross-check each LLM-extracted stock against the TWSE/TPEx dict (stockdict).

    - spoken name known to the dict -> trust the dict's ticker (repair if the model's differs);
    - name unknown but the model's ticker is a real code -> keep it (likely an alias/variant);
    - neither confirmable -> drop the code so the instruction lists the name without a bogus
      ticker (these are usually foreign stocks/ETFs/indices the TWSE dict doesn't cover).

    Returns the validated stocks plus a report of what was verified / corrected / unverified.
    """
    out: list[Stock] = []
    report: dict[str, list[str]] = {"verified": [], "corrected": [], "unverified": []}
    for s in stocks:
        name = s.name.strip()
        name_ticker = stockdict.BY_NAME.get(name)
        if name_ticker:
            if s.ticker == name_ticker:
                report["verified"].append(f"{name}({s.ticker})")
            else:
                report["corrected"].append(f"{name} {s.ticker or '—'}→{name_ticker}")
                s = s.model_copy(update={"ticker": name_ticker})
        elif s.ticker and s.ticker in stockdict.BY_TICKER:
            report["verified"].append(f"{name}({s.ticker})")  # valid code, name not in dict
        else:
            report["unverified"].append(f"{name}" + (f" (dropped bad code {s.ticker})" if s.ticker else ""))
            if s.ticker:
                s = s.model_copy(update={"ticker": ""})
        # prefer the dict's authoritative English over the model's guess when we have a code
        eng = stockdict.english(s.ticker) if s.ticker else None
        if eng:
            s = s.model_copy(update={"english": eng})
        out.append(s)
    return out, report


def build_system_instruction(g: Glossary) -> str:
    """Assemble the ASR system instruction. Mirrors glossary.py's phrasing, with a second
    paragraph for technical terms so the recognizer pins jargon too."""
    parts: list[str] = ["這是一段台灣股市分析的廣播。"]
    if g.stocks:
        items = "、".join(f"{s.name}（{s.ticker}）" if s.ticker else s.name for s in g.stocks)
        parts.append("內容會提到下列台灣上市櫃公司，請正確辨識並轉寫這些公司的中文名稱與股票代號："
                     + items + "。")
    if g.terms:
        items = "、".join(t.term for t in g.terms)
        parts.append("內容也會提到下列股市／財經專有名詞，請正確辨識並轉寫：" + items + "。")
    return "".join(parts)


def main():
    ap = argparse.ArgumentParser(description="Build an ASR glossary system instruction from a "
                                             "stock show's YouTube subtitle, via an LLM.")
    ap.add_argument("url", help="YouTube link of the stock show")
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"analyzer model (default {DEFAULT_MODEL})")
    ap.add_argument("--out", help="output .txt path (default results/<video_id>.si.txt)")
    args = ap.parse_args()

    vid = fetch._video_id(args.url)
    out_path = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "results", f"{vid}.si.txt")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    print(f"fetching subtitle for {vid} ...", flush=True)
    sub_path = download_subtitle(args.url)
    print(f"using {os.path.basename(sub_path)}", flush=True)
    segs = fetch.parse_vtt(sub_path)
    subtitle = fetch.subtitle_text(segs)
    if not subtitle.strip():
        sys.exit("empty subtitle — no zh-Hant auto-captions found for this video")
    print(f"subtitle: {len(subtitle)} chars | analyzing with {args.model} ...", flush=True)

    g = analyze(subtitle, api_key=load_api_key(), model=args.model)
    g.stocks, report = validate_stocks(g.stocks)
    si = build_system_instruction(g)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(si + "\n")
    # structured glossary alongside the SI, for inspection / reuse
    json_path = os.path.splitext(out_path)[0] + ".glossary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(g.model_dump(), f, ensure_ascii=False, indent=2)

    print(f"\nfound {len(g.stocks)} stocks, {len(g.terms)} terms", flush=True)
    print(f"tickers: {len(report['verified'])} verified, {len(report['corrected'])} corrected, "
          f"{len(report['unverified'])} unverified", flush=True)
    if report["corrected"]:
        print("  corrected:", report["corrected"], flush=True)
    if report["unverified"]:
        print("  unverified:", report["unverified"], flush=True)
    print("terms :", [t.term for t in g.terms], flush=True)
    print("\n----- system instruction -----")
    print(si)
    print("------------------------------")
    print(f"\nwrote {out_path}\nwrote {json_path}", flush=True)


if __name__ == "__main__":
    main()

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
import re
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
   - english：該公司的英文名稱，**每一檔都必須填**（例如 "TSMC"、"MediaTek"）。
     若沒有通用英文名，就給最接近的英文音譯／翻譯，絕對不要留空。
   - example：一句示範該公司講法的短句（**最多 25 字、只能一句**，每一檔都要給）。
     參考字幕中講到它的語氣自行改寫成通順短句，不要照抄一大段；句中必須含公司名稱。
     例如「台積電今天帶量大漲衝上歷史新高」「久元連續兩天漲停板」。

2. terms：字幕中提到的股市／財經／技術專有名詞，包含兩種：
   (a) 一般辨識模型容易聽錯的詞（指標、籌碼術語、外資機構名等）；
   (b) 重要的半導體／產業／製程／材料名詞，即使不算難聽錯也要收（如「晶圓」「製程」「封裝」「良率」）。
   - term：標準中文寫法。
   - english：翻成英文時應使用的標準英文說法（例如 矽光子→"silicon photonics"、
     晶圓→"wafer"、外資→"foreign institutional investors"）；若本身就是英文縮寫（如 CoWoS、HBM），原樣重複。
   - example：一句示範該名詞講法的短句（**最多 25 字、只能一句**，每個詞都要給）。
     字幕沒有標點，請參考其語氣自行改寫成通順短句，**切勿照抄一整段**；句中必須含該名詞。
   - explanation：一句話簡短說明（10–25 字）。

規則：
- 只收錄字幕中真的有提到的項目；寧缺勿濫。
- 同一標的／名詞只列一次，使用最標準的寫法。
- 一般常用詞（如「股票」「上漲」「公司」）不要收錄。
- example 一定要精簡，最多 25 字，超過就是錯的。
{must_include}
字幕內容：
---
{subtitle}
---
"""


class Stock(BaseModel):
    name: str
    ticker: str = ""
    english: str = ""
    example: str = ""  # an example sentence mirroring how the show uses the name


class Term(BaseModel):
    term: str
    english: str = ""
    example: str = ""  # an example sentence mirroring how the show uses the term
    explanation: str = ""


class Glossary(BaseModel):
    stocks: list[Stock]
    terms: list[Term]


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


EXAMPLE_MAX = 30  # hard cap; captions are unpunctuated so the model sometimes over-extracts


def analyze(subtitle: str, *, api_key: str, model: str = DEFAULT_MODEL,
            must_include: list[str] = ()) -> Glossary:
    """Single pass over the whole subtitle -> structured glossary. `must_include` terms are
    forced into the prompt so they're always captured (even common ones the model would skip)."""
    must = ""
    if must_include:
        must = ("- 下列名詞務必收錄到 terms（即使常見也要收），並比照格式給 english／example／explanation："
                + "、".join(must_include) + "。\n")
    client = genai.Client(api_key=api_key)
    resp = client.models.generate_content(
        model=model,
        contents=EXTRACT_PROMPT.format(subtitle=subtitle, must_include=must),
        config=types.GenerateContentConfig(
            system_instruction=ANALYZER_SYSTEM,
            response_mime_type="application/json",
            response_schema=Glossary,
            temperature=0.0,
            # Gemini 3 reasoning effort — "medium" for this scan-and-extract task.
            thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.MEDIUM),
        ),
    )
    g = resp.parsed if resp.parsed is not None else Glossary(**json.loads(resp.text))
    for s in g.stocks:
        s.example = _clamp_example(s.example)
    for t in g.terms:
        t.example = _clamp_example(t.example)
    return g


def _clamp_example(ex: str) -> str:
    """Keep examples short. Captions have no punctuation, so the model occasionally returns a
    huge span; trim to the first sentence-ish unit and hard-cap the length."""
    ex = re.split(r"[。！？!?\n]", ex.strip(), 1)[0].strip()
    return ex[:EXAMPLE_MAX] if len(ex) > EXAMPLE_MAX else ex


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


# EXPERIMENT (hardcode): extra, near-verbatim example sentences for names the live model keeps
# mis-hearing on dqDTSms4Imo (京元電子→「金源電」, 欣銓→「新全/新權」, Claude→「Crown/Cloud/Crowd」).
# Taken almost verbatim from the subtitle to give the recognizer a stronger prior. Keyed by glossary
# name; injected only when that name is present, so it's inert on videos that don't mention these.
# Remove after the trial.
EXTRA_EXAMPLES: dict[str, list[str]] = {
    "京元電子": [
        "京元電子來講他算是突破了一個大型區間",
        "封測族群日月光投控京元電子精材跟欣銓",
        "你可能就要特別擔心京元電子",
    ],
    "欣銓": [
        "第三檔的是欣銓今天漲停昨天也漲停",
        "這三檔都不是第一天漲停南茂雍智欣銓",
        "南茂雍智欣銓我們就來看他們的K線",
        "封測族群日月光投控京元電子精材跟欣銓",
        "欣銓來講他是受惠到最近ASIC的晶片",
        "欣銓的話股價在這兩天比較強一點點",
    ],
    "Claude": [
        "我不管你是Gemini還是ChatGPT還是Claude",
        "不管是Gemini還是ChatGPT還是Claude他們誰贏",
        "像Gemini、ChatGPT、Claude這些AI模型",
    ],
}

# EXPERIMENT (hardcode): negative cues — name the exact wrong homophone the live model keeps
# producing so it steers away from it. Injected when the name is present. Use for a true homophone
# the recognizer can't disambiguate by sound (精材/精彩, 外溢/外意); can be combined with a phonetic
# cue (欣銓 needs both — the phonetic cue alone let 新銓 creep back).
NEGATIVE_CUES: dict[str, list[str]] = {
    "精材": ["精彩"],
    "外溢": ["外意"],
    "欣銓": ["新全", "新權", "新詮", "新銓"],
    "晶圓": ["金融"],  # 晶圓巨頭/晶圓代工 在半導體語境常被誤聽成「金融」
}

# EXPERIMENT (hardcode): phonetic cues — instruct by reading so every same-sound variant maps to
# the right characters at once. Maps glossary name -> pinyin. Injected when present.
PHONETIC_CUES: dict[str, str] = {
    "欣銓": "xīn quán",
}


def build_system_instruction(g: Glossary) -> str:
    """Assemble the system instruction. Covers both directions for the translate model:
    (1) Chinese ASR pinning — companies + jargon, mirroring glossary.py's phrasing, plus an
        in-context example sentence per company so the recognizer anchors on real usage; and
    (2) English translation — the canonical English name for each company/term, so the
    translated output uses the right proper nouns instead of guessing or transliterating."""
    parts: list[str] = ["這是一段台灣股市分析的廣播。"]
    if g.stocks:
        items = "、".join(f"{s.name}（{s.ticker}）" if s.ticker else s.name for s in g.stocks)
        parts.append("內容會提到下列台灣上市櫃公司，請正確辨識並轉寫這些公司的中文名稱與股票代號："
                     + items + "。")
    if g.terms:
        items = "、".join(t.term for t in g.terms)
        parts.append("內容也會提到下列股市／財經專有名詞，請正確辨識並轉寫：" + items + "。")

    # Example sentences: how each company/term tends to appear in the show, to anchor recognition.
    examples = [s.example.strip() for s in g.stocks if s.example.strip()]
    examples += [t.example.strip() for t in g.terms if t.example.strip()]
    present = {s.name for s in g.stocks} | {t.term for t in g.terms}
    for name, extra in EXTRA_EXAMPLES.items():
        if name in present:
            examples += extra
    if examples:
        parts.append("這些名稱與專有名詞在節目中的用法範例如下，請依此正確辨識："
                     + "、".join(f"「{e}」" for e in examples) + "。")

    # Phonetic cues: pin by reading, so every same-sound variant maps to the right characters.
    stock_by_name = {s.name: s for s in g.stocks}
    phon = []
    for name, pinyin in PHONETIC_CUES.items():
        if name not in present:
            continue
        s = stock_by_name.get(name)
        tag = f"（{s.english}，{s.ticker}）" if s and s.ticker else ""
        phon.append(f"「{name}」{tag}唸作 {pinyin}，凡聽到這個音請一律轉寫成「{name}」")
    if phon:
        parts.append("讀音提示：" + "；".join(phon) + "。")

    # Negative cues: name the exact wrong homophones to steer away from, for names that keep
    # slipping even with positive examples.
    cues = [f"{name}（不要寫成{'、'.join(f'「{w}」' for w in wrong)}）"
            for name, wrong in NEGATIVE_CUES.items() if name in present]
    if cues:
        parts.append("特別注意，下列名稱很容易被誤聽，請務必轉寫成正確寫法，切勿寫成括號內的錯誤詞："
                     + "、".join(cues) + "。")

    # English-translation guidance: name↔English mappings for the output side.
    pairs = [f"{s.name}＝{s.english}" for s in g.stocks if s.english]
    pairs += [f"{t.term}＝{t.english}" for t in g.terms if t.english]
    if pairs:
        parts.append("翻譯成英文時，請使用下列標準英文名稱與術語：" + "、".join(pairs) + "。")
    return "".join(parts)


def main():
    ap = argparse.ArgumentParser(description="Build an ASR glossary system instruction from a "
                                             "stock show's YouTube subtitle, via an LLM.")
    ap.add_argument("url", help="YouTube link of the stock show")
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"analyzer model (default {DEFAULT_MODEL})")
    ap.add_argument("--out", help="output .txt path (default results/<video_id>.si.txt)")
    ap.add_argument("--include", help="comma-separated terms to force into the glossary "
                                      "(e.g. 晶圓,製程), even if the model would skip them")
    args = ap.parse_args()
    must_include = [t.strip() for t in (args.include or "").split(",") if t.strip()]

    vid = fetch._video_id(args.url)
    out_path = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "results", f"{vid}.si.txt")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    print(f"fetching subtitle for {vid} ...", flush=True)
    sub_path = fetch.download_subtitle(args.url)
    print(f"using {os.path.basename(sub_path)}", flush=True)
    segs = fetch.parse_vtt(sub_path)
    subtitle = fetch.subtitle_text(segs)
    if not subtitle.strip():
        sys.exit("empty subtitle — no Chinese auto-captions found for this video")
    print(f"subtitle: {len(subtitle)} chars | analyzing with {args.model} ...", flush=True)

    g = analyze(subtitle, api_key=load_api_key(), model=args.model, must_include=must_include)
    g.stocks, report = validate_stocks(g.stocks)

    missing = [t for t in must_include if not any(t == term.term for term in g.terms)]
    if missing:
        print(f"  ⚠ requested terms not captured by the model: {missing}", flush=True)

    # Guarantee every stock carries an English name (the prompt requires one and the dict fills
    # confirmed tickers). If anything still slipped through, fall back to the Chinese name so the
    # field is never empty, and flag which ones for review.
    no_eng = [s.name for s in g.stocks if not s.english.strip()]
    for s in g.stocks:
        if not s.english.strip():
            s.english = s.name
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
    print(f"english: {len(g.stocks)}/{len(g.stocks)} stocks have a name", flush=True)
    if no_eng:
        print(f"  ⚠ model left {len(no_eng)} blank; filled from Chinese name: {no_eng}", flush=True)
    print("terms :", [t.term for t in g.terms], flush=True)
    print("\n----- system instruction -----")
    print(si)
    print("------------------------------")
    print(f"\nwrote {out_path}\nwrote {json_path}", flush=True)


if __name__ == "__main__":
    main()

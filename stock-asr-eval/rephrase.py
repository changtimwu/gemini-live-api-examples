"""Post-transcribe rephrasing: re-read a raw Chinese ASR transcript with a cheap text model
and fix domain-term mishearings using the glossary knowledge.

This is the Python port of the live app's "glossary arm" (gemini-live-asr-glossary-compare's
asr-config.ts: glossaryForRephrase + REPHRASE_INSTRUCTION, asr-bridge.ts: rephraseTurn). A model
that sees the whole phrase fixes true homophones (外溢/外意, 精材/精彩, 風測/封測, 金源電/京元電子)
far better than priming the streaming recognizer does — which is why correcting *after* the fact
beats baking the glossary into the recognizer's system instruction.

Two knobs differ from the live app, both because we score CER against the YouTube caption:
  - add_tickers defaults False. The live app appends " (2330)" after company names for the
    on-screen viewer; the ground-truth caption has no such annotations, so adding them would
    inflate CER and unfairly flatter ticker_recall. The eval wants pure correction. Pass
    add_tickers=True to reproduce the live behaviour.
  - we chunk the flat transcript ourselves (the live app rephrases per streamed turn). Each chunk
    is rephrased independently, mirroring the instruction's "輸出長度必須與輸入相近" expectation.
"""
import asyncio
import re

from google import genai
from google.genai import types

DEFAULT_REPHRASE_MODEL = "gemini-3.1-flash-lite"
REPHRASE_CHUNK_CHARS = 200    # pack sentences into ~this many chars per rephrase call
REPHRASE_CONCURRENCY = 4      # parallel flash-lite calls per transcript

# Sections of glossary_llm's system instruction the rephrase agent should NOT see. Mirrors
# asr-config.ts:glossaryForRephrase — the agent only needs the name/ticker/term lists plus the
# precise negative cues. The example sentences get regurgitated, the English map gets appended as
# glosses, and the phonetic ("always write X for this sound") rule is far too broad for a
# context-aware corrector and pulls in homophones.
_DROP_MARKERS = ("用法範例", "翻譯成英文", "讀音提示")


def glossary_for_rephrase(si_text: str) -> str:
    """Strip the example / English-map / phonetic sentences from a glossary_llm SI, keeping the
    name+ticker+term lists and the exact-wrong-form negative cues (特別注意…)."""
    parts = re.split(r"(?<=。)", si_text)  # split after each 。, keeping the delimiter
    return "".join(p for p in parts if not any(m in p for m in _DROP_MARKERS))


# Job 1 — correction only. Strict about NOT touching common words that merely sound like a name,
# and about wiping the leftover characters a mishearing leaves behind. No ticker annotation.
_CORRECT_ONLY = (
    "你是台灣股市節目的『中文逐字稿校正員』。下面的『詞彙知識』列出節目會提到的公司名稱、股票代號、"
    "專有名詞，以及常見的同音誤聽。請『只』針對我提供的這段逐字稿做一件事：\n"
    "把明顯被誤聽、其實是詞彙知識中公司／專有名詞的地方，整個誤聽的詞一併換成正確寫法——"
    "包含誤聽殘留的多餘字也要一起換掉，絕不可留下原本誤聽的任何字。只有在明確是在講某家公司／某個"
    "專有名詞的語境下才更正；一般常用詞（例如「新聞」「測試」「需求」「基本面」「漲停」「投信」"
    "「安全」「信心」）即使發音接近某個名稱，也絕不可改成公司名。\n"
    "範例（特別注意要清掉誤聽殘留字）：「台積電電它的」→「台積電它的」；「那欣銓權來講」→「那欣銓來講」；"
    "「日月光投控光族群」→「日月光投控族群」；「風測族群」→「封測族群」；「金源電」→「京元電子」。\n"
    "嚴格要求：除了上述更正，其他文字、語序、口語、標點一律原樣保留；不可加上股票代號或任何括號、"
    "不可翻譯、不可附加任何英文、不可解釋、不可自行造句、不可加入詞彙知識裡的例句或清單。"
    "詞彙知識只是查字典用的參考，絕對不可把輸入逐字稿中『沒有出現』的公司或名詞補進輸出。"
    "若這段只是片段或標點，直接原樣輸出，不要回問或說明。"
    "輸出長度必須與輸入相近，只輸出處理後的這段逐字稿本身。"
)

# Job 1 + Job 2 — also append " (代號)" after company names. Verbatim from the live app's
# REPHRASE_INSTRUCTION (asr-config.ts); use when you want the on-screen viewer experience rather
# than a fair-CER eval against the ticker-free caption.
_CORRECT_AND_TICKER = (
    "你是台灣股市節目的『中文逐字稿校正員』。下面的『詞彙知識』列出節目會提到的公司名稱、股票代號、"
    "專有名詞，以及常見的同音誤聽與讀音提示。請『只』針對我提供的這段逐字稿做兩件事：\n"
    "1. 把明顯被誤聽、其實是詞彙知識中公司／專有名詞的地方，整個誤聽的詞一併換成正確寫法——"
    "包含誤聽殘留的多餘字也要一起換掉，絕不可留下原本誤聽的任何字。只有在明確是在講某家公司／某個"
    "專有名詞的語境下才更正；一般常用詞（例如「新聞」「測試」「需求」「基本面」「漲停」「投信」"
    "「安全」「信心」）即使發音接近某個名稱，也絕不可改成公司名。讀音提示只在這種情況下參考使用。\n"
    "2. 只在『公司名稱』後面加上『 (代號)』，代號必須是詞彙知識『公司清單』裡該公司自己標註的數字代號，"
    "不可張冠李戴，例如「台積電」→「台積電 (2330)」、「日月光投控」→「日月光投控 (3711)」。"
    "詞彙知識『專有名詞清單』裡的詞（例如封測、矽光子、CoWoS、先進封裝、ASIC、外溢、台指期）都不是公司，"
    "一律不可加代號、也不可加任何括號——例如「封測族群」維持「封測族群」，"
    "絕不可變成「封測 (3711)」或「封測 (封測)」。英文縮寫與一般詞同樣不加。"
    "沒有代號的公司（例如外國公司）維持原樣。\n"
    "範例（特別注意要清掉誤聽殘留字）：「台積電電它的」→「台積電 (2330) 它的」；"
    "「那欣銓權來講」→「那欣銓 (3264) 來講」；「日月光投控光族群」→「日月光投控 (3711) 族群」。\n"
    "嚴格要求：除上述兩點外，其他文字、語序、口語、標點一律原樣保留；不可翻譯、不可附加任何英文、"
    "不可解釋、不可自行造句、不可加入詞彙知識裡的例句或清單。"
    "詞彙知識只是查字典用的參考，絕對不可把輸入逐字稿中『沒有出現』的公司或名詞補進輸出。"
    "若這段只是片段或標點，直接原樣輸出，不要回問或說明。"
    "輸出長度必須與輸入相近，只輸出處理後的這段逐字稿本身。"
)


def build_rephrase_instruction(si_text: str, add_tickers: bool = False) -> str:
    """Assemble the rephrase system instruction: the fixed corrector preamble + the trimmed
    glossary knowledge appended as a reference dictionary."""
    preamble = _CORRECT_AND_TICKER if add_tickers else _CORRECT_ONLY
    return preamble + "\n\n詞彙知識：\n" + glossary_for_rephrase(si_text)


def chunk_transcript(text: str, target: int = REPHRASE_CHUNK_CHARS) -> list[str]:
    """Split a flat transcript into ~target-char chunks on natural boundaries.

    Prefer sentence ends (。！？\\n), fall back to clause commas (，、) and spaces for unpunctuated
    ASR, and hard-window anything still longer than target so a single call never sees a runaway
    span. Independent chunks keep each rephrase call's input short, matching the instruction's
    'output length must be close to input' expectation and bounding the blast radius of any drift.
    """
    text = text.strip()
    if not text:
        return []
    units = re.split(r"(?<=[。！？\n])", text)            # sentence-ish units, delimiter kept
    units = [u for u in (s.strip() for s in units) if u]
    if len(units) <= 1:                                   # unpunctuated — split on clause marks
        units = [u for u in (s.strip() for s in re.split(r"(?<=[，、 ])", text)) if u] or [text]

    chunks: list[str] = []
    buf = ""
    for u in units:
        while len(u) > target:                            # a single unit longer than target
            head, u = u[:target], u[target:]
            if buf:
                chunks.append(buf); buf = ""
            chunks.append(head)
        if len(buf) + len(u) > target and buf:
            chunks.append(buf); buf = u
        else:
            buf += u
    if buf:
        chunks.append(buf)
    return chunks


async def _rephrase_chunk(client: genai.Client, model: str, instruction: str, chunk: str,
                          retries: int = 3) -> tuple[str, bool]:
    """Correct one chunk. Returns (text, ok); ok=False means every attempt failed and we fell back
    to the raw chunk (never drop content). The caller surfaces ok=False so a failed rephrase can't
    masquerade as 'rephrasing made no change' — critical when an API quota/credit error silently
    knocks out the whole pass."""
    config = types.GenerateContentConfig(
        system_instruction=instruction,
        temperature=0.0,
        thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW),
    )
    last_err = None
    for attempt in range(retries + 1):
        try:
            resp = await client.aio.models.generate_content(
                model=model, contents=chunk, config=config)
            out = (resp.text or "").strip()
            if out:
                return out, True
        except Exception as e:
            last_err = e
        if attempt < retries:
            await asyncio.sleep(min(20, 2 * 2 ** attempt))  # 2,4,8,16s backoff
    if last_err is not None:
        print(f"      rephrase chunk failed ({type(last_err).__name__}: {str(last_err)[:80]}); "
              f"kept raw", flush=True)
    return chunk, False  # leave this chunk uncorrected rather than lose it


async def rephrase_transcript(transcript: str, instruction: str, *, api_key: str,
                              model: str = DEFAULT_REPHRASE_MODEL,
                              target: int = REPHRASE_CHUNK_CHARS,
                              concurrency: int = REPHRASE_CONCURRENCY) -> tuple[str, int, int]:
    """Chunk a raw ASR transcript, rephrase every chunk concurrently, re-join in order.
    Returns (corrected_text, failed_chunks, total_chunks) — failed_chunks > 0 means the result is
    degraded (some/all chunks are raw passthrough), so the caller can flag it instead of trusting it."""
    chunks = chunk_transcript(transcript, target)
    if not chunks:
        return "", 0, 0
    client = genai.Client(api_key=api_key)
    sem = asyncio.Semaphore(concurrency)

    async def one(ch: str) -> tuple[str, bool]:
        async with sem:
            return await _rephrase_chunk(client, model, instruction, ch)

    results = await asyncio.gather(*(one(ch) for ch in chunks))
    failed = sum(1 for _, ok in results if not ok)
    return "".join(t for t, _ in results), failed, len(chunks)


# Trailing " (1234)" / "（1234）" annotations the add_tickers mode appends. Stripped before CER so
# even a ticker-annotated run is scored fairly against the ticker-free caption.
_TICKER_ANNOT = re.compile(r"\s*[（(]\d{4,6}[A-Z]?[)）]")


def strip_ticker_annotations(text: str) -> str:
    return _TICKER_ANNOT.sub("", text)


if __name__ == "__main__":
    import argparse
    import asyncio as _asyncio
    import os
    import sys

    from glossary_llm import load_api_key

    ap = argparse.ArgumentParser(description="Rephrase a raw ASR transcript with a glossary SI.")
    ap.add_argument("transcript", help="raw transcript .txt file (or - for stdin)")
    ap.add_argument("si", help="glossary system-instruction .txt file (glossary_llm output)")
    ap.add_argument("--model", default=DEFAULT_REPHRASE_MODEL)
    ap.add_argument("--add-tickers", action="store_true",
                    help="also append company ticker codes (live-app behaviour; hurts CER vs caption)")
    ap.add_argument("--chunk", type=int, default=REPHRASE_CHUNK_CHARS, help="chars per rephrase call")
    args = ap.parse_args()

    raw = sys.stdin.read() if args.transcript == "-" else open(args.transcript, encoding="utf-8").read()
    si = open(args.si, encoding="utf-8").read()
    instr = build_rephrase_instruction(si, add_tickers=args.add_tickers)
    out, failed, total = _asyncio.run(rephrase_transcript(raw, instr, api_key=load_api_key(),
                                                          model=args.model, target=args.chunk))
    if failed:
        print(f"# ⚠ {failed}/{total} chunks failed to rephrase (kept raw)", file=sys.stderr)
    print(out)

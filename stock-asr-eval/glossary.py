"""Find the stocks mentioned in the ground-truth subtitle and build a glossary
system-instruction for the B (with-SI) run.

Two extraction paths, unioned:
  1. 公司名（代號）  — explicit ticker in parens (subtitles that annotate codes).
  2. dictionary name scan — captions are pure speech (names, no codes), so we scan for
     known TWSE Chinese names. Returns {ticker: canonical_name}.
"""
import re

import stockdict

TICKER_IN_PARENS = re.compile(r"[（(](\d{4,6})[)）]")

# 2-char tickers that are also common Chinese words → exclude from name-scan to avoid FPs.
STOPLIST = {"材料", "大量", "卓越", "主流", "全國", "其他"}


def extract_terms(text: str, min_name_len: int = 2) -> dict[str, str]:
    terms: dict[str, str] = {}
    # 1) explicit codes, if any
    for ticker in TICKER_IN_PARENS.findall(text):
        name = stockdict.canonical(ticker)
        if name:
            terms.setdefault(ticker, name)
    # 2) dictionary name scan (captions are speech: names, no codes)
    for name, ticker in stockdict.BY_NAME.items():
        if len(name) >= min_name_len and name not in STOPLIST and name in text:
            terms.setdefault(ticker, name)
    # 3) drop substring FPs: a short name fully contained in another matched name
    #    (聯發⊂聯發科, 南亞⊂南亞科, 卓越⊂…台灣卓越, …)
    names = set(terms.values())
    return {t: n for t, n in terms.items()
            if not any(n != m and n in m for m in names)}


def build_system_instruction(terms: dict[str, str]) -> str | None:
    if not terms:
        return None
    items = "、".join(f"{name}（{ticker}）" for ticker, name in terms.items())
    return (
        "這是一段台灣股市分析的廣播。內容會提到下列台灣上市櫃公司，"
        "請正確辨識並轉寫這些公司的中文名稱與股票代號：" + items + "。"
    )

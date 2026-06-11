"""Scoring: CER + domain-term recall, with Trad/Simp + punctuation normalization.

The live model often transcribes in Simplified while TW subtitles are Traditional, so we
canonicalize BOTH sides to Simplified and strip punctuation/whitespace before scoring —
otherwise a pure script mismatch inflates CER without being a real error."""
import re

import jiwer
import opencc

import stockdict

_t2s = opencc.OpenCC("t2s")  # Traditional -> Simplified (canonical compare form)
_KEEP = re.compile(r"[^一-鿿A-Za-z0-9]")


def normalize(text: str) -> str:
    return _KEEP.sub("", _t2s.convert(text or ""))


def cer(ref: str, hyp: str) -> float:
    r, h = normalize(ref), normalize(hyp)
    if not r:
        return float("nan")
    return jiwer.cer(r, h)


def term_recall(terms: dict[str, str], hyp: str) -> dict:
    """terms: {ticker: canonical_name}. A name hits if ANY known alias appears in hyp."""
    h = normalize(hyp)
    per = {}
    for ticker, name in terms.items():
        names = stockdict.aliases(ticker) or [name]
        per[f"{name}({ticker})"] = {
            "name_hit": any(normalize(a) in h for a in names),
            "ticker_hit": ticker in h,
        }
    n = len(terms)
    return {
        "per_term": per,
        "name_recall": (sum(v["name_hit"] for v in per.values()) / n) if n else None,
        "ticker_recall": (sum(v["ticker_hit"] for v in per.values()) / n) if n else None,
        "n_terms": n,
    }

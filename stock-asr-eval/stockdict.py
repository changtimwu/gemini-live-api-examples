"""TWSE/TPEx Chinese-name <-> ticker dictionary, derived from stocks.json
(see build_stocks_map.py). Loaded once; used for clean glossary names + alias-aware recall."""
import json
import os

_PATH = os.path.join(os.path.dirname(__file__), "tw_stocks.json")
_d = json.load(open(_PATH, encoding="utf-8"))

BY_TICKER: dict[str, list[str]] = _d["by_ticker"]    # ticker -> [chinese names]
BY_NAME: dict[str, str] = _d["by_name"]              # chinese name -> ticker
BY_ENGLISH: dict[str, dict] = _d.get("by_english", {})  # ticker -> {"name", "abbr"}


def canonical(ticker: str) -> str | None:
    names = BY_TICKER.get(ticker)
    return names[0] if names else None


def aliases(ticker: str) -> list[str]:
    return BY_TICKER.get(ticker, [])


def english(ticker: str, prefer_abbr: bool = True) -> str | None:
    """English name for a ticker. Prefers the short internationally-recognized abbr (TSMC)
    over the long legal name when one exists; returns None if the dict has no English entry."""
    e = BY_ENGLISH.get(ticker)
    if not e:
        return None
    if prefer_abbr and e.get("abbr"):
        return e["abbr"]
    return e.get("name") or None

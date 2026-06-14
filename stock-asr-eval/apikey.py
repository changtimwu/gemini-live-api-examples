"""Shared Gemini API-key loader.

Lets a cloner put the key in one place: `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) in the
environment, or in a `.env.local` file (searched in this dir, then the repo root, then the
cwd). Every script in this package uses this, so the README's setup ("export ... or put it
in .env.local") holds for all of them, not just the LLM glossary step.
"""
import os
import sys


def load_api_key() -> str:
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
    sys.exit("no API key: set GEMINI_API_KEY (or GOOGLE_API_KEY), or put it in .env.local")

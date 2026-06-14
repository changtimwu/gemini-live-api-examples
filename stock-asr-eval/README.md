# stock-asr-eval

Offline harness for measuring how well a glossary helps Gemini transcribe a Taiwanese stock show.

The headline finding: **correcting the transcript *after* the fact (post-transcribe rephrasing)
beats baking the glossary into the recognizer.** A cheap text model that sees the whole phrase
fixes true homophones (`風測→封測`, `金源電→京元電子`, `新權→欣銓`, `精彩→精材`, `外溢/外意`,
`全職股→權值股`, `貴買→櫃買`) far better than priming the streaming ASR. This is the same
general-vs-glossary split the live app (`../gemini-live-asr-glossary-compare`) demonstrates on
screen; this harness measures it.

## The pipeline (one video → one result)

`pipeline.py` chains the four steps end-to-end for a single video — the unit of work:

1. **fetch** (`fetch.py`) — download the video's audio (16k mono PCM) + best Chinese auto-caption
   (the ground truth). Cache-first under `data/`.
2. **glossary** (`glossary_llm.py`) — read the caption with an LLM, surface companies (+ tickers,
   cross-checked against the TWSE/TPEx dictionary) and jargon, and build a glossary system
   instruction.
3. **transcribe + rephrase** — transcribe the audio with a *plain* recognizer prompt (no glossary
   baked in) — **batch `gemini-3.1-flash-lite`** by default, or the Live translate model via
   `--asr-model` — then post-correct that raw transcript chunk-by-chunk with `rephrase.py` using
   the glossary SI.
4. **score** (`score.py`) — CER + domain-term recall of **raw vs rephrased**, both against the
   caption. Same glossary-free transcription feeds both arms, so the rephrase pass is the only
   variable — exactly the effect we want to isolate.

```bash
# whole video
python pipeline.py --url https://youtu.be/<id>
# a segment (handy for smoke tests)
python pipeline.py --url https://youtu.be/<id> --start 0 --end 300
```

Writes `results/<id>_pipeline.json` incrementally (survives Ctrl-C; see the `status` field). Each
result carries the glossary, the system instruction, the raw + rephrased transcripts, and per-arm
CER / name-recall / ticker-recall. It also dumps `<id>_pipeline.raw.txt` / `.rephrased.txt` for
eyeballing, and records **`missed_stocks_rephrased`** — glossary stocks whose name never surfaced
even after rephrasing (the list to inspect when tuning the glossary).

## Result (validated)

Across **7 full 中集 episodes** of 錢線百分百 (339 min of audio, one trial each, translate-model ASR),
post-transcribe rephrasing lifts Taiwan stock-name recall from **~53% to ~96%** — every episode
improved (gains of 32–59 pp), zero regressions:

| metric | raw ASR | rephrased | Δ |
| --- | --- | --- | --- |
| **name_recall** | **53.3%** | **95.9%** | **+42.6 pp** |
| CER | 0.242 | 0.236 | +0.005 |
| ticker_recall | 9.1% | 15.3% | +6.1 pp |

CER moves little because these episodes carry a high baseline error rate from *general* mishearings
the glossary rephrase deliberately doesn't touch; the gain is concentrated exactly where it should
be — the stock names. Real corrections seen: `文業→文曄`, `大成剛→大成鋼`, `富邦美→富邦媒`,
`那雅科→南亞科`, `一頂→宜鼎`, `台子期→台指期`, `除席→除息`.

## Selecting a corpus

Not every episode of a finance show discusses individual stocks (many are macro/policy/personal-
finance). `classify.py` scans each cached caption against the TWSE/TPEx name dictionary and labels
episodes `stock` vs `non_stock` by stock-name density — so the eval corpus isn't diluted:

```bash
python playlist.py <playlist_url> --recent 100 --what subs   # cache captions
python classify.py <playlist_url>                            # -> results/classification.json + stock_urls.txt
python run_parallel.py --file results/stock_urls.txt --concurrency 3
```

It also tags full-version episodes vs the chopped per-segment clips (the playlist carries both),
which matters because the segments heavily overlap the full versions.

## Running many videos in parallel

`run_parallel.py` fans the pipeline out across a corpus — **one isolated `pipeline.py` subprocess
per video** (separate Live API sessions, independent crash domain, its own result file). It rolls
the per-video results up into a corpus summary.

```bash
# explicit URLs
python run_parallel.py <url> <url> <url>
# a file of URLs (one per line, # comments)
python run_parallel.py --file urls.txt
# the N newest videos of a playlist
python run_parallel.py --playlist <playlist_url> --recent 10 --concurrency 3
```

`--concurrency` caps how many videos run at once. The real ceiling is the **Gemini Live API's
concurrent-session quota**, so keep it modest (default 3). Writes `results/parallel_summary.json`
plus each `results/<id>_pipeline.json`.

All `pipeline.py` flags pass through: `--trials`, `--start/--end` (applied to every video),
`--analyzer-model`, `--rephrase-model`, `--add-tickers`, `--chunk`.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install google-genai pydantic jiwer opencc-python-reimplemented
# yt-dlp + ffmpeg on PATH (brew install yt-dlp ffmpeg)
export GEMINI_API_KEY=...        # or put it in .env.local (GEMINI_API_KEY=... / GOOGLE_API_KEY=...)
```

## Notes & knobs

- **Why correction-only by default.** The ground-truth caption has no `(2330)` annotations, so the
  default rephrase only *corrects* mishearings — it does not append tickers. `--add-tickers`
  reproduces the live app's on-screen behaviour (`台積電 (2330)`); those annotations are stripped
  before CER so the comparison stays fair either way (they do help `ticker_recall`).
- **ASR model.** Default is **batch `gemini-3.1-flash-lite`** (`transcribe_batch.py`): a plain
  `generateContent` call with the audio inline — no audio output, no streaming/session limits, far
  cheaper. On full episodes it also beat the Live translate model in our tests (e.g. `DqnT1cp5hRQ`:
  CER 0.11 vs 0.22, raw recall 80% vs 57%) — the Live model drops/garbles content over long audio.
  `--asr-model gemini-3.5-live-translate-preview` uses the higher-fidelity Live model the app runs.
  (`gemini-3.1-flash-live-preview` is wired but non-functional here — see `transcribe.py`.)
- **Chunking.** Live ASR splits audio into `--chunk` (default 480s ≈ 8 min) windows under the Live
  audio-session limit; batch ASR splits into ≤300s inline pieces under the request-size cap. Either
  way pieces are concatenated. The rephrase pass re-chunks the flat transcript into ~200-char pieces
  on sentence boundaries and corrects each independently (bounding drift).
- **Trials.** `--trials` (default 1) repeats transcribe+rephrase and averages; transcription has
  run-to-run variance, so >1 tightens the estimate at the cost of more API calls. For breadth,
  prefer more videos over more trials.
- **`eval_video.py` / `eval_full.py`** are the older A/B (no-SI vs glossary-SI-*in-the-recognizer*).
  `pipeline.py` is the post-transcribe-rephrase variant and the one to use going forward.

## Building blocks (usable standalone)

| script | does | CLI |
| --- | --- | --- |
| `fetch.py` | audio (PCM) + caption download, cache-first | `python fetch.py <url> [start] [end]` |
| `glossary_llm.py` | LLM glossary → system instruction `.txt` + `.glossary.json` | `python glossary_llm.py <url>` |
| `transcribe.py` | one PCM file → Chinese ASR via Live API (translate model) | `python transcribe.py <pcm> [si]` |
| `transcribe_batch.py` | one PCM file → Chinese ASR via batch gemini-3.1 (cheap, default) | `python transcribe_batch.py <pcm>` |
| `rephrase.py` | raw transcript + glossary SI → corrected transcript | `python rephrase.py <transcript.txt> <si.txt>` |
| `score.py` | CER + term recall (Trad/Simp + punctuation normalized) | (library) |
| `classify.py` | label playlist episodes stock vs non-stock by name density | `python classify.py <playlist_url>` |
| `playlist.py` | bulk-cache a playlist's audio/captions/metadata | `python playlist.py <playlist_url>` |

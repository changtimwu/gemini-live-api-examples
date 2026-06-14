# CLAUDE.md

## Shared `data/` and `results/` across git worktrees

`stock-asr-eval/data/` and `stock-asr-eval/results/` are large, **gitignored** caches — downloaded
audio (PCM/m4a), YouTube captions + `info.json`, and eval outputs/transcripts (often **>1 GB**).
Because they're gitignored, git does **not** share them across worktrees, and they are **deleted
when a worktree is removed**. This project uses many worktrees, so we keep **one** copy, shared by
every worktree via symlinks:

- no 1 GB+ duplication per worktree,
- the audio/caption cache is reused on every branch (re-runs skip download + decode), and
- nothing is lost when a feature worktree is thrown away.

**Canonical location — the main worktree** (the first entry of `git worktree list`, i.e.
`…/gemini-live-api-examples/stock-asr-eval/{data,results}`). The main worktree holds the real
directories; every *other* worktree symlinks to them.

**Set up a fresh worktree** — run inside it (NOT in the main worktree), before any `stock-asr-eval`
script:

```bash
MAIN=$(git worktree list --porcelain | awk '/^worktree /{print $2; exit}')
cd "$(git rev-parse --show-toplevel)/stock-asr-eval"
rm -rf data results                          # drop the empty/duplicate per-worktree dirs
ln -s "$MAIN/stock-asr-eval/data"    data
ln -s "$MAIN/stock-asr-eval/results" results
```

The symlinks sit under the gitignored paths, so they're never committed — each worktree makes its
own. (`.gitignore` ignores `data`/`results` whether they're real dirs or symlinks.)

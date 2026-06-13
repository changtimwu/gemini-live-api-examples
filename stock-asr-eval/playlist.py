"""Download & cache a YouTube playlist's per-video artifacts for the ASR eval corpus.

For each video in a playlist, cache three artifacts under
``data/playlists/<playlist_id>/`` (the playlist id is the ``list=`` URL param):

  - ``<id>.info.json``   metadata dump (yt-dlp ``--write-info-json``)
  - ``<id>.<lang>.vtt``  source auto-caption, first available of ``--sub-langs``
  - ``<id>.f<fmt>.m4a``  audio track by explicit format id (default 140 = m4a/AAC)

Everything is cache-first: an artifact already on disk is skipped (``--force`` to
refetch). ``--recent N`` pulls only the N newest videos (the front of the playlist —
this channel's playlist is newest-first; flat enumeration carries no usable upload
timestamp, so playlist order is the recency signal).

Subtitle resolution reads the per-video metadata, so requesting ``subs`` also
populates ``info.json`` (it's the caption index).

CLI:
  python playlist.py <playlist_url> [--recent N] [--what info,audio,subs]
         [--audio-format 140] [--sub-langs zh-TW,zh-Hant,zh-Hans,zh]
         [--list] [--force] [--data DIR]
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time

DATA = os.path.join(os.path.dirname(__file__), "data")
DEFAULT_AUDIO_FORMAT = "140"                          # m4a / AAC mp4a.40.2 ~129k 44.1kHz
# Preferred zh auto-caption codes, best first (shared source of truth; fetch.py aliases this).
# Many Taiwan channels publish the source as zh-TW (already Traditional); "<lang>-zh-TW" entries
# are machine re-translations, so rank below the real source.
DEFAULT_SUB_LANGS = ["zh-Hant", "zh-TW", "zh-Hant-zh-TW", "zh-Hans", "zh"]
WHAT_ALL = ("info", "audio", "subs")


def _playlist_id(url: str) -> str:
    m = re.search(r"[?&]list=([\w-]+)", url)
    return m.group(1) if m else re.sub(r"\W+", "_", url)[-16:]


def _video_url(vid: str) -> str:
    return f"https://www.youtube.com/watch?v={vid}"


def _yt(args: list[str]) -> None:
    """Run yt-dlp quietly; raise CalledProcessError on failure."""
    subprocess.run(["yt-dlp", "-q", "--no-warnings", *args], check=True)


# ---------------------------------------------------------------- enumeration

def enumerate_playlist(url: str, recent: int | None = None) -> dict:
    """Flat-enumerate a playlist (metadata only, no per-video network).

    With ``recent`` set, slices the front N via ``yt-dlp -I 1:N`` so we never
    page through the whole playlist. Returns ``{playlist_id, title, url, count, entries}``.
    """
    cmd = ["yt-dlp", "-q", "--no-warnings", "--flat-playlist", "-J"]
    if recent:
        cmd += ["-I", f"1:{recent}"]
    cmd.append(url)
    out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    data = json.loads(out)
    entries = [
        {"id": e.get("id"), "title": e.get("title"),
         "timestamp": e.get("timestamp"), "duration": e.get("duration")}
        for e in (data.get("entries") or []) if e.get("id")
    ]
    return {"playlist_id": _playlist_id(url), "title": data.get("title"),
            "url": url, "count": len(entries), "entries": entries}


def iter_playlist(url: str, recent: int | None = None) -> list[dict]:
    """Public API: the (id, title, duration) entries of a playlist, newest-first."""
    return enumerate_playlist(url, recent)["entries"]


def cached_audio(vid: str, fmt: str = DEFAULT_AUDIO_FORMAT, data_dir: str = DATA) -> str | None:
    """Path to a full audio track already cached for ``vid`` (any playlist), or None.

    Lets other tools (e.g. fetch.py) clip a segment from the full track instead of
    re-downloading; matches the requested format id so the codec is known.
    """
    hits = glob.glob(os.path.join(data_dir, "playlists", "*", f"{vid}.f{fmt}.*"))
    return hits[0] if hits else None


def cached_subtitle(vid: str, langs: list[str] | None = None,
                    data_dir: str = DATA) -> str | None:
    """Path to a cached source caption for ``vid`` (any playlist), highest-priority lang first."""
    for lang in (langs or DEFAULT_SUB_LANGS):
        hits = glob.glob(os.path.join(data_dir, "playlists", "*", f"{vid}.{lang}.vtt"))
        if hits:
            return hits[0]
    return None


# --------------------------------------------------------------- per-artifact

def ensure_info(pdir: str, vid: str, force: bool = False) -> str | None:
    """Cache ``<id>.info.json``. Returns its path (or None on failure)."""
    path = os.path.join(pdir, f"{vid}.info.json")
    if os.path.exists(path) and not force:
        return path
    _yt(["--skip-download", "--write-info-json",
         "-o", os.path.join(pdir, "%(id)s.%(ext)s"), _video_url(vid)])
    return path if os.path.exists(path) else None


def ensure_audio(pdir: str, vid: str, fmt: str, force: bool = False) -> str | None:
    """Cache the audio track at format id ``fmt`` as ``<id>.f<fmt>.<ext>``."""
    hits = glob.glob(os.path.join(pdir, f"{vid}.f{fmt}.*"))
    if hits and not force:
        return hits[0]
    _yt(["-f", fmt, "-o", os.path.join(pdir, f"{vid}.f{fmt}.%(ext)s"), _video_url(vid)])
    hits = glob.glob(os.path.join(pdir, f"{vid}.f{fmt}.*"))
    return hits[0] if hits else None


def _auto_caption_langs(info_path: str | None) -> set[str]:
    if not info_path or not os.path.exists(info_path):
        return set()
    with open(info_path, encoding="utf-8") as f:
        info = json.load(f)
    # only automatic captions — manual/uploaded subs are out of scope (v1)
    return set((info.get("automatic_captions") or {}).keys())


def ensure_subs(pdir: str, vid: str, langs: list[str], force: bool = False,
                info_path: str | None = None) -> str | None:
    """Cache the source auto-caption as ``<id>.<lang>.vtt``.

    Picks the first language in ``langs`` that the video actually exposes
    (read from ``info.json``); falls back to letting yt-dlp try the whole list.
    Returns the highest-priority ``.vtt`` present afterwards.
    """
    if not force:
        for lang in langs:
            p = os.path.join(pdir, f"{vid}.{lang}.vtt")
            if os.path.exists(p):
                return p
    avail = _auto_caption_langs(info_path)
    chosen = next((lang for lang in langs if lang in avail), None)
    target = chosen or ",".join(langs)
    _yt(["--skip-download", "--write-auto-subs", "--sub-langs", target,
         "--sub-format", "vtt", "-o", os.path.join(pdir, "%(id)s.%(ext)s"), _video_url(vid)])
    for lang in langs:
        p = os.path.join(pdir, f"{vid}.{lang}.vtt")
        if os.path.exists(p):
            return p
    return None


def ensure_video(pdir: str, vid: str, what=WHAT_ALL,
                 audio_format: str = DEFAULT_AUDIO_FORMAT,
                 sub_langs: list[str] | None = None, force: bool = False) -> dict:
    """Cache the requested artifacts for one video. Cache-first; ``force`` refetches."""
    sub_langs = sub_langs or DEFAULT_SUB_LANGS
    os.makedirs(pdir, exist_ok=True)
    res: dict = {"id": vid, "info": None, "audio": None, "subs": None}
    info_path = None
    if "info" in what or "subs" in what:                 # subs needs the caption index
        info_path = ensure_info(pdir, vid, force=force)
        res["info"] = info_path
    if "audio" in what:
        res["audio"] = ensure_audio(pdir, vid, audio_format, force=force)
    if "subs" in what:
        res["subs"] = ensure_subs(pdir, vid, sub_langs, force=force, info_path=info_path)
    return res


# -------------------------------------------------------------------- driver

def download_playlist(url: str, *, recent: int | None = None, what=WHAT_ALL,
                      audio_format: str = DEFAULT_AUDIO_FORMAT,
                      sub_langs: list[str] | None = None, force: bool = False,
                      data_dir: str = DATA) -> dict:
    sub_langs = sub_langs or DEFAULT_SUB_LANGS
    meta = enumerate_playlist(url, recent)
    pdir = os.path.join(data_dir, "playlists", meta["playlist_id"])
    os.makedirs(pdir, exist_ok=True)
    with open(os.path.join(pdir, "index.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    results = []
    n = meta["count"]
    for i, e in enumerate(meta["entries"], 1):
        vid = e["id"]
        print(f"[{i}/{n}] {vid}  {(e.get('title') or '')[:50]}", flush=True)
        try:
            results.append(ensure_video(pdir, vid, what, audio_format, sub_langs, force))
        except subprocess.CalledProcessError as ex:
            print(f"   ! failed: {ex}", file=sys.stderr)
            results.append({"id": vid, "error": str(ex)})
    return {"playlist_dir": pdir, "results": results}


def print_index(meta: dict) -> None:
    print(f"# {meta.get('title')}  ({meta['count']} videos)  [{meta['playlist_id']}]")
    for i, e in enumerate(meta["entries"], 1):
        ts = e.get("timestamp")
        date = time.strftime("%Y-%m-%d", time.gmtime(ts)) if ts else "    -     "
        mins = f"{int(e.get('duration') or 0) // 60:>3}m"
        print(f"{i:>4}  {e['id']}  {date}  {mins}  {(e.get('title') or '')[:60]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Cache YouTube playlist artifacts (info/audio/subs).")
    ap.add_argument("url", help="YouTube playlist URL")
    ap.add_argument("--recent", type=int, help="only the N newest videos (front of playlist)")
    ap.add_argument("--what", default="info,audio,subs",
                    help="comma list of artifacts: info,audio,subs (default all)")
    ap.add_argument("--audio-format", default=DEFAULT_AUDIO_FORMAT,
                    help=f"yt-dlp audio format id (default {DEFAULT_AUDIO_FORMAT} = m4a/AAC)")
    ap.add_argument("--sub-langs", default=",".join(DEFAULT_SUB_LANGS),
                    help="ordered source-caption preference; first available wins")
    ap.add_argument("--list", action="store_true",
                    help="enumerate only; print the index and download nothing")
    ap.add_argument("--force", action="store_true", help="re-download even if cached")
    ap.add_argument("--data", default=DATA, help="data root dir (default ./data)")
    args = ap.parse_args()

    what = [w.strip() for w in args.what.split(",") if w.strip()]
    bad = [w for w in what if w not in WHAT_ALL]
    if bad:
        ap.error(f"unknown --what values {bad}; allowed: {', '.join(WHAT_ALL)}")
    sub_langs = [s.strip() for s in args.sub_langs.split(",") if s.strip()]

    if args.list:
        print_index(enumerate_playlist(args.url, args.recent))
        sys.exit(0)

    res = download_playlist(args.url, recent=args.recent, what=what,
                            audio_format=args.audio_format, sub_langs=sub_langs,
                            force=args.force, data_dir=args.data)
    ok = sum(1 for r in res["results"] if not r.get("error"))
    print(f"\ndone: {ok}/{len(res['results'])} videos OK -> {res['playlist_dir']}")

"""Fetch a YouTube video's audio (16k mono PCM) + ground-truth subtitle text.

Ground truth = the best available Chinese auto-caption (tries zh-Hant, zh-TW, ... best-first;
not every channel exposes zh-Hant). Supports a time range so long videos can be evaluated in
segments. Caches downloads under data/, and reuses any full audio/caption already fetched by
the playlist downloader (data/playlists/*/) so segments are clipped locally, not re-downloaded.

CLI:  python fetch.py <url> [start_sec] [end_sec]
"""
import os
import re
import subprocess
import sys

import playlist

SAMPLE_RATE = 16000
AUDIO_FORMAT = "140"                                  # m4a / AAC — matches playlist.py's cache
ZH_SUB_LANGS = playlist.DEFAULT_SUB_LANGS             # shared zh caption priority (best first)
DATA = os.path.join(os.path.dirname(__file__), "data")
TS = re.compile(r"(\d\d):(\d\d):(\d\d)\.(\d\d\d)\s+-->\s+(\d\d):(\d\d):(\d\d)\.(\d\d\d)")
TAG = re.compile(r"<[^>]+>")


def _video_id(url: str) -> str:
    m = re.search(r"(?:v=|youtu\.be/)([\w-]{11})", url)
    return m.group(1) if m else re.sub(r"\W+", "_", url)[-11:]


def _hms(sec: float) -> str:
    sec = int(sec)
    return f"{sec//3600:02d}:{(sec%3600)//60:02d}:{sec%60:02d}"


def parse_vtt(path: str) -> list[tuple[float, float, str]]:
    segs: list[tuple[float, float, str]] = []
    cur = None
    for line in open(path, encoding="utf-8"):
        m = TS.search(line)
        if m:
            a = int(m[1]) * 3600 + int(m[2]) * 60 + int(m[3]) + int(m[4]) / 1000
            b = int(m[5]) * 3600 + int(m[6]) * 60 + int(m[7]) + int(m[8]) / 1000
            cur = [a, b, ""]
            segs.append(cur)
        elif cur is not None:
            t = TAG.sub("", line).strip()
            if t and t not in ("WEBVTT",):
                cur[2] = (cur[2] + " " + t).strip() if cur[2] else t
    # drop empties + consecutive duplicates (YouTube rolling captions)
    out: list[tuple[float, float, str]] = []
    for a, b, t in segs:
        if not t:
            continue
        if out and out[-1][2] == t:
            continue
        out.append((a, b, t))
    return out


def subtitle_text(segs, t0: float | None = None, t1: float | None = None) -> str:
    parts = [t for a, b, t in segs
             if (t0 is None or b > t0) and (t1 is None or a < t1)]
    return " ".join(parts)


def _flat_cached_sub(vid: str) -> str | None:
    for lang in ZH_SUB_LANGS:
        p = os.path.join(DATA, f"{vid}.{lang}.vtt")
        if os.path.exists(p):
            return p
    return None


def download_subtitle(url: str) -> str:
    """Download the best available Chinese auto-caption to data/, returning its path.

    Tries several zh codes best-first and reuses any already-cached track — including one
    fetched by the playlist downloader (data/playlists/*/).
    """
    vid = _video_id(url)
    hit = playlist.cached_subtitle(vid, ZH_SUB_LANGS, DATA) or _flat_cached_sub(vid)
    if hit:
        return hit
    os.makedirs(DATA, exist_ok=True)
    # One language at a time, best first: requesting all langs at once makes yt-dlp pull every
    # matching (incl. machine-translated) track and trip YouTube's 429 rate limit.
    for lang in ZH_SUB_LANGS:
        subprocess.run(["yt-dlp", "-q", "--no-warnings", "--skip-download", "--write-auto-subs",
                        "--sub-langs", lang, "--sub-format", "vtt",
                        "-o", os.path.join(DATA, "%(id)s.%(ext)s"), url],
                       capture_output=True)  # tolerate per-lang failure (missing track / 429)
        p = os.path.join(DATA, f"{vid}.{lang}.vtt")
        if os.path.exists(p):
            return p
    raise RuntimeError(f"no Chinese auto-captions found for {url} "
                       f"(tried {', '.join(ZH_SUB_LANGS)}); check `yt-dlp --list-subs {url}`")


def _to_pcm(src: str, pcm: str, t0: float | None = None, t1: float | None = None) -> None:
    """Decode src -> 16k mono s16le PCM, optionally clipping to [t0, t1] seconds."""
    cmd = ["ffmpeg", "-y"]
    if t0:
        cmd += ["-ss", _hms(t0)]
    if t1 is not None:
        cmd += ["-t", _hms(t1 - (t0 or 0))]
    cmd += ["-i", src, "-ar", str(SAMPLE_RATE), "-ac", "1", "-f", "s16le", pcm]
    subprocess.run(cmd, check=True, capture_output=True)


def download_audio_pcm(url: str, t0: float | None = None, t1: float | None = None) -> bytes:
    vid = _video_id(url)
    tag = f"{int(t0 or 0)}_{int(t1) if t1 else 'end'}"
    pcm = os.path.join(DATA, f"{vid}.{tag}.pcm")
    if not os.path.exists(pcm):
        os.makedirs(DATA, exist_ok=True)
        full = playlist.cached_audio(vid, AUDIO_FORMAT, DATA)
        if full:
            # clip the requested window from the already-downloaded full track (no network)
            _to_pcm(full, pcm, t0, t1)
        else:
            src = os.path.join(DATA, f"{vid}.{tag}.m4a")
            cmd = ["yt-dlp", "-q", "--no-warnings", "-f", AUDIO_FORMAT, "-o", src]
            if t0 is not None or t1 is not None:
                cmd += ["--download-sections", f"*{_hms(t0 or 0)}-{_hms(t1) if t1 else '99:59:59'}"]
            subprocess.run(cmd + [url], check=True)
            _to_pcm(src, pcm)  # src already clipped by --download-sections
    with open(pcm, "rb") as f:
        return f.read()


def fetch(url: str, t0: float | None = None, t1: float | None = None) -> tuple[bytes, str]:
    sub = download_subtitle(url)
    segs = parse_vtt(sub)
    gt = subtitle_text(segs, t0, t1)
    pcm = download_audio_pcm(url, t0, t1)
    return pcm, gt


if __name__ == "__main__":
    url = sys.argv[1]
    t0 = float(sys.argv[2]) if len(sys.argv) > 2 else None
    t1 = float(sys.argv[3]) if len(sys.argv) > 3 else None
    pcm, gt = fetch(url, t0, t1)
    print(f"audio: {len(pcm)/(SAMPLE_RATE*2):.1f}s   gt chars: {len(gt)}")
    print("gt sample:", gt[:200])

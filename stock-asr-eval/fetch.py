"""Fetch a YouTube video's audio (16k mono PCM) + ground-truth subtitle text.

Ground truth = YouTube auto-caption (zh-Hant). Supports a time range so long videos can be
evaluated in segments. Caches downloads under data/.

CLI:  python fetch.py <url> [start_sec] [end_sec]
"""
import os
import re
import subprocess
import sys

SAMPLE_RATE = 16000
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


def download_subtitle(url: str) -> str:
    vid = _video_id(url)
    path = os.path.join(DATA, f"{vid}.zh-Hant.vtt")
    if not os.path.exists(path):
        subprocess.run(["yt-dlp", "-q", "--no-warnings", "--skip-download",
                        "--write-auto-subs", "--sub-langs", "zh-Hant", "--sub-format", "vtt",
                        "-o", os.path.join(DATA, "%(id)s.%(ext)s"), url], check=True)
    return path


def download_audio_pcm(url: str, t0: float | None = None, t1: float | None = None) -> bytes:
    vid = _video_id(url)
    tag = f"{int(t0 or 0)}_{int(t1) if t1 else 'end'}"
    src = os.path.join(DATA, f"{vid}.{tag}.m4a")
    pcm = os.path.join(DATA, f"{vid}.{tag}.pcm")
    if not os.path.exists(pcm):
        cmd = ["yt-dlp", "-q", "--no-warnings", "-f", "bestaudio", "-o", src]
        if t0 is not None or t1 is not None:
            cmd += ["--download-sections", f"*{_hms(t0 or 0)}-{_hms(t1) if t1 else '99:59:59'}"]
        subprocess.run(cmd + [url], check=True)
        subprocess.run(["ffmpeg", "-y", "-i", src, "-ar", str(SAMPLE_RATE), "-ac", "1",
                        "-f", "s16le", pcm], check=True, capture_output=True)
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

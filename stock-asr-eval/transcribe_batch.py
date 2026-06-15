"""Batch (non-Live) ASR: transcribe audio with an ordinary gemini-3.1 generateContent call.

Unlike the Live path (transcribe.py), this sends the audio as a normal request and reads the TEXT
response — no audio output, no WebSocket/session limits, no partial-transcript variance. Much
cheaper for bulk transcription (you pay audio-input + cheap text-output tokens; the Live translate
model's ~NT$1/min audio output is the line this avoids). It measures a *batch* recognizer, not the
live app's streaming model — fine for the rephrasing eval, which only needs a Chinese ASR to correct.

Audio goes inline as WAV; long input is split into <= MAX_INLINE_SECS pieces (the request has a
~20MB inline cap) and the per-piece transcripts are concatenated.

Usage: python transcribe_batch.py <pcm_file_16k_mono_s16le> [--model gemini-3.1-flash]
"""
import asyncio
import struct

from google import genai
from google.genai import types

DEFAULT_BATCH_MODEL = "gemini-3.1-flash-lite"   # the available 3.1 batch model (no plain -flash exists)
SAMPLE_RATE = 16000
# 300s * 16k * 2B = 9.6MB raw -> ~12.8MB base64 on the wire, comfortably under the ~20MB request cap.
MAX_INLINE_SECS = 300
CONCURRENCY = 4

TRANSCRIBE_PROMPT = (
    "請把這段台灣股市分析節目的音訊逐字轉寫成繁體中文逐字稿，盡量正確辨識所有內容。"
    "只輸出逐字稿文字本身，不要加時間戳、說話者標記、標題或任何說明，也不要翻譯。"
)


def is_batch_model(model: str) -> bool:
    """A non-Live model id (no 'live'/'translate') → use the batch generateContent path."""
    return "live" not in model and "translate" not in model


def _wav(pcm: bytes, rate: int = SAMPLE_RATE) -> bytes:
    """Wrap raw 16k mono s16le PCM in a minimal WAV container (the API needs a real audio format)."""
    n = len(pcm)
    return (b"RIFF" + struct.pack("<I", 36 + n) + b"WAVEfmt "
            + struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16)
            + b"data" + struct.pack("<I", n) + pcm)


async def _transcribe_piece(client, model, sysi, piece: bytes, retries: int = 3) -> str:
    cfg = types.GenerateContentConfig(
        system_instruction=sysi, temperature=0.0, max_output_tokens=32768,
        thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW))
    contents = [TRANSCRIBE_PROMPT, types.Part.from_bytes(data=_wav(piece), mime_type="audio/wav")]
    for attempt in range(retries + 1):
        try:
            resp = await client.aio.models.generate_content(model=model, contents=contents, config=cfg)
            t = (resp.text or "").strip()
            if t:
                return t
        except Exception as e:
            print(f"      batch piece error ({type(e).__name__}: {str(e)[:70]}); retry {attempt+1}/{retries}", flush=True)
        if attempt < retries:
            await asyncio.sleep(min(30, 3 * 2 ** attempt))
    return ""  # this piece failed; caller sees a short/empty result and can mark the trial invalid


async def batch_transcribe_pcm(pcm: bytes, *, api_key: str, model: str = DEFAULT_BATCH_MODEL,
                               system_instruction: str | None = None,
                               max_inline_secs: int = MAX_INLINE_SECS) -> dict:
    """Transcribe a whole (possibly long) PCM track: split into inline-sized WAV pieces, transcribe
    each concurrently, join in order. Shape matches transcribe_pcm (translation_en unused)."""
    client = genai.Client(api_key=api_key)
    step = SAMPLE_RATE * 2 * max_inline_secs
    pieces = [pcm[i:i + step] for i in range(0, len(pcm), step)]
    sysi = types.Content(parts=[types.Part(text=system_instruction)]) if system_instruction else None
    sem = asyncio.Semaphore(CONCURRENCY)

    async def one(p):
        async with sem:
            return await _transcribe_piece(client, model, sysi, p)

    parts = await asyncio.gather(*(one(p) for p in pieces))
    return {"source_zh": " ".join(x for x in parts if x).strip(), "translation_en": ""}


if __name__ == "__main__":
    import argparse
    import os
    import sys
    ap = argparse.ArgumentParser()
    ap.add_argument("pcm", help="16k mono s16le PCM file")
    ap.add_argument("--model", default=DEFAULT_BATCH_MODEL)
    args = ap.parse_args()
    from apikey import load_api_key
    key = load_api_key()
    pcm = open(args.pcm, "rb").read()
    print(f"audio {len(pcm)/(SAMPLE_RATE*2):.0f}s  model={args.model}", flush=True)
    res = asyncio.run(batch_transcribe_pcm(pcm, api_key=key, model=args.model,
                                           system_instruction="這是一段台灣股市分析的廣播。"))
    print(res["source_zh"])

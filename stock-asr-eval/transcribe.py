"""Capture the SOURCE-language (Chinese) transcription from gemini-3.5-live-translate-preview.

We stream raw 16kHz mono PCM into a Live API session (translation mode) and collect
`input_transcription` — the model's ASR of what it heard. The translated (output) text/audio
is ignored for scoring (no English ground truth). Optionally pass a system_instruction
(glossary) for the A/B test.

The translate model streams translated audio for a long time after the input ends, so we
stop when the *input* transcription has been quiet for a while (not when output stops).

Usage: python transcribe.py <pcm_file_16k_mono_s16le> [system_instruction_text]
"""
import asyncio
import os
import sys
import time

from google import genai
from google.genai import types

MODEL = "gemini-3.5-live-translate-preview"
SAMPLE_RATE = 16000
CHUNK_BYTES = 3200       # 100 ms @ 16k mono s16le
SEND_SLEEP = 0.04        # ~2.5x real time
INPUT_IDLE = 8.0         # stop once input transcription is quiet this long (post-send)
POLL = 2.0               # receive poll granularity


async def transcribe_pcm(pcm: bytes, *, api_key: str, model: str = MODEL,
                         system_instruction: str | None = None,
                         send_sleep: float = SEND_SLEEP, input_idle: float = INPUT_IDLE,
                         target_language: str = "en") -> dict:
    client = genai.Client(api_key=api_key, http_options=types.HttpOptions(api_version="v1beta"))
    cfg_kwargs = dict(
        response_modalities=[types.Modality.AUDIO],
        translation_config=types.TranslationConfig(
            target_language_code=target_language, echo_target_language=True),
        input_audio_transcription=types.AudioTranscriptionConfig(),
        output_audio_transcription=types.AudioTranscriptionConfig(),
    )
    if system_instruction:
        cfg_kwargs["system_instruction"] = types.Content(parts=[types.Part(text=system_instruction)])
    config = types.LiveConnectConfig(**cfg_kwargs)

    src_parts: list[str] = []
    out_parts: list[str] = []
    audio_secs = len(pcm) / (SAMPLE_RATE * 2)
    hard_cap = audio_secs * (send_sleep / 0.1) + 90   # streaming time + generous margin

    async with client.aio.live.connect(model=model, config=config) as session:
        async def send_audio():
            for i in range(0, len(pcm), CHUNK_BYTES):
                await session.send_realtime_input(
                    audio=types.Blob(data=pcm[i:i + CHUNK_BYTES],
                                     mime_type=f"audio/pcm;rate={SAMPLE_RATE}"))
                await asyncio.sleep(send_sleep)
            await session.send_realtime_input(audio_stream_end=True)

        send_task = asyncio.create_task(send_audio())
        gen = session.receive()
        started = time.monotonic()
        last_input = time.monotonic()
        while True:
            if send_task.done() and (time.monotonic() - last_input) > input_idle:
                break
            if time.monotonic() - started > hard_cap:
                break
            try:
                resp = await asyncio.wait_for(gen.__anext__(), timeout=POLL)
            except asyncio.TimeoutError:
                continue
            except StopAsyncIteration:
                break
            sc = getattr(resp, "server_content", None)
            if not sc:
                continue
            if sc.input_transcription and sc.input_transcription.text:
                src_parts.append(sc.input_transcription.text)
                last_input = time.monotonic()
            if sc.output_transcription and sc.output_transcription.text:
                out_parts.append(sc.output_transcription.text)
        if not send_task.done():
            send_task.cancel()
        elif send_task.exception():
            raise send_task.exception()

    return {"source_zh": "".join(src_parts).strip(),
            "translation_en": "".join(out_parts).strip()}


def _read_key() -> str:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        sys.exit("set GEMINI_API_KEY")
    return key


if __name__ == "__main__":
    pcm_path = sys.argv[1]
    si = sys.argv[2] if len(sys.argv) > 2 else None
    with open(pcm_path, "rb") as f:
        pcm = f.read()
    print(f"audio: {pcm_path}  ({len(pcm)/(SAMPLE_RATE*2):.1f}s)  SI={'yes' if si else 'no'}", flush=True)
    res = asyncio.run(transcribe_pcm(pcm, api_key=_read_key(), system_instruction=si))
    print("SOURCE (zh):", res["source_zh"])
    print("TRANSLATION (en):", res["translation_en"])

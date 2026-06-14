"""Capture the SOURCE-language (Chinese) ASR from a Gemini Live model.

We stream raw 16kHz mono PCM into a Live session and collect `input_transcription` — the model's
ASR of what it heard. Two model paths (`input_transcription` is read identically for both):

  - gemini-3.5-live-translate-preview (translate, DEFAULT) — AUDIO response + translationConfig, as
    the live app uses. Emits an English translation (output_transcription), kept for reference only.
    This is the only path that reliably transcribes in our harness today.
  - gemini-3.1-flash-live-preview (flash) — intended as a cheaper recognizer, but it does NOT work
    here: it rejects a TEXT response (native-audio model), and with an AUDIO response the Live
    session closes ~3s in with zero transcription (tried v1beta/v1alpha, ±realtimeInputConfig,
    several send rates). Left selectable via --model in case it's fixed/enabled later. See issue #19.

Both paths use an AUDIO response; only the translate model adds translationConfig.

Usage: python transcribe.py <pcm_file_16k_mono_s16le> [system_instruction_text] [--model M] [--out path]
The transcript is saved to results/<pcm_stem>.transcript.json by default (--out '' to skip).
"""
import asyncio
import json
import os
import sys
import time

from google import genai
from google.genai import types

MODEL = "gemini-3.5-live-translate-preview"           # default: the only model that works here
FLASH_MODEL = "gemini-3.1-flash-live-preview"          # cheaper in theory, but non-functional (see docstring)
SAMPLE_RATE = 16000
CHUNK_BYTES = 3200       # 100 ms @ 16k mono s16le
SEND_SLEEP = 0.04        # ~2.5x real time
INPUT_IDLE = 8.0         # stop once input transcription is quiet this long (post-send)
POLL = 2.0               # receive poll granularity


def is_translate_model(model: str) -> bool:
    return "translate" in model


async def transcribe_pcm(pcm: bytes, *, api_key: str, model: str = MODEL,
                         system_instruction: str | None = None,
                         send_sleep: float = SEND_SLEEP, input_idle: float = INPUT_IDLE,
                         target_language: str = "en") -> dict:
    client = genai.Client(api_key=api_key, http_options=types.HttpOptions(api_version="v1beta"))
    if is_translate_model(model):
        cfg_kwargs = dict(
            response_modalities=[types.Modality.AUDIO],
            translation_config=types.TranslationConfig(
                target_language_code=target_language, echo_target_language=True),
            input_audio_transcription=types.AudioTranscriptionConfig(),
            output_audio_transcription=types.AudioTranscriptionConfig(),
        )
    else:
        # Plain live model (e.g. flash-live): native-audio, so AUDIO is the only supported response
        # modality (TEXT is rejected). We read the Chinese ASR from input_audio_transcription and
        # ignore the model's spoken turn. No translationConfig, so it won't translate.
        cfg_kwargs = dict(
            response_modalities=[types.Modality.AUDIO],
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
    from apikey import load_api_key
    return load_api_key()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("pcm", help="16k mono s16le PCM file")
    ap.add_argument("si", nargs="?", default=None, help="optional system instruction (glossary) text")
    ap.add_argument("--model", default=MODEL, help=f"Live model (default {MODEL}; {FLASH_MODEL} is non-functional)")
    ap.add_argument("--out", help="write transcript JSON here "
                                  "(default results/<pcm_stem>.transcript.json; use '' to skip)")
    args = ap.parse_args()
    with open(args.pcm, "rb") as f:
        pcm = f.read()
    print(f"audio: {args.pcm}  ({len(pcm)/(SAMPLE_RATE*2):.1f}s)  model={args.model}  "
          f"SI={'yes' if args.si else 'no'}", flush=True)
    res = asyncio.run(transcribe_pcm(pcm, api_key=_read_key(), model=args.model, system_instruction=args.si))
    print("SOURCE (zh):", res["source_zh"])
    print("TRANSLATION (en):", res["translation_en"])

    # persist by default — a one-off transcription is otherwise lost the moment it scrolls off
    out_path = args.out if args.out is not None else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "results",
        os.path.splitext(os.path.basename(args.pcm))[0] + ".transcript.json")
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        json.dump({"pcm": args.pcm, "system_instruction": args.si, **res},
                  open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print("wrote", out_path)

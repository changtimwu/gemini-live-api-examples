# ASR Glossary Compare

Side-by-side live transcription that shows what a **domain glossary** does to
speech recognition. You feed in audio — your **browser mic** or the **playback
of a browser tab** (a YouTube video, a meeting, a podcast) — and the same audio
is fed to **two** Gemini Live sessions at once:

- **Left — With glossary**: system instruction pins a set of TWSE-listed company
  names and tickers (台積電 / 2330, 宜鼎 / 5289, …).
- **Right — General**: same domain framing, but no specific company list.

Both Chinese transcripts append to their scrollable column in real time. A
checkbox toggles whether the English translation is shown right under each
Chinese line.

This is the live counterpart to the offline A/B harness in
[`../stock-asr-eval`](../stock-asr-eval) — same model, same glossary idea.

## How it works

```
Browser mic ─┐
             ├─publishes audio──▶ LiveKit room
Browser tab ─┘                       │
                 ┌───────────────────┴───────────────────┐
                 ▼                                         ▼
        AsrBridge "glossary"                       AsrBridge "general"
        (glossary systemInstruction)               (general systemInstruction)
                 │                                         │
                 ▼                                         ▼
        Gemini Live (translate)                    Gemini Live (translate)
        input_transcription  → zh                  input_transcription  → zh
        output_transcription → en                  output_transcription → en
                 │                                         │
                 └──────────── publishData ───────────────┘
                                     │
                                     ▼
                Browser appends transcripts to two columns
```

Each bridge is a server-side LiveKit participant that joins the room as a bot
(`asr-glossary` / `asr-general`), subscribes to whatever audio track you publish
(mic or tab), streams PCM to its own Gemini Live WebSocket, and publishes
transcripts back over the reliable data channel. The bridge is source-agnostic —
tab audio is purely a client-side capture detail, so the server is unchanged. The model is `gemini-3.5-live-translate-preview`, which emits both the
source (Chinese) ASR and the (English) translation — the translation is always
produced; the checkbox only controls whether it's displayed.

## Prerequisites

- Node.js 18+
- A [Gemini API key](https://aistudio.google.com/apikey)
- A running LiveKit server (local is fine — see below)

> Browser mic capture requires a secure context. `http://localhost` counts, so
> local dev works out of the box. Over the LAN you'd need HTTPS for the mic.

## Setup

### 1. Install dependencies

```bash
npm install
```

### 2. Start a local LiveKit server

With Docker (`--dev` enables the `devkey` / `secret` keys):

```bash
docker run --rm -p 7880:7880 -p 7881:7881 -p 7882:7882/udp \
  -e LIVEKIT_KEYS="devkey: secret" \
  livekit/livekit-server --dev --bind 0.0.0.0
```

Or with the CLI:

```bash
brew install livekit
livekit-server --dev --bind 0.0.0.0
```

### 3. Configure environment

`.env.local` (already created for local dev — add your Gemini key):

```env
LIVEKIT_API_KEY=devkey
LIVEKIT_API_SECRET=secret
LIVEKIT_URL=ws://localhost:7880
NEXT_PUBLIC_LIVEKIT_URL=ws://localhost:7880
GEMINI_API_KEY=your-gemini-api-key-here

# Optional: load the glossary arm's system instruction from a file instead of
# the built-in TWSE list (see "Customizing the glossary").
# GLOSSARY_SI_FILE=../stock-asr-eval/results/<videoId>.si.txt
```

### 4. Run the app

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000), then pick an input:

- **Start microphone** — allow mic access and start speaking (Mandarin, ideally
  about stocks).
- **Use a browser tab's audio** — in the share picker, choose the **Chrome Tab**
  option, select the tab that's playing audio, and turn on **"Also share tab
  audio"**. The tab's playback is transcribed instead of your mic. Chrome/Edge
  on desktop only; Firefox/Safari don't expose tab audio.

> Tab audio captures the *playback* of another tab — handy for transcribing a
> YouTube clip, a recorded meeting, or any Mandarin audio without a mic. The
> captured track is published to LiveKit exactly like the mic, so both Gemini
> sessions see it the same way.

## Customizing the glossary

The glossary and the two system instructions are fixed in config files — edit
them and restart `npm run dev`:

- `src/lib/glossary.ts` — the curated TWSE term list (`name` ↔ `ticker`) and the
  Chinese glossary instruction builder.
- `src/lib/asr-config.ts` — the model, target language, and both system
  instructions (`glossary` vs `general`).

### Load a generated, show-specific instruction

Instead of the built-in list, point the glossary arm at a system-instruction
file with `GLOSSARY_SI_FILE`. This pairs with the offline tool
[`../stock-asr-eval/glossary_llm.py`](../stock-asr-eval/glossary_llm.py), which
reads a stock show's YouTube subtitle and writes a tailored instruction (company
names + tickers + jargon, plus English-translation mappings):

```bash
# 1. generate the instruction for the show you'll play into the tab
python ../stock-asr-eval/glossary_llm.py "https://youtu.be/<videoId>"
#    → ../stock-asr-eval/results/<videoId>.si.txt

# 2. point the glossary arm at it, then restart
echo 'GLOSSARY_SI_FILE=../stock-asr-eval/results/<videoId>.si.txt' >> .env.local
npm run dev
```

The path resolves against the dev-server cwd (this app's directory). It's read
once at startup, so restart to re-apply. If the file is missing or empty the app
logs a warning and falls back to the built-in glossary. The **General** arm is
unchanged, so the A/B now compares *that show's* glossary against no glossary.

## Demo: drive the A/B without a mic

`demo/drive-asr.mjs` runs the whole comparison from the terminal — no mic, no
browser. It joins the LiveKit room as the speaker, publishes a PCM clip as its
audio track, starts both bridges, and prints the glossary-vs-general transcripts.
Handy for a reproducible A/B on a known clip (and for seeing what
`GLOSSARY_SI_FILE` changes).

```bash
# 1. grab a clip of the show as 16k mono PCM (start/end seconds)
python ../stock-asr-eval/fetch.py "https://youtu.be/<videoId>" 1220 1310
#    → ../stock-asr-eval/data/<videoId>.1220_1310.pcm

# 2. with LiveKit + `npm run dev` running (GLOSSARY_SI_FILE pointed at that show)
node demo/drive-asr.mjs ../stock-asr-eval/data/<videoId>.1220_1310.pcm
```

LiveKit creds and URLs come from the same env as the app (`LIVEKIT_URL`,
`LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`, `APP_URL`), defaulting to local dev.

## Project structure

```
src/
├── app/
│   ├── api/
│   │   ├── asr/route.ts      # POST start / DELETE stop the two bridges
│   │   └── token/route.ts    # LiveKit token for the speaker
│   ├── globals.css
│   ├── layout.tsx
│   └── page.tsx              # Whole UI: mic/tab capture, checkbox, two columns
└── lib/
    ├── glossary.ts           # TWSE terms + instruction builder
    ├── asr-config.ts         # Model + the two system instructions
    ├── asr-bridge.ts         # LiveKit ↔ Gemini bridge (one per variant)
    └── asr-session-manager.ts# Singleton: two bridges per room
```

## Key design decisions

- **One source, two sessions** — mic or tab audio, the only variable between
  columns is the system instruction, so differences are attributable to the
  glossary.
- **Tab audio is client-only** — `getDisplayMedia({ audio: true })` captures the
  tab's playback; the audio track is published to LiveKit as a Microphone-source
  track, so the server bridge needs no changes.
- **Translate model for plain ASR** — `gemini-3.5-live-translate-preview` gives
  `input_transcription` (the Chinese ASR we compare) for free, plus the English
  translation for the toggle.
- **Text only** — bridges don't publish audio back; they only forward transcripts
  over `publishData` (reliable data channel), independent of track subscription.
- **Append-by-turn** — transcripts are grouped by Gemini turn (`segmentId`);
  deltas accumulate into a block (Chinese line + optional English line below).

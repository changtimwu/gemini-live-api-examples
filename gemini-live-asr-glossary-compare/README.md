# ASR Glossary Compare

Side-by-side live transcription that shows what a **domain glossary** does to
speech recognition. You speak into your browser mic; the same audio is fed to
**two** Gemini Live sessions at once:

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
Browser mic ──publishes audio──▶ LiveKit room
                                     │
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
(`asr-glossary` / `asr-general`), subscribes to your mic, streams PCM to its own
Gemini Live WebSocket, and publishes transcripts back over the reliable data
channel. The model is `gemini-3.5-live-translate-preview`, which emits both the
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
```

### 4. Run the app

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000), click **Start microphone**,
allow mic access, and start speaking (Mandarin, ideally about stocks).

## Customizing the glossary

The glossary and the two system instructions are fixed in config files — edit
them and restart `npm run dev`:

- `src/lib/glossary.ts` — the curated TWSE term list (`name` ↔ `ticker`) and the
  Chinese glossary instruction builder.
- `src/lib/asr-config.ts` — the model, target language, and both system
  instructions (`glossary` vs `general`).

## Project structure

```
src/
├── app/
│   ├── api/
│   │   ├── asr/route.ts      # POST start / DELETE stop the two bridges
│   │   └── token/route.ts    # LiveKit token for the speaker
│   ├── globals.css
│   ├── layout.tsx
│   └── page.tsx              # Whole UI: mic, checkbox, two transcript columns
└── lib/
    ├── glossary.ts           # TWSE terms + instruction builder
    ├── asr-config.ts         # Model + the two system instructions
    ├── asr-bridge.ts         # LiveKit ↔ Gemini bridge (one per variant)
    └── asr-session-manager.ts# Singleton: two bridges per room
```

## Key design decisions

- **One mic, two sessions** — the only variable between columns is the system
  instruction, so differences are attributable to the glossary.
- **Translate model for plain ASR** — `gemini-3.5-live-translate-preview` gives
  `input_transcription` (the Chinese ASR we compare) for free, plus the English
  translation for the toggle.
- **Text only** — bridges don't publish audio back; they only forward transcripts
  over `publishData` (reliable data channel), independent of track subscription.
- **Append-by-turn** — transcripts are grouped by Gemini turn (`segmentId`);
  deltas accumulate into a block (Chinese line + optional English line below).

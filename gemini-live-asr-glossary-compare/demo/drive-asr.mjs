/**
 * Headless A/B driver — exercise the two ASR bridges without a mic or browser.
 *
 * Normally you feed audio by clicking "Start microphone" / "Use a browser tab's
 * audio" in the UI. That's impossible to script (the tab-audio picker is native
 * Chrome UI). This driver instead joins the LiveKit room as the speaker, publishes
 * a PCM clip as its audio track, starts both bridges via POST /api/asr, and prints
 * the glossary-vs-general transcripts it receives back over the data channel — so
 * you can see what GLOSSARY_SI_FILE does to recognition from the terminal.
 *
 * Prerequisites:
 *   1. LiveKit running (see README "Start a local LiveKit server").
 *   2. The dev server running (`npm run dev`) — set GLOSSARY_SI_FILE in .env.local
 *      to the show's instruction so the glossary arm is primed for that audio.
 *   3. A 16 kHz mono s16le PCM clip of the show. Produce one with the sibling
 *      harness, e.g.:
 *        python ../stock-asr-eval/fetch.py <youtubeUrl> <startSec> <endSec>
 *        # writes ../stock-asr-eval/data/<id>.<start>_<end>.pcm
 *
 * Usage:
 *   node demo/drive-asr.mjs <pcm_16k_mono_s16le> [seconds]
 *
 * Config via env (sensible local-dev defaults): LIVEKIT_URL, LIVEKIT_API_KEY,
 * LIVEKIT_API_SECRET, APP_URL, DEMO_ROOM.
 */
import {
  Room, RoomEvent, AudioSource, LocalAudioTrack, AudioFrame,
  TrackPublishOptions, TrackSource,
} from "@livekit/rtc-node";
import { AccessToken } from "livekit-server-sdk";
import { readFileSync } from "node:fs";

const WS = process.env.LIVEKIT_URL || "ws://localhost:7880";
const KEY = process.env.LIVEKIT_API_KEY || "devkey";
const SECRET = process.env.LIVEKIT_API_SECRET || "secret";
const APP = process.env.APP_URL || "http://localhost:3001";
const ROOM = process.env.DEMO_ROOM || "demo-glossary";
const SPEAKER = "speaker";
const SR = 16000, CH = 1; // what stock-asr-eval/fetch.py emits

const PCM_PATH = process.argv[2];
if (!PCM_PATH) {
  console.error("usage: node demo/drive-asr.mjs <pcm_16k_mono_s16le> [seconds]");
  process.exit(1);
}
const SECONDS = Number(process.argv[3] || Infinity);

// Per-segment so we can show the rephrase agent's correction (which replaces a segment's zh).
const arms = {
  glossary: { order: [], src: {}, cor: {} },
  general: { order: [], src: {}, cor: {} },
};

const at = new AccessToken(KEY, SECRET, { identity: SPEAKER });
at.addGrant({ roomJoin: true, room: ROOM, canPublish: true, canSubscribe: true, canPublishData: true });
const token = await at.toJwt();

const room = new Room();
room.on(RoomEvent.DataReceived, (payload) => {
  try {
    const m = JSON.parse(new TextDecoder().decode(payload));
    if (m.type !== "transcript") return;
    const a = arms[m.variant];
    if (!a) return;
    if (!a.order.includes(m.segmentId)) a.order.push(m.segmentId);
    if (m.kind === "source") a.src[m.segmentId] = (a.src[m.segmentId] || "") + m.text;
    else if (m.kind === "source-correction") a.cor[m.segmentId] = m.text;
    // translation ignored for this zh-focused comparison
  } catch { /* ignore non-transcript data */ }
});
await room.connect(WS, token, { autoSubscribe: false, dynacast: false });
console.log(`speaker joined ${ROOM} @ ${WS}`);

const source = new AudioSource(SR, CH);
const track = LocalAudioTrack.createAudioTrack("audio", source);
await room.localParticipant.publishTrack(track, new TrackPublishOptions({ source: TrackSource.SOURCE_MICROPHONE }));
console.log("published audio track");

const start = await fetch(`${APP}/api/asr`, {
  method: "POST",
  headers: { "content-type": "application/json" },
  body: JSON.stringify({ room: ROOM, speakerIdentity: SPEAKER }),
});
console.log("POST /api/asr ->", start.status, await start.text());
await new Promise((r) => setTimeout(r, 1500)); // let bridges subscribe

// Stream PCM in real time: 100 ms frames.
const pcm = readFileSync(PCM_PATH);
const bytesPerFrame = (SR / 10) * 2 * CH; // 3200
const total = Math.min(pcm.length, SECONDS === Infinity ? pcm.length : SECONDS * SR * 2 * CH);
const t0 = Date.now();
let n = 0;
for (let off = 0; off < total; off += bytesPerFrame) {
  const end = Math.min(off + bytesPerFrame, total);
  const ab = pcm.buffer.slice(pcm.byteOffset + off, pcm.byteOffset + end); // fresh, 2-byte aligned
  const int16 = new Int16Array(ab);
  await source.captureFrame(new AudioFrame(int16, SR, CH, int16.length));
  const target = ++n * 100, elapsed = Date.now() - t0;
  if (target > elapsed) await new Promise((r) => setTimeout(r, target - elapsed));
}
console.log(`streamed ${(total / (SR * 2 * CH)).toFixed(0)}s; waiting for tail transcripts...`);
await new Promise((r) => setTimeout(r, 12000));

await fetch(`${APP}/api/asr`, {
  method: "DELETE", headers: { "content-type": "application/json" },
  body: JSON.stringify({ room: ROOM }),
});
await room.disconnect();

// Per-segment final text: rephrased correction if present, else the raw ASR.
const final = (a) => a.order.map((id) => a.cor[id] ?? a.src[id] ?? "").join("").replace(/\s+/g, " ").trim();
const raw = (a) => a.order.map((id) => a.src[id] ?? "").join("").replace(/\s+/g, " ").trim();
console.log("\n================ A/B RESULT (Chinese ASR) ================");
console.log("\n--- GENERAL  arm — raw ASR ---\n" + raw(arms.general));
console.log("\n--- GLOSSARY arm — after rephrase ---\n" + final(arms.glossary));
console.log("\n--- rephrase agent corrections (glossary arm) ---");
const fixed = arms.glossary.order.filter((id) => arms.glossary.cor[id]);
if (!fixed.length) console.log("  (none)");
for (const id of fixed) console.log(`  "${arms.glossary.src[id]}"\n   -> "${arms.glossary.cor[id]}"`);
process.exit(0);

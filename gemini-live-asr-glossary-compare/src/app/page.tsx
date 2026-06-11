"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  LiveKitRoom,
  useDataChannel,
  useLocalParticipant,
  useTracks,
  TrackToggle,
} from "@livekit/components-react";
import "@livekit/components-styles";
import { Track, LocalTrackPublication } from "livekit-client";

type Variant = "glossary" | "general";
type InputMode = "mic" | "tab";

// Display order, left → right. Server bridges are unaffected (see asr-config VARIANTS).
const VARIANTS: Variant[] = ["general", "glossary"];
// The arm whose transcript is rephrased (see asr-config REPHRASE_VARIANTS). Only this column
// is held back in "polished only" mode.
const REPHRASED_VARIANT: Variant = "glossary";
const LABELS: Record<Variant, string> = {
  glossary: "Tuned with glossary",
  general: "General",
};
const SUBTITLES: Record<Variant, string> = {
  glossary: "Rephrased by gemini-3.1-flash-lite",
  general: "Raw ASR",
};

interface Segment {
  id: string;
  zh: string;
  en: string;
  corrected?: boolean; // zh was replaced by the rephrase agent's correction
}
type Transcripts = Record<Variant, Segment[]>;

// Break a (still-streaming) transcript into sentence lines, keeping the
// terminating punctuation attached. Handles both CJK (。！？；…) and Latin
// (.!?) terminators plus explicit newlines. The trailing, not-yet-finished
// sentence stays on its own line and keeps growing until punctuation arrives.
function splitSentences(text: string): string[] {
  if (!text) return [];
  return text
    .split(/(?<=[。！？；…!?\n])/)
    .map((s) => s.trim())
    .filter(Boolean);
}

const SPEAKER_IDENTITY = "speaker";

// Capture a browser tab's *playback* audio via getDisplayMedia. Chrome/Edge
// only offer the "Also share tab audio" checkbox when video is requested too
// and the user picks the "Chrome Tab" surface, so we ask for both, then keep
// only the audio. The video track is left alive (Chrome ties the capture
// session to it) but disabled, and is never published.
async function captureTabAudio(): Promise<MediaStream> {
  if (!navigator.mediaDevices?.getDisplayMedia) {
    throw new Error(
      "Tab audio capture isn't supported here. Use Chrome or Edge on desktop."
    );
  }
  const stream = await navigator.mediaDevices.getDisplayMedia({
    video: { displaySurface: "browser" },
    // Keep the tab's audio raw — browser DSP would hurt ASR quality.
    audio: {
      echoCancellation: false,
      noiseSuppression: false,
      autoGainControl: false,
    },
  });
  if (stream.getAudioTracks().length === 0) {
    stream.getTracks().forEach((t) => t.stop());
    throw new Error(
      'No tab audio captured. In the picker, choose the "Chrome Tab" option and turn on "Also share tab audio".'
    );
  }
  const video = stream.getVideoTracks()[0];
  if (video) video.enabled = false;
  return stream;
}

export default function Home() {
  const [started, setStarted] = useState(false);
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [room, setRoom] = useState("");
  const [token, setToken] = useState("");
  const [serverUrl, setServerUrl] = useState("");
  const [showEnglish, setShowEnglish] = useState(false);
  const [polishedOnly, setPolishedOnly] = useState(true);
  const [inputMode, setInputMode] = useState<InputMode>("mic");
  const [tabStream, setTabStream] = useState<MediaStream | null>(null);

  async function start(mode: InputMode) {
    setError(null);
    setStarting(true);
    // For tab mode, grab the audio first — getDisplayMedia must run inside the
    // click gesture, and we want to bail before touching the server if it fails.
    let stream: MediaStream | null = null;
    try {
      if (mode === "tab") stream = await captureTabAudio();

      const roomName = `asr-${crypto.randomUUID().slice(0, 8)}`;

      // 1. Spin up the two server-side ASR bridges (glossary + general).
      const startRes = await fetch("/api/asr", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ room: roomName, speakerIdentity: SPEAKER_IDENTITY }),
      });
      if (!startRes.ok) {
        const data = await startRes.json().catch(() => ({}));
        throw new Error(data.error || "Failed to start ASR bridges");
      }

      // 2. Get a LiveKit token for the speaker.
      const tokenRes = await fetch(
        `/api/token?room=${roomName}&identity=${SPEAKER_IDENTITY}`
      );
      const tokenData = await tokenRes.json();
      if (tokenData.error) throw new Error(tokenData.error);

      setInputMode(mode);
      setTabStream(stream);
      setRoom(roomName);
      setToken(tokenData.token);
      setServerUrl(tokenData.serverUrl);
      setStarted(true);
    } catch (err) {
      stream?.getTracks().forEach((t) => t.stop());
      setError((err as Error).message);
    } finally {
      setStarting(false);
    }
  }

  const stop = useCallback(async () => {
    if (room) {
      await fetch("/api/asr", {
        method: "DELETE",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ room }),
      }).catch(() => {});
    }
    tabStream?.getTracks().forEach((t) => t.stop());
    setTabStream(null);
    setInputMode("mic");
    setStarted(false);
    setToken("");
    setRoom("");
  }, [room, tabStream]);

  // Tear down bridges if the tab closes mid-session.
  useEffect(() => {
    if (!room) return;
    const handler = () => {
      navigator.sendBeacon?.(
        "/api/asr",
        new Blob([JSON.stringify({ room })], { type: "application/json" })
      );
    };
    window.addEventListener("beforeunload", handler);
    return () => window.removeEventListener("beforeunload", handler);
  }, [room]);

  if (!started) {
    return (
      <div className="page">
        <div className="container" style={{ textAlign: "center" }}>
          <h1 className="display display-xl enter" style={{ marginBottom: 24 }}>
            ASR <em>Glossary</em> Compare
          </h1>
          <p
            className="body enter-d1"
            style={{ maxWidth: 380, margin: "0 auto 40px" }}
          >
            Feed in your mic or a browser tab&apos;s audio. The same audio is
            transcribed by two live sessions side by side — one primed
            with a TWSE stock glossary, one general — so you can see the
            difference.
          </p>

          <div
            className="enter-d2"
            style={{
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              gap: 12,
            }}
          >
            <button
              className="btn btn-dark"
              onClick={() => start("mic")}
              disabled={starting}
            >
              {starting ? (
                <>
                  <span className="spinner" /> Starting…
                </>
              ) : (
                "Start microphone"
              )}
            </button>

            <button
              className="btn btn-outline"
              onClick={() => start("tab")}
              disabled={starting}
            >
              Use a browser tab&apos;s audio
            </button>

            <p
              className="body-sm"
              style={{ maxWidth: 320, margin: "4px auto 12px", opacity: 0.7 }}
            >
              Tab audio needs Chrome or Edge — in the picker, choose a tab and
              turn on &ldquo;Also share tab audio&rdquo;.
            </p>

            <label
              style={{
                display: "flex",
                alignItems: "center",
                gap: 8,
                cursor: "pointer",
              }}
            >
              <input
                type="checkbox"
                checked={showEnglish}
                onChange={(e) => setShowEnglish(e.target.checked)}
              />
              <span className="body-sm">Show English translation under each line</span>
            </label>
          </div>

          {error && (
            <p
              className="body-sm"
              style={{ color: "var(--error)", marginTop: 24 }}
            >
              {error}
            </p>
          )}

          <p className="mono enter-d4" style={{ marginTop: 56 }}>
            Powered by Gemini Live API + LiveKit
          </p>
        </div>
      </div>
    );
  }

  return (
    <LiveKitRoom
      video={false}
      audio={inputMode === "mic"}
      token={token}
      serverUrl={serverUrl}
      onDisconnected={() => stop()}
      style={{ height: "100vh" }}
    >
      <LiveSession
        showEnglish={showEnglish}
        setShowEnglish={setShowEnglish}
        polishedOnly={polishedOnly}
        setPolishedOnly={setPolishedOnly}
        onStop={stop}
        inputMode={inputMode}
        tabStream={tabStream}
      />
    </LiveKitRoom>
  );
}

function LiveSession({
  showEnglish,
  setShowEnglish,
  polishedOnly,
  setPolishedOnly,
  onStop,
  inputMode,
  tabStream,
}: {
  showEnglish: boolean;
  setShowEnglish: (v: boolean) => void;
  polishedOnly: boolean;
  setPolishedOnly: (v: boolean) => void;
  onStop: () => void;
  inputMode: InputMode;
  tabStream: MediaStream | null;
}) {
  const [transcripts, setTranscripts] = useState<Transcripts>({
    glossary: [],
    general: [],
  });
  const { localParticipant } = useLocalParticipant();
  const micTracks = useTracks([Track.Source.Microphone]);
  const isMicOn = micTracks.some(
    (t) =>
      t.participant.identity === localParticipant.identity &&
      !t.publication.isMuted
  );

  // In tab mode the captured audio isn't a real mic, so LiveKitRoom won't
  // auto-publish it. Publish it ourselves as a Microphone-source track (the
  // server bridge subscribes to any audio track; tagging it Microphone also
  // lets the waveform/"Listening" indicator below work unchanged).
  const tabPubRef = useRef<LocalTrackPublication | null>(null);
  const publishedRef = useRef(false);
  useEffect(() => {
    if (inputMode !== "tab" || !tabStream || publishedRef.current) return;
    const audioTrack = tabStream.getAudioTracks()[0];
    if (!audioTrack) return;
    publishedRef.current = true;
    localParticipant
      .publishTrack(audioTrack, {
        source: Track.Source.Microphone,
        name: "tab-audio",
      })
      .then((pub) => {
        tabPubRef.current = pub;
      })
      .catch((err) => console.error("Failed to publish tab audio:", err));
  }, [inputMode, tabStream, localParticipant]);

  // If the user hits "Stop sharing" in the browser bar, end the session. Kept
  // in its own effect so the listener stays live regardless of publish state.
  useEffect(() => {
    if (inputMode !== "tab" || !tabStream) return;
    const audioTrack = tabStream.getAudioTracks()[0];
    if (!audioTrack) return;
    const onEnded = () => onStop();
    audioTrack.addEventListener("ended", onEnded);
    return () => audioTrack.removeEventListener("ended", onEnded);
  }, [inputMode, tabStream, onStop]);

  const toggleTabMute = useCallback(() => {
    const pub = tabPubRef.current;
    if (!pub) return;
    if (pub.isMuted) pub.unmute();
    else pub.mute();
  }, []);

  // Receive transcripts from the ASR bridges over the data channel.
  useDataChannel("transcript", (msg) => {
    try {
      const data = JSON.parse(new TextDecoder().decode(msg.payload));
      if (data.type !== "transcript") return;
      const { variant, kind, segmentId, text } = data as {
        variant: Variant;
        kind: "source" | "translation" | "source-correction";
        segmentId: string;
        text: string;
      };
      if (!text) return;

      setTranscripts((prev) => {
        const segs = prev[variant].slice();
        const i = segs.findIndex((s) => s.id === segmentId);
        if (i === -1) {
          segs.push({
            id: segmentId,
            zh: kind === "translation" ? "" : text,
            en: kind === "translation" ? text : "",
            corrected: kind === "source-correction",
          });
        } else {
          const s = segs[i];
          segs[i] = {
            ...s,
            // "source" appends a live delta; "source-correction" replaces the whole turn.
            zh:
              kind === "source"
                ? s.zh + text
                : kind === "source-correction"
                ? text
                : s.zh,
            en: kind === "translation" ? s.en + text : s.en,
            corrected: kind === "source-correction" ? true : s.corrected,
          };
        }
        return { ...prev, [variant]: segs };
      });
    } catch {
      // ignore malformed payloads
    }
  });

  return (
    <div
      style={{
        height: "100vh",
        display: "flex",
        flexDirection: "column",
        padding: "20px 24px",
        gap: 16,
      }}
    >
      {/* Control bar */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          gap: 16,
          flexWrap: "wrap",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <div className={`waveform ${isMicOn ? "active" : "idle"}`}>
            {Array.from({ length: 5 }).map((_, i) => (
              <div key={i} className="waveform-bar" />
            ))}
          </div>
          <span
            className="status"
            style={{ color: isMicOn ? "var(--success)" : "var(--fg-ghost)" }}
          >
            <span className={`status-dot ${isMicOn ? "pulse" : ""}`} />
            {isMicOn
              ? inputMode === "tab"
                ? "Capturing tab audio"
                : "Listening"
              : "Muted"}
          </span>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: 20 }}>
          <label
            style={{
              display: "flex",
              alignItems: "center",
              gap: 8,
              cursor: "pointer",
            }}
          >
            <input
              type="checkbox"
              checked={showEnglish}
              onChange={(e) => setShowEnglish(e.target.checked)}
            />
            <span className="body-sm">Show English</span>
          </label>
          <label
            style={{
              display: "flex",
              alignItems: "center",
              gap: 8,
              cursor: "pointer",
            }}
            title="Hide the raw transcript on the rephrased side; show each line only after it's polished"
          >
            <input
              type="checkbox"
              checked={polishedOnly}
              onChange={(e) => setPolishedOnly(e.target.checked)}
            />
            <span className="body-sm">Polished only</span>
          </label>

          {inputMode === "tab" ? (
            <button
              onClick={toggleTabMute}
              style={{
                padding: "8px 16px",
                fontFamily: "var(--font-body)",
                fontSize: "13px",
                fontWeight: 500,
                border: "1px solid var(--border)",
                borderRadius: 0,
                background: "transparent",
                color: "var(--fg)",
                cursor: "pointer",
              }}
            >
              {isMicOn ? "Mute" : "Unmute"}
            </button>
          ) : (
            <TrackToggle
              source={Track.Source.Microphone}
              showIcon={false}
              style={{
                padding: "8px 16px",
                fontFamily: "var(--font-body)",
                fontSize: "13px",
                fontWeight: 500,
                border: "1px solid var(--border)",
                borderRadius: 0,
                background: "transparent",
                color: "var(--fg)",
                cursor: "pointer",
              }}
            >
              {isMicOn ? "Mute" : "Unmute"}
            </TrackToggle>
          )}

          <button className="btn-danger" onClick={onStop}>
            Stop
          </button>
        </div>
      </div>

      {/* Two transcript columns */}
      <div
        style={{
          flex: 1,
          minHeight: 0,
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: 16,
        }}
      >
        {VARIANTS.map((variant) => (
          <TranscriptColumn
            key={variant}
            variant={variant}
            segments={transcripts[variant]}
            showEnglish={showEnglish}
            polishedOnly={polishedOnly}
          />
        ))}
      </div>
    </div>
  );
}

function TranscriptColumn({
  variant,
  segments,
  showEnglish,
  polishedOnly,
}: {
  variant: Variant;
  segments: Segment[];
  showEnglish: boolean;
  polishedOnly: boolean;
}) {
  const scrollRef = useRef<HTMLDivElement>(null);

  // In "polished only" mode, the rephrased column shows a chunk only once its
  // correction has arrived (segments with `corrected`); other columns are unchanged.
  const visible =
    polishedOnly && variant === REPHRASED_VARIANT
      ? segments.filter((s) => s.corrected)
      : segments;

  // Auto-scroll to the newest line.
  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [visible, showEnglish]);

  // Flatten turns into display items: one item = one Chinese sentence paired
  // (by index) with its English translation. The pairing is approximate — the
  // model splits sentences differently across languages — but we only want an
  // item-by-item readout, so exactness doesn't matter.
  // Merge all visible chunks into one stream, then split into sentences — so the rephrase
  // chunk size doesn't dictate line length (lines break on punctuation, not per chunk).
  const zhLines = splitSentences(visible.map((s) => s.zh).join(""));
  const enLines = showEnglish ? splitSentences(visible.map((s) => s.en).join("")) : [];
  const count = Math.max(zhLines.length, enLines.length);
  const items = Array.from({ length: count }, (_, i) => ({
    key: `line-${i}`,
    zh: zhLines[i] || "",
    en: enLines[i] || "",
  }));

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        minHeight: 0,
        border: "1px solid var(--border)",
        background: "var(--bg-elevated)",
      }}
    >
      <div
        style={{ padding: "14px 18px", borderBottom: "1px solid var(--border)" }}
      >
        <div className="label" style={{ color: "var(--fg)" }}>
          {LABELS[variant]}
        </div>
        <div className="body-sm">{SUBTITLES[variant]}</div>
      </div>

      <div
        ref={scrollRef}
        style={{
          flex: 1,
          minHeight: 0,
          overflowY: "auto",
          padding: "18px",
          display: "flex",
          flexDirection: "column",
          gap: 0,
        }}
      >
        {items.length === 0 ? (
          <p className="body-sm italic">Waiting for speech…</p>
        ) : (
          items.map((item) => (
            <div
              key={item.key}
              style={{
                padding: "10px 0",
                borderBottom: "1px solid var(--border-light)",
              }}
            >
              {item.zh && (
                <p style={{ fontSize: 16, lineHeight: 1.5, color: "var(--fg)" }}>
                  {item.zh}
                </p>
              )}
              {showEnglish && item.en && (
                <p
                  className="body-sm"
                  style={{ marginTop: 4, color: "var(--fg-secondary)" }}
                >
                  {item.en}
                </p>
              )}
            </div>
          ))
        )}
      </div>
    </div>
  );
}

/**
 * AsrBridge: connects a LiveKit room to a Gemini Live API WebSocket for
 * real-time transcription (ASR) of the speaker's microphone.
 *
 * Each bridge instance:
 *   1. Joins the LiveKit room as a bot participant (e.g. "asr-glossary")
 *   2. Subscribes to the speaker's audio track
 *   3. Pipes PCM audio frames to a Gemini Live (translate) session
 *   4. Publishes transcriptions back over the reliable data channel:
 *        - input_transcription  → Chinese ASR   (kind "source")
 *        - output_transcription → English text  (kind "translation")
 *
 * Two bridges run per room — one with the glossary systemInstruction and one
 * with the general one — so the page can show them side by side. Unlike the
 * sibling translate app, this bridge does NOT publish audio back; we only care
 * about the text. (responseModalities must still be AUDIO for the translate
 * model; the audio frames are simply ignored.)
 */

import {
  Room,
  RoomEvent,
  RemoteTrackPublication,
  RemoteParticipant,
  RemoteAudioTrack,
  TrackKind,
  AudioStream,
  AudioFrame,
} from "@livekit/rtc-node";
import WebSocket from "ws";
import {
  Variant,
  MODEL,
  TARGET_LANGUAGE,
  SYSTEM_INSTRUCTIONS,
  REPHRASE_MODEL,
  REPHRASE_VARIANTS,
  REPHRASE_INSTRUCTION,
} from "./asr-config";

export type BridgeStatus = "starting" | "active" | "error" | "closed";
// "source-correction" replaces a segment's Chinese text with the rephrase agent's fix.
export type TranscriptKind = "source" | "translation" | "source-correction";

// The translate model keeps one long turn over continuous speech (turnComplete rarely fires), so
// the rephrase arm flushes a chunk for correction after a pause, or once it grows this long.
const REPHRASE_IDLE_MS = 1500;
const REPHRASE_MAX_CHARS = 80;

export class AsrBridge {
  private room: Room | null = null;
  private geminiWs: WebSocket | null = null;
  private readonly rephrase: boolean;
  private turnSource: string = ""; // accumulates the in-progress chunk's Chinese ASR
  private idleTimer: ReturnType<typeof setTimeout> | null = null;
  private turnId: number = 0;
  private framesSentToGemini: number = 0;
  private geminiSetupComplete: boolean = false;
  private audioWired: boolean = false;

  public readonly variant: Variant;
  public readonly room_: string;
  public readonly identity: string;
  public status: BridgeStatus = "starting";

  private readonly speakerIdentity: string;
  private readonly geminiApiKey: string;
  private readonly inputSampleRate: number = 48000; // LiveKit default
  private readonly channels: number = 1;

  private readonly livekitUrl: string;
  private readonly livekitApiKey: string;
  private readonly livekitApiSecret: string;

  constructor(
    room: string,
    variant: Variant,
    speakerIdentity: string,
    config: {
      geminiApiKey: string;
      livekitUrl: string;
      livekitApiKey: string;
      livekitApiSecret: string;
    }
  ) {
    this.room_ = room;
    this.variant = variant;
    this.speakerIdentity = speakerIdentity;
    this.identity = `asr-${variant}`;
    this.geminiApiKey = config.geminiApiKey;
    this.livekitUrl = config.livekitUrl;
    this.livekitApiKey = config.livekitApiKey;
    this.livekitApiSecret = config.livekitApiSecret;
    this.rephrase = REPHRASE_VARIANTS.includes(variant);
  }

  private log(...args: unknown[]): void {
    console.log(`[AsrBridge:${this.variant}]`, ...args);
  }

  async start(): Promise<void> {
    this.log(`Starting bridge for room ${this.room_}`);
    try {
      await this.joinLiveKitRoom();
      await this.connectGemini();
      this.subscribeToSpeaker();
      this.status = "active";
      this.log("Bridge is active");
    } catch (error) {
      this.log("Failed to start:", error);
      this.status = "error";
      throw error;
    }
  }

  async stop(): Promise<void> {
    this.log("Stopping bridge");
    this.status = "closed";
    if (this.idleTimer) {
      clearTimeout(this.idleTimer);
      this.idleTimer = null;
    }
    if (this.geminiWs) {
      this.geminiWs.close();
      this.geminiWs = null;
    }
    if (this.room) {
      await this.room.disconnect();
      this.room = null;
    }
    this.geminiSetupComplete = false;
  }

  private async joinLiveKitRoom(): Promise<void> {
    const { AccessToken } = await import("livekit-server-sdk");

    const at = new AccessToken(this.livekitApiKey, this.livekitApiSecret, {
      identity: this.identity,
      name: `ASR (${this.variant})`,
    });
    at.addGrant({
      roomJoin: true,
      room: this.room_,
      canPublish: false,
      canSubscribe: true,
      canPublishData: true,
    });
    const token = await at.toJwt();

    this.room = new Room();
    this.room.on(RoomEvent.Disconnected, () => {
      this.log("Disconnected from room");
      this.status = "closed";
    });

    await this.room.connect(this.livekitUrl, token, {
      autoSubscribe: false,
      dynacast: false,
    });
    this.log(`Joined room as ${this.identity}`);
  }

  private async connectGemini(): Promise<void> {
    return new Promise<void>((resolve, reject) => {
      this.openGeminiSocket();

      const checkSetup = setInterval(() => {
        if (this.geminiSetupComplete) {
          clearInterval(checkSetup);
          resolve();
        }
      }, 100);

      setTimeout(() => {
        if (!this.geminiSetupComplete) {
          clearInterval(checkSetup);
          reject(new Error("Gemini setup timeout"));
        }
      }, 15000);
    });
  }

  /** Open (or reopen) the Gemini WebSocket and wire its handlers. */
  private openGeminiSocket(): void {
    const wsUrl = `wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent?key=${this.geminiApiKey}`;
    this.geminiWs = new WebSocket(wsUrl);

    this.geminiWs.on("open", () => {
      this.log("Gemini WebSocket connected");
      this.sendGeminiSetup();
    });

    this.geminiWs.on("message", (data: WebSocket.Data) => {
      this.handleGeminiMessage(data);
    });

    this.geminiWs.on("error", (error) => {
      this.log("Gemini WebSocket error:", error);
    });

    this.geminiWs.on("close", (code: number, reason: Buffer) => {
      this.log("Gemini WebSocket closed", { code, reason: reason.toString() });
      // Auto-reconnect on unexpected closure while still active.
      if (this.status === "active") {
        this.geminiSetupComplete = false;
        setTimeout(() => {
          if (this.status === "active") this.openGeminiSocket();
        }, 1000);
      }
    });
  }

  private sendGeminiSetup(): void {
    const systemInstruction = SYSTEM_INSTRUCTIONS[this.variant];
    const setupMessage = {
      setup: {
        model: `models/${MODEL}`,
        ...(systemInstruction
          ? { systemInstruction: { parts: [{ text: systemInstruction }] } }
          : {}),
        inputAudioTranscription: {},
        outputAudioTranscription: {},
        generationConfig: {
          responseModalities: ["AUDIO"],
          translationConfig: {
            targetLanguageCode: TARGET_LANGUAGE,
            echoTargetLanguage: true,
          },
        },
        realtimeInputConfig: {
          automaticActivityDetection: { disabled: false },
        },
      },
    };
    this.geminiWs!.send(JSON.stringify(setupMessage));
  }

  private handleGeminiMessage(data: WebSocket.Data): void {
    try {
      const message = JSON.parse(data.toString());

      if (message.setupComplete) {
        this.log("Gemini setup complete");
        this.geminiSetupComplete = true;
        return;
      }

      const sc = message?.serverContent;

      // Chinese ASR of what the speaker said.
      if (sc?.inputTranscription?.text) {
        this.publishTranscript("source", sc.inputTranscription.text);
        if (this.rephrase) {
          this.turnSource += sc.inputTranscription.text;
          if (this.turnSource.length >= REPHRASE_MAX_CHARS) this.flushTurn();
          else this.scheduleIdleFlush();
        }
      }
      // English translation of the same turn.
      if (sc?.outputTranscription?.text) {
        this.publishTranscript("translation", sc.outputTranscription.text);
      }
      // We ignore sc.modelTurn audio parts — no audio is published back.

      if (sc?.turnComplete) {
        if (this.rephrase) this.flushTurn();
        else this.turnId++;
      }
    } catch (error) {
      this.log("Error parsing Gemini message:", error);
    }
  }

  private subscribeToSpeaker(): void {
    if (!this.room) return;

    const trySubscribe = (participant: RemoteParticipant) => {
      if (participant.identity !== this.speakerIdentity) return;
      for (const [, pub] of participant.trackPublications) {
        if (pub.kind === TrackKind.KIND_AUDIO) pub.setSubscribed(true);
      }
    };

    // Speaker may already be in the room.
    for (const [, participant] of this.room.remoteParticipants) {
      trySubscribe(participant);
    }

    // …or join / publish later.
    this.room.on(RoomEvent.TrackPublished, (pub, participant) => {
      if (
        participant.identity === this.speakerIdentity &&
        pub.kind === TrackKind.KIND_AUDIO
      ) {
        pub.setSubscribed(true);
      }
    });

    // Single subscription handler → pipe to Gemini once.
    this.room.on(
      RoomEvent.TrackSubscribed,
      (
        track: RemoteAudioTrack,
        pub: RemoteTrackPublication,
        participant: RemoteParticipant
      ) => {
        if (
          participant.identity === this.speakerIdentity &&
          pub.kind === TrackKind.KIND_AUDIO
        ) {
          this.pipeTrackToGemini(track);
        }
      }
    );
  }

  private pipeTrackToGemini(track: RemoteAudioTrack): void {
    if (this.audioWired) return; // guard against duplicate subscriptions
    this.audioWired = true;
    this.log("Subscribed to speaker audio, piping to Gemini");

    const audioStream = new AudioStream(track, {
      sampleRate: this.inputSampleRate,
      numChannels: this.channels,
      frameSizeMs: 100,
    });

    const reader = audioStream.getReader();
    const readLoop = async () => {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        this.sendAudioToGemini(value);
      }
    };
    readLoop().catch((err: Error) => this.log("Audio stream error:", err));
  }

  private sendAudioToGemini(frame: AudioFrame): void {
    if (
      !this.geminiWs ||
      this.geminiWs.readyState !== WebSocket.OPEN ||
      !this.geminiSetupComplete
    ) {
      return;
    }
    try {
      const int16 = frame.data;
      const buffer = Buffer.from(int16.buffer, int16.byteOffset, int16.byteLength);
      const base64 = buffer.toString("base64");
      this.framesSentToGemini++;
      this.geminiWs.send(
        JSON.stringify({
          realtimeInput: {
            audio: {
              mimeType: `audio/pcm;rate=${this.inputSampleRate}`,
              data: base64,
            },
          },
        })
      );
    } catch (error) {
      this.log("Error sending audio to Gemini:", error);
    }
  }

  private async publishTranscript(kind: TranscriptKind, text: string): Promise<void> {
    if (!this.room?.localParticipant) return;
    try {
      const payload = JSON.stringify({
        type: "transcript",
        variant: this.variant,
        kind,
        segmentId: `${this.variant}-${this.turnId}`,
        text,
        timestamp: Date.now(),
      });
      await this.room.localParticipant.publishData(
        new TextEncoder().encode(payload),
        { reliable: true, topic: "transcript" }
      );
    } catch (error) {
      this.log("Error publishing transcript:", error);
    }
  }

  /** (Re)arm the pause timer; firing flushes the accumulated chunk to the rephrase agent. */
  private scheduleIdleFlush(): void {
    if (this.idleTimer) clearTimeout(this.idleTimer);
    this.idleTimer = setTimeout(() => {
      this.idleTimer = null;
      this.flushTurn();
    }, REPHRASE_IDLE_MS);
  }

  /** Close the current chunk: rephrase it for its segment, then start a new segment. */
  private flushTurn(): void {
    if (this.idleTimer) {
      clearTimeout(this.idleTimer);
      this.idleTimer = null;
    }
    const raw = this.turnSource;
    this.turnSource = "";
    const segmentId = `${this.variant}-${this.turnId}`;
    this.turnId++;
    if (raw.trim()) {
      this.rephraseTurn(raw, segmentId).catch((e) => this.log("rephrase failed:", e));
    }
  }

  /**
   * Post-transcribe correction: send the completed turn's raw Chinese ASR to the
   * rephrase agent (gemini-3.1-flash-lite), which fixes domain-term mishearings
   * using the glossary knowledge, then publish the corrected text so the client
   * replaces that segment. On any failure we leave the live transcript untouched.
   */
  private async rephraseTurn(raw: string, segmentId: string): Promise<void> {
    const url =
      `https://generativelanguage.googleapis.com/v1beta/models/${REPHRASE_MODEL}` +
      `:generateContent?key=${this.geminiApiKey}`;
    const res = await fetch(url, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        systemInstruction: { parts: [{ text: REPHRASE_INSTRUCTION }] },
        contents: [{ role: "user", parts: [{ text: raw }] }],
        generationConfig: { temperature: 0 },
      }),
    });
    if (!res.ok) {
      this.log("rephrase HTTP", res.status, (await res.text()).slice(0, 120));
      return;
    }
    const data = await res.json();
    const corrected: string | undefined = data?.candidates?.[0]?.content?.parts
      ?.map((p: { text?: string }) => p.text ?? "")
      .join("")
      .trim();
    if (corrected && corrected !== raw.trim()) {
      this.log(`rephrased: "${raw.trim()}" -> "${corrected}"`);
      await this.publishCorrection(segmentId, corrected);
    }
  }

  /** Publish a corrected Chinese transcript for a finished segment (client replaces it). */
  private async publishCorrection(segmentId: string, text: string): Promise<void> {
    if (!this.room?.localParticipant) return;
    try {
      const payload = JSON.stringify({
        type: "transcript",
        variant: this.variant,
        kind: "source-correction" as TranscriptKind,
        segmentId,
        text,
        timestamp: Date.now(),
      });
      await this.room.localParticipant.publishData(
        new TextEncoder().encode(payload),
        { reliable: true, topic: "transcript" }
      );
    } catch (error) {
      this.log("Error publishing correction:", error);
    }
  }
}

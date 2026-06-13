/**
 * AsrSessionManager: singleton that owns the bridges for each room.
 *
 * A room here = one speaker. Starting a room spins up exactly two bridges
 * (glossary + general), both subscribed to the same mic. Stopping tears both
 * down. Idempotent: re-starting an already-running room is a no-op.
 */
import { AsrBridge, BridgeStatus } from "./asr-bridge";
import { ArmConfig, ARMS, Variant, VARIANTS } from "./asr-config";

export interface BridgeInfo {
  variant: Variant;
  identity: string;
  status: BridgeStatus;
  rephrase: boolean; // whether this arm runs the post-transcribe rephrase pass
  config: ArmConfig; // this arm's resolved config, for the client's "Config" popup
}

class AsrSessionManager {
  private static instance: AsrSessionManager;
  private rooms: Map<string, AsrBridge[]> = new Map();

  static getInstance(): AsrSessionManager {
    if (!AsrSessionManager.instance) {
      AsrSessionManager.instance = new AsrSessionManager();
    }
    return AsrSessionManager.instance;
  }

  private config() {
    return {
      geminiApiKey: process.env.GEMINI_API_KEY!,
      livekitUrl:
        process.env.LIVEKIT_URL ||
        process.env.NEXT_PUBLIC_LIVEKIT_URL ||
        "ws://localhost:7880",
      livekitApiKey: process.env.LIVEKIT_API_KEY!,
      livekitApiSecret: process.env.LIVEKIT_API_SECRET!,
    };
  }

  async start(room: string, speakerIdentity: string): Promise<BridgeInfo[]> {
    const existing = this.rooms.get(room);
    if (existing && existing.some((b) => b.status === "active")) {
      return existing.map(this.toInfo);
    }
    // Clean up any stale bridges first.
    if (existing) await this.stop(room);

    const bridges = VARIANTS.map(
      (variant) =>
        new AsrBridge(room, variant, speakerIdentity, this.config())
    );
    this.rooms.set(room, bridges);

    try {
      await Promise.all(bridges.map((b) => b.start()));
    } catch (error) {
      await this.stop(room);
      throw error;
    }
    return bridges.map(this.toInfo);
  }

  async stop(room: string): Promise<void> {
    const bridges = this.rooms.get(room);
    if (!bridges) return;
    await Promise.all(bridges.map((b) => b.stop()));
    this.rooms.delete(room);
  }

  private toInfo(b: AsrBridge): BridgeInfo {
    return {
      variant: b.variant,
      identity: b.identity,
      status: b.status,
      rephrase: b.rephrase,
      config: ARMS[b.variant],
    };
  }
}

export default AsrSessionManager;

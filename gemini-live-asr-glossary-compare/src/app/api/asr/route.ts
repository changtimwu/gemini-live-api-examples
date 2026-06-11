import { NextRequest, NextResponse } from "next/server";
import AsrSessionManager from "@/lib/asr-session-manager";

// POST /api/asr — start the two ASR bridges (glossary + general) for a room.
export async function POST(req: NextRequest) {
  try {
    const { room, speakerIdentity } = await req.json();
    if (!room || !speakerIdentity) {
      return NextResponse.json(
        { error: "Missing room or speakerIdentity" },
        { status: 400 }
      );
    }
    if (!process.env.GEMINI_API_KEY) {
      return NextResponse.json(
        { error: "GEMINI_API_KEY not configured" },
        { status: 500 }
      );
    }

    const manager = AsrSessionManager.getInstance();
    const bridges = await manager.start(room, speakerIdentity);
    return NextResponse.json({ room, bridges });
  } catch (error) {
    console.error("Error starting ASR bridges:", error);
    return NextResponse.json(
      { error: "Failed to start ASR: " + (error as Error).message },
      { status: 500 }
    );
  }
}

// DELETE /api/asr — tear down the bridges for a room.
export async function DELETE(req: NextRequest) {
  try {
    const { room } = await req.json();
    if (!room) {
      return NextResponse.json({ error: "Missing room" }, { status: 400 });
    }
    await AsrSessionManager.getInstance().stop(room);
    return NextResponse.json({ success: true });
  } catch (error) {
    console.error("Error stopping ASR bridges:", error);
    return NextResponse.json({ error: "Failed to stop ASR" }, { status: 500 });
  }
}

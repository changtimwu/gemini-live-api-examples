import { NextRequest, NextResponse } from "next/server";
import { AccessToken } from "livekit-server-sdk";

// GET /api/token — generate a LiveKit access token for the speaker.
// The speaker publishes their mic; the ASR bridges (server-side bots) join the
// same room and publish transcripts over the data channel, which the speaker
// receives without any extra grant.
export async function GET(req: NextRequest) {
  const room = req.nextUrl.searchParams.get("room");
  const identity = req.nextUrl.searchParams.get("identity");

  if (!room || !identity) {
    return NextResponse.json(
      { error: "Missing room or identity parameter" },
      { status: 400 }
    );
  }

  const apiKey = process.env.LIVEKIT_API_KEY;
  const apiSecret = process.env.LIVEKIT_API_SECRET;

  if (!apiKey || !apiSecret) {
    return NextResponse.json(
      { error: "LiveKit credentials not configured" },
      { status: 500 }
    );
  }

  const at = new AccessToken(apiKey, apiSecret, {
    identity,
    name: identity,
    ttl: "4h",
  });

  at.addGrant({
    roomJoin: true,
    room,
    canPublish: true, // mic
    canSubscribe: true,
  });

  const token = await at.toJwt();
  const serverUrl = process.env.LIVEKIT_URL || "ws://localhost:7880";

  return NextResponse.json({ token, serverUrl });
}

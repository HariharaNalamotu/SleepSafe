import { randomUUID } from "node:crypto";
import { NextResponse } from "next/server";
import { getDeviceStatus, sendCommand, type Command } from "@/lib/device";

export const dynamic = "force-dynamic";

/** GET /api/device: the recorder's latest published status. */
export async function GET() {
  try {
    return NextResponse.json(await getDeviceStatus());
  } catch (e) {
    return NextResponse.json({ error: (e as Error).message }, { status: 502 });
  }
}

/** POST /api/device {action: "start"|"stop", source?, demo_mode?}: queue a command for the Pi. */
export async function POST(req: Request) {
  let body: { action?: string; source?: string; demo_mode?: boolean };
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON" }, { status: 400 });
  }
  if (body.action !== "start" && body.action !== "stop") {
    return NextResponse.json({ error: "action must be start or stop" }, { status: 400 });
  }
  const source = body.source === "sample" ? "sample" : "headset";
  const now = new Date();
  const cmd: Command = {
    id: randomUUID(),
    action: body.action,
    requested_at: now.toISOString(),
    ...(body.action === "start" && {
      source,
      demo_mode: Boolean(body.demo_mode),
      session_id: `${source === "sample" ? "sample" : "night"}-${now.toISOString().slice(0, 19).replace(/[-:]/g, "").replace("T", "-")}`,
    }),
  };
  try {
    await sendCommand(cmd);
    return NextResponse.json({ queued: cmd });
  } catch (e) {
    return NextResponse.json({ error: (e as Error).message }, { status: 502 });
  }
}

import { NextResponse } from "next/server";
import { listSessions } from "@/lib/databricks";

export const dynamic = "force-dynamic";
export const maxDuration = 60;

/** GET /api/sessions: one summary row per session, newest first. */
export async function GET() {
  try {
    return NextResponse.json({ sessions: await listSessions() });
  } catch (e) {
    return NextResponse.json({ error: (e as Error).message }, { status: 502 });
  }
}

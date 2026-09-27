import { NextResponse } from "next/server";
import { getSession } from "@/lib/databricks";

export const dynamic = "force-dynamic";
export const maxDuration = 60;

/** GET /api/sessions/{id}: the full model report plus the written report. */
export async function GET(_req: Request, { params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  try {
    const data = await getSession(decodeURIComponent(id));
    if (!data) return NextResponse.json({ error: `Unknown session ${id}` }, { status: 404 });
    return NextResponse.json({ ...data.report, written_report: data.written });
  } catch (e) {
    return NextResponse.json({ error: (e as Error).message }, { status: 502 });
  }
}

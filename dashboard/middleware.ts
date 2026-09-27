import { NextResponse, type NextRequest } from "next/server";

// Sleep recordings are health data: every page and API route sits behind HTTP Basic auth
// (DASHBOARD_USER / DASHBOARD_PASSWORD). Without a password configured, nothing is served.
export function middleware(req: NextRequest) {
  const user = process.env.DASHBOARD_USER ?? "sleepsafe";
  const pass = process.env.DASHBOARD_PASSWORD;
  if (!pass) return new NextResponse("DASHBOARD_PASSWORD is not configured", { status: 503 });
  const header = req.headers.get("authorization") ?? "";
  if (header.startsWith("Basic ")) {
    const [u, ...rest] = atob(header.slice(6)).split(":");
    if (u === user && rest.join(":") === pass) return NextResponse.next();
  }
  return new NextResponse("Authentication required", {
    status: 401,
    headers: { "WWW-Authenticate": 'Basic realm="SleepSafe", charset="UTF-8"' },
  });
}

export const config = { matcher: ["/((?!_next/static|_next/image|favicon.ico).*)"] };

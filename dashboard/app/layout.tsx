import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "SleepSafe",
  description: "Audio-based sleep screening sessions and reports",
  robots: { index: false, follow: false },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <div className="wrap">
          <header className="top">
            <Link href="/" className="brand">
              SleepSafe <span>· sleep screening</span>
            </Link>
            <span className="muted">Screening aid, not a diagnosis · times in UTC</span>
          </header>
          {children}
        </div>
      </body>
    </html>
  );
}

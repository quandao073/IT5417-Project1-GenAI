import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";
import { Disclaimer } from "@/components/Disclaimer";
import { HealthBadge } from "@/components/HealthBadge";

export const metadata: Metadata = {
  title: "CXR Semantic Retrieval",
  description: "Research prototype: semantic retrieval of chest X-rays by pathology, view and negation.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body className="min-h-screen antialiased">
        <header className="border-b border-slate-200 bg-white">
          <div className="mx-auto flex max-w-7xl flex-wrap items-center gap-x-6 gap-y-2 px-4 py-3">
            <Link href="/" className="text-lg font-semibold tracking-tight text-slate-900">
              CXR Semantic Retrieval
            </Link>
            <nav className="flex gap-4 text-sm text-slate-600">
              <Link href="/" className="hover:text-slate-900">
                Search
              </Link>
              <Link href="/compare" className="hover:text-slate-900">
                Compare modes
              </Link>
            </nav>
            <div className="ml-auto">
              <HealthBadge />
            </div>
          </div>
        </header>
        <main className="mx-auto max-w-7xl px-4 py-6 pb-24">{children}</main>
        <Disclaimer />
      </body>
    </html>
  );
}

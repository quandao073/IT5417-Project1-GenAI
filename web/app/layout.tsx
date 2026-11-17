import type { Metadata } from "next";
import type { ReactNode } from "react";

import { Nav } from "@/components/Nav";
import { DISCLAIMER } from "@/lib/present";

import "./globals.css";

export const metadata: Metadata = {
  title: "Tìm ảnh X-quang ngực theo ngữ nghĩa",
  description: "Demo nghiên cứu: tìm ảnh X-quang ngực bằng câu hỏi EN/VI qua KG tri thức",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="vi">
      <body>
        <Nav />
        <main>{children}</main>
        <footer className="disclaimer" role="note">{DISCLAIMER}</footer>
      </body>
    </html>
  );
}

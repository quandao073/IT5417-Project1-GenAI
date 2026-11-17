"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";

import { health } from "@/lib/api";

export function Nav() {
  const path = usePathname();
  const [status, setStatus] = useState("đang kiểm API…");

  useEffect(() => {
    health()
      .then((h) => setStatus(`API sẵn sàng · KG ${h.kg_build_id ?? "tắt"} · Neo4j ${h.neo4j} · LLM ${h.llm}`))
      .catch(() => setStatus("API chưa sẵn sàng (xem README, mục Chạy demo)"));
  }, []);

  const links = [
    { href: "/", label: "Tìm kiếm" },
    { href: "/compare", label: "So sánh B0 · B1 · B2" },
  ];
  return (
    <header className="top">
      <span className="brand">Tìm ảnh X-quang ngực theo ngữ nghĩa</span>
      <nav aria-label="Trang">
        {links.map((l) => (
          <Link key={l.href} href={l.href} aria-current={path === l.href ? "page" : undefined}>
            {l.label}
          </Link>
        ))}
      </nav>
      <span className="status" aria-live="polite">{status}</span>
    </header>
  );
}

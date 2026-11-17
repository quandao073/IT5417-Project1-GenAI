"use client";

import { useState } from "react";

import type { View } from "@/lib/api";
import { EXAMPLES } from "@/lib/present";

export interface FormValue {
  query: string;
  topK: number;
  views: View[];
  explain: boolean;
}

const VIEWS: { label: string; value: View[] }[] = [
  { label: "Mọi view", value: [] },
  { label: "AP", value: ["AP"] },
  { label: "PA", value: ["PA"] },
];

export function SearchForm({ initial, busy, withExplain, onSubmit }: {
  initial: FormValue; busy: boolean; withExplain: boolean; onSubmit: (v: FormValue) => void;
}) {
  const [v, setV] = useState(initial);
  const submit = (next: FormValue) => {
    setV(next);
    if (next.query.trim()) onSubmit({ ...next, query: next.query.trim() });
  };
  return (
    <section className="panel">
      <form className="searchbar" role="search" onSubmit={(e) => { e.preventDefault(); submit(v); }}>
        <label htmlFor="q" className="sr-only" style={{ position: "absolute", left: -9999 }}>Câu hỏi</label>
        <input id="q" type="search" value={v.query} maxLength={500} autoComplete="off"
               placeholder="Câu hỏi tiếng Việt hoặc tiếng Anh, ví dụ: tràn dịch màng phổi, không tràn khí"
               onChange={(e) => setV({ ...v, query: e.target.value })} />
        <button className="primary" type="submit" disabled={busy || !v.query.trim()}>
          {busy ? "Đang tìm…" : "Tìm"}
        </button>
      </form>
      <div className="controls">
        <label>
          Top-k
          <select value={v.topK} onChange={(e) => setV({ ...v, topK: Number(e.target.value) })}>
            {[5, 10, 20].map((k) => <option key={k} value={k}>{k}</option>)}
          </select>
        </label>
        <span className="seg" role="group" aria-label="Lọc view">
          {VIEWS.map((o) => (
            <button key={o.label} type="button" aria-pressed={v.views.join() === o.value.join()}
                    onClick={() => setV({ ...v, views: o.value })}>
              {o.label}
            </button>
          ))}
        </span>
        {withExplain && (
          <label>
            <input type="checkbox" checked={v.explain}
                   onChange={(e) => setV({ ...v, explain: e.target.checked })} />
            Diễn giải
          </label>
        )}
      </div>
      <div className="examples" aria-label="Câu ví dụ">
        {EXAMPLES.map((ex) => (
          <button key={ex.text} type="button" disabled={busy} onClick={() => submit({ ...v, query: ex.text })}>
            {ex.text}<small>{ex.note}</small>
          </button>
        ))}
      </div>
    </section>
  );
}

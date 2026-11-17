"use client";

import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useCallback, useEffect, useRef, useState } from "react";

import { ResultCard } from "@/components/ResultGrid";
import type { FormValue } from "@/components/SearchForm";
import { SearchForm } from "@/components/SearchForm";
import type { Intent, KGResult, ResultsEvent, SearchResponse, View } from "@/lib/api";
import { searchBaseline, searchKGOnce } from "@/lib/api";

type Key = "B1" | "B2" | "B0";

const SYSTEMS: { key: Key; title: string; note: string }[] = [
  { key: "B1", title: "B1 · parser + câu mẫu + CLIP", note: "14 nhãn, không KG" },
  { key: "B2", title: "B2 · KG (hệ chính)", note: "KG chọn cột, trộn cosine w 0,2" },
  { key: "B0", title: "B0 · dịch EN + CLIP", note: "không parser" },
];

interface Column {
  results: KGResult[];
  intents: Intent[];
  text?: string | null;
  warnings: string[];
  error?: string;
}

const asKG = (r: SearchResponse): KGResult[] =>
  r.results.map((x) => ({ ...x, view: "", intents: [] }));

function ComparePage() {
  const params = useSearchParams();
  const router = useRouter();
  const views = (params.get("view") ?? "").split(",").filter((x): x is View => x === "AP" || x === "PA");
  const initial: FormValue = { query: params.get("q") ?? "", topK: Number(params.get("k") ?? 10) || 10,
                               views, explain: false };
  const [cols, setCols] = useState<Partial<Record<Key, Column>>>({});
  const [busy, setBusy] = useState(false);

  const run = useCallback(async (v: FormValue) => {
    setBusy(true);
    setCols({});
    const fail = (e: unknown): Column => ({ results: [], intents: [], warnings: [], error: String(e) });
    const b1 = searchBaseline("B1", v.query, v.topK, v.views)
      .then((r): Column => ({ results: asKG(r), intents: [], text: r.query_text, warnings: r.warnings }), fail);
    const b0 = searchBaseline("B0", v.query, v.topK, v.views)
      .then((r): Column => ({ results: asKG(r), intents: [], text: r.query_text, warnings: r.warnings }), fail);
    const b2 = searchKGOnce({ query: v.query, top_k: v.topK, views: v.views })
      .then((r: ResultsEvent): Column => ({ results: r.results, intents: r.intents,
                                            text: r.trace.fallback?.text, warnings: r.warnings }), fail);
    const [c1, c2, c0] = await Promise.all([b1, b2, b0]);
    setCols({ B1: c1, B2: c2, B0: c0 });
    setBusy(false);
  }, []);

  const submit = (v: FormValue) => {
    const q = new URLSearchParams({ q: v.query, k: String(v.topK) });
    if (v.views.length) q.set("view", v.views.join(","));
    router.replace(`/compare?${q}`);
    run(v);
  };

  const first = useRef(true);
  useEffect(() => {
    if (first.current && initial.query) run(initial);
    first.current = false;
  }, [initial, run]);

  const ids = (k: Key) => new Set((cols[k]?.results ?? []).map((r) => r.image_id));
  const others = (k: Key, id: string) =>
    SYSTEMS.filter((s) => s.key !== k && ids(s.key).has(id)).map((s) => s.key);
  const b1b2 = [...ids("B1")].filter((id) => ids("B2").has(id)).length;

  return (
    <>
      <SearchForm initial={initial} busy={busy} withExplain={false} onSubmit={submit} />
      {busy && <div className="skeleton" style={{ height: 240 }} aria-label="Đang tìm" />}
      {cols.B1 && cols.B2 && (
        <p className="small">
          Viền đậm: ảnh có ở hệ khác (nhãn ghi hệ nào). B1 và B2 chung {b1b2} ảnh trong top-k.
          Thứ hạng là thứ hạng của đánh giá: cùng code với <code>cxr eval-kg</code>.
        </p>
      )}
      <div className="compare">
        {SYSTEMS.map((s) => {
          const c = cols[s.key];
          if (!c) return null;
          return (
            <section key={s.key} className="panel" aria-labelledby={`h-${s.key}`}>
              <h2 id={`h-${s.key}`}>{s.title}</h2>
              <p className="small muted">{s.note}{c.text ? ` · văn bản: “${c.text}”` : ""}</p>
              {c.error && <p className="error">{c.error}</p>}
              {c.warnings.map((w) => <span key={w} className="tag warn">{w}</span>)}
              <div className="grid">
                {c.results.map((r) => {
                  const o = others(s.key, r.image_id);
                  return <ResultCard key={r.image_id} r={r} intents={c.intents}
                                     badge={o.length ? `cũng ở ${o.join(", ")}` : undefined} />;
                })}
              </div>
            </section>
          );
        })}
      </div>
    </>
  );
}

export default function Page() {
  return <Suspense><ComparePage /></Suspense>;
}

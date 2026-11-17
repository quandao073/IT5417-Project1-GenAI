"use client";

import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useCallback, useEffect, useRef, useState } from "react";

import { Detail } from "@/components/Detail";
import { Explanation } from "@/components/Explanation";
import { OutOfScope } from "@/components/OutOfScope";
import { ResultGrid } from "@/components/ResultGrid";
import type { FormValue } from "@/components/SearchForm";
import { SearchForm } from "@/components/SearchForm";
import { Understanding } from "@/components/Understanding";
import type { ExplanationEvent, KGResult, ResultsEvent, View } from "@/lib/api";
import { health, searchKG } from "@/lib/api";

function fromParams(p: URLSearchParams): FormValue {
  const views = (p.get("view") ?? "").split(",").filter((x): x is View => x === "AP" || x === "PA");
  return { query: p.get("q") ?? "", topK: Number(p.get("k") ?? 10) || 10, views,
           explain: p.get("explain") !== "0" };
}

function SearchPage() {
  const params = useSearchParams();
  const router = useRouter();
  const initial = fromParams(params);
  const [results, setResults] = useState<ResultsEvent | null>(null);
  const [explanation, setExplanation] = useState<ExplanationEvent | null>(null);
  const [busy, setBusy] = useState(false);
  const [explaining, setExplaining] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [open, setOpen] = useState<KGResult | null>(null);
  const [showReport, setShowReport] = useState(false);
  const abort = useRef<AbortController | null>(null);

  useEffect(() => { health().then((h) => setShowReport(h.show_report), () => {}); }, []);

  const run = useCallback(async (v: FormValue) => {
    abort.current?.abort();
    const ctrl = new AbortController();
    abort.current = ctrl;
    setBusy(true); setExplaining(v.explain); setError(null);
    setResults(null); setExplanation(null); setOpen(null);
    try {
      for await (const ev of searchKG({ query: v.query, top_k: v.topK, views: v.views,
                                        explain: v.explain }, ctrl.signal)) {
        if (ev.event === "results") { setResults(ev); setBusy(false); }
        else setExplanation(ev);
      }
    } catch (e) {
      if (!ctrl.signal.aborted) setError(String(e));
    } finally {
      if (abort.current === ctrl) { setBusy(false); setExplaining(false); }
    }
  }, []);

  const submit = (v: FormValue) => {
    const q = new URLSearchParams({ q: v.query, k: String(v.topK) });
    if (v.views.length) q.set("view", v.views.join(","));
    if (!v.explain) q.set("explain", "0");
    router.replace(`/?${q}`);
    run(v);
  };

  // a link with ?q= runs its search once (deep links, the screenshots)
  const first = useRef(true);
  useEffect(() => {
    if (first.current && initial.query) run(initial);
    first.current = false;
  }, [initial, run]);

  const line = open && explanation?.images?.find((x) => x.rank === open.rank)?.line;
  return (
    <>
      {/* keyed by the URL's question: an example tried from the refusal shows in the box */}
      <SearchForm key={initial.query} initial={initial} busy={busy} withExplain onSubmit={submit} />
      {error && <p className="error" role="alert">{error}</p>}
      {busy && !results && <div className="skeleton" style={{ height: 240 }} aria-label="Đang tìm" />}
      {results?.status === "out_of_scope" && (
        <OutOfScope data={results} onTry={(q) => submit({ ...initial, query: q })} />
      )}
      {results && results.status !== "out_of_scope" && (
        <div className="layout">
          <div>
            <h2>Kết quả · {results.results.length} ảnh</h2>
            <ResultGrid results={results.results} intents={results.intents} onOpen={setOpen} />
          </div>
          <div>
            <Understanding data={results} />
            {(explaining || explanation) && (
              <Explanation event={explanation} loading={explaining} intents={results.intents}
                           textRoute={Boolean(results.trace.fallback)} />
            )}
          </div>
        </div>
      )}
      {open && results && (
        <Detail r={open} data={results} line={line || undefined} showReport={showReport}
                onClose={() => setOpen(null)} />
      )}
    </>
  );
}

export default function Page() {
  return <Suspense><SearchPage /></Suspense>;
}

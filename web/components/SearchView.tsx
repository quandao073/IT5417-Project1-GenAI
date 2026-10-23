"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { errorMessage, search } from "@/lib/api";
import type { Mode, SearchRequest, SearchResponse, SearchResult } from "@/lib/types";
import { DEFAULT_FILTERS, Filters, type FilterValues } from "./Filters";
import { ModeSelector } from "./ModeSelector";
import { ParsedQueryChips } from "./ParsedQueryChips";
import { ResultGrid } from "./ResultGrid";
import { SearchBar } from "./SearchBar";
import { EmptyState, ErrorBanner, LoadingSkeleton, WarningsList } from "./States";
import { StudyDetailPanel } from "./StudyDetailPanel";

function buildRequest(query: string, mode: Mode, filters: FilterValues): SearchRequest {
  return { query, mode, ...filters };
}

export function SearchView() {
  const [query, setQuery] = useState("");
  const [mode, setMode] = useState<Mode>("vector");
  const [filters, setFilters] = useState<FilterValues>(DEFAULT_FILTERS);
  const [request, setRequest] = useState<SearchRequest | null>(null);
  const [response, setResponse] = useState<SearchResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [selected, setSelected] = useState<SearchResult | null>(null);

  useEffect(() => {
    if (!request) return;
    const controller = new AbortController();
    setLoading(true);
    setError(null);
    search(request, controller.signal).then(
      (res) => {
        if (controller.signal.aborted) return;
        setResponse(res);
        setLoading(false);
      },
      (err: unknown) => {
        if (controller.signal.aborted) return;
        setResponse(null);
        setError(errorMessage(err));
        setLoading(false);
      },
    );
    return () => controller.abort();
  }, [request]);

  function submit(q: string) {
    setQuery(q);
    setRequest(buildRequest(q, mode, filters));
  }

  function changeMode(next: Mode) {
    setMode(next);
    if (request) setRequest(buildRequest(request.query, next, filters));
  }

  function changeFilters(patch: Partial<FilterValues>) {
    const next = { ...filters, ...patch };
    setFilters(next);
    if (request) setRequest(buildRequest(request.query, mode, next));
  }

  const showSkeleton = loading && response === null;

  return (
    <div className="space-y-5">
      <section className="space-y-4 rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
        <SearchBar value={query} onChange={setQuery} onSubmit={submit} busy={loading} />
        <div className="flex flex-wrap items-end justify-between gap-4">
          <ModeSelector value={mode} onChange={changeMode} />
          <Filters value={filters} onChange={changeFilters} />
        </div>
      </section>

      {error && <ErrorBanner message={error} onRetry={() => request && setRequest({ ...request })} />}

      {response && (
        <section className="space-y-3">
          <ParsedQueryChips parsed={response.parsed_query} />
          <WarningsList warnings={response.warnings} />
          <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-slate-500">
            <span data-testid="result-count">
              {response.results.length} result{response.results.length === 1 ? "" : "s"}
            </span>
            {response.total_matches !== null && (
              <span data-testid="total-matches">{response.total_matches} studies match the constraints</span>
            )}
            <span data-testid="latency">{response.latency_ms} ms</span>
            <span className="font-mono">build {response.build_id}</span>
            {loading && <span className="text-sky-700">updating…</span>}
            {request && (
              <Link
                data-testid="compare-link"
                href={{ pathname: "/compare", query: { q: request.query } }}
                className="ml-auto text-sky-700 underline-offset-2 hover:underline"
              >
                Compare this query across modes →
              </Link>
            )}
          </div>
          {response.results.length === 0 ? (
            <EmptyState message="No study matches this query under the current mode and policies. Try another mode, relax the view filter or the absence policy." />
          ) : (
            <ResultGrid results={response.results} onOpen={setSelected} dimmed={loading} />
          )}
        </section>
      )}

      {showSkeleton && <LoadingSkeleton />}

      {!response && !loading && !error && (
        <EmptyState message="Describe the chest X-rays you are looking for, in English or Vietnamese, or pick an example above." />
      )}

      {selected && request && response && (
        <StudyDetailPanel
          result={selected}
          mode={request.mode}
          buildId={response.build_id}
          onClose={() => setSelected(null)}
        />
      )}
    </div>
  );
}

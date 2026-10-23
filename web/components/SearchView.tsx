"use client";

import { useEffect, useState } from "react";
import { errorMessage, search } from "@/lib/api";
import { formatScore, SCORE_CAPTION } from "@/lib/modes";
import type { Mode, SearchRequest, SearchResponse } from "@/lib/types";
import { DEFAULT_FILTERS, Filters, type FilterValues } from "./Filters";
import { ModeSelector } from "./ModeSelector";
import { ParsedQueryChips } from "./ParsedQueryChips";
import { SearchBar } from "./SearchBar";
import { EmptyState, ErrorBanner, LoadingSkeleton, WarningsList } from "./States";

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
          <div className="flex flex-wrap gap-x-4 text-xs text-slate-500">
            <span data-testid="result-count">{response.results.length} results</span>
            {response.total_matches !== null && (
              <span data-testid="total-matches">{response.total_matches} studies match the constraints</span>
            )}
            <span data-testid="latency">{response.latency_ms} ms</span>
            <span className="font-mono">build {response.build_id}</span>
          </div>
          {response.results.length === 0 ? (
            <EmptyState message="No study matches this query under the current mode and policies." />
          ) : (
            <ol className="divide-y divide-slate-200 rounded-lg border border-slate-200 bg-white text-sm">
              {response.results.map((r) => (
                <li key={r.study_id} className="flex items-center gap-4 px-3 py-2">
                  <span className="font-mono text-xs text-slate-500">#{r.rank}</span>
                  <span className="flex-1 font-mono text-xs">{r.study_id}</span>
                  {r.score !== null && (
                    <span className="font-mono">
                      {formatScore(r.score)} <span className="text-xs text-slate-500">{SCORE_CAPTION}</span>
                    </span>
                  )}
                </li>
              ))}
            </ol>
          )}
        </section>
      )}

      {loading && response === null && <LoadingSkeleton />}

      {!response && !loading && !error && (
        <EmptyState message="Describe the chest X-rays you are looking for, in English or Vietnamese, or pick an example above." />
      )}
    </div>
  );
}

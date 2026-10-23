"use client";

import { useEffect, useState } from "react";
import { errorMessage, search } from "@/lib/api";
import {
  DEFAULT_COMPARE_MODES,
  MAX_COMPARE_MODES,
  MIN_COMPARE_MODES,
  MODES,
  modeLabel,
} from "@/lib/modes";
import type { Mode, SearchResponse, SearchResult } from "@/lib/types";
import { DEFAULT_FILTERS, Filters, type FilterValues } from "./Filters";
import { ParsedQueryChips } from "./ParsedQueryChips";
import { ResultCard } from "./ResultCard";
import { SearchBar } from "./SearchBar";
import { EmptyState, ErrorBanner, WarningsList } from "./States";
import { StudyDetailPanel } from "./StudyDetailPanel";

interface CompareRequest {
  query: string;
  modes: Mode[];
  filters: FilterValues;
}

type Column =
  | { kind: "loading" }
  | { kind: "ok"; response: SearchResponse }
  | { kind: "error"; message: string };

export function CompareView() {
  const [query, setQuery] = useState("");
  const [modes, setModes] = useState<Mode[]>([...DEFAULT_COMPARE_MODES]);
  const [filters, setFilters] = useState<FilterValues>(DEFAULT_FILTERS);
  const [request, setRequest] = useState<CompareRequest | null>(null);
  const [columns, setColumns] = useState<Partial<Record<Mode, Column>>>({});
  const [selected, setSelected] = useState<{ result: SearchResult; mode: Mode; buildId: string } | null>(null);

  // `?q=` lets the search page hand its query over.
  useEffect(() => {
    const q = new URLSearchParams(window.location.search).get("q")?.trim();
    if (q) {
      setQuery(q);
      setRequest({ query: q, modes: [...DEFAULT_COMPARE_MODES], filters: DEFAULT_FILTERS });
    }
  }, []);

  useEffect(() => {
    if (!request) return;
    const controller = new AbortController();
    setColumns(Object.fromEntries(request.modes.map((m) => [m, { kind: "loading" }])));
    for (const mode of request.modes) {
      search({ query: request.query, mode, ...request.filters }, controller.signal).then(
        (response) => {
          if (controller.signal.aborted) return;
          setColumns((c) => ({ ...c, [mode]: { kind: "ok", response } }));
        },
        (err: unknown) => {
          if (controller.signal.aborted) return;
          setColumns((c) => ({ ...c, [mode]: { kind: "error", message: errorMessage(err) } }));
        },
      );
    }
    return () => controller.abort();
  }, [request]);

  function run(q: string, nextModes = modes, nextFilters = filters) {
    setQuery(q);
    setRequest({ query: q, modes: nextModes, filters: nextFilters });
  }

  function toggleMode(mode: Mode) {
    const next = modes.includes(mode) ? modes.filter((m) => m !== mode) : [...modes, mode];
    if (next.length < MIN_COMPARE_MODES || next.length > MAX_COMPARE_MODES) return;
    const ordered = MODES.map((m) => m.value).filter((m) => next.includes(m));
    setModes(ordered);
    if (request) run(request.query, ordered, filters);
  }

  function changeFilters(patch: Partial<FilterValues>) {
    const next = { ...filters, ...patch };
    setFilters(next);
    if (request) run(request.query, modes, next);
  }

  const shown = request?.modes ?? [];
  const busy = shown.some((m) => columns[m]?.kind === "loading");
  const firstOk = shown.map((m) => columns[m]).find((c) => c?.kind === "ok");
  const parsed = firstOk && firstOk.kind === "ok" ? firstOk.response.parsed_query : null;

  return (
    <div className="space-y-5">
      <section className="space-y-4 rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
        <div>
          <h1 className="text-lg font-semibold text-slate-900">Compare retrieval modes</h1>
          <p className="text-sm text-slate-600">
            The same query, the same policies, {MIN_COMPARE_MODES} to {MAX_COMPARE_MODES} modes side by side.
          </p>
        </div>
        <SearchBar value={query} onChange={setQuery} onSubmit={(q) => run(q)} busy={busy} inputTestId="compare-query-input" />
        <div className="flex flex-wrap items-end justify-between gap-4">
          <fieldset>
            <legend className="mb-1 text-xs font-medium uppercase tracking-wide text-slate-500">
              Columns ({modes.length} of {MAX_COMPARE_MODES})
            </legend>
            <div className="flex flex-wrap gap-x-4 gap-y-1">
              {MODES.map((m) => {
                const checked = modes.includes(m.value);
                const disabled = checked
                  ? modes.length <= MIN_COMPARE_MODES
                  : modes.length >= MAX_COMPARE_MODES;
                return (
                  <label
                    key={m.value}
                    title={m.hint}
                    className={`flex items-center gap-1.5 text-sm ${disabled ? "text-slate-400" : "text-slate-700"}`}
                  >
                    <input
                      type="checkbox"
                      data-testid={`compare-mode-${m.value}`}
                      checked={checked}
                      disabled={disabled}
                      onChange={() => toggleMode(m.value)}
                      className="h-4 w-4 accent-sky-700"
                    />
                    {m.label}
                  </label>
                );
              })}
            </div>
          </fieldset>
          <Filters value={filters} onChange={changeFilters} />
        </div>
      </section>

      {parsed && <ParsedQueryChips parsed={parsed} />}

      {request === null ? (
        <EmptyState message="Run a query to see how each mode ranks the corpus for it." />
      ) : (
        <div className="overflow-x-auto pb-2">
          <div
            data-testid="compare-grid"
            className="grid gap-4"
            style={{ gridTemplateColumns: `repeat(${shown.length}, minmax(260px, 1fr))` }}
          >
            {shown.map((mode) => {
              const column = columns[mode];
              return (
                <section
                  key={mode}
                  data-testid="compare-column"
                  data-mode={mode}
                  className="flex min-w-0 flex-col gap-3 rounded-lg border border-slate-200 bg-slate-100/60 p-3"
                >
                  <header className="space-y-1">
                    <h2 className="text-sm font-semibold text-slate-900">{modeLabel(mode)}</h2>
                    {column?.kind === "ok" && (
                      <p className="flex flex-wrap gap-x-3 text-xs text-slate-500">
                        <span data-testid="column-latency">{column.response.latency_ms} ms</span>
                        <span>{column.response.results.length} results</span>
                        {column.response.total_matches !== null && (
                          <span data-testid="column-total-matches">{column.response.total_matches} match</span>
                        )}
                      </p>
                    )}
                  </header>
                  {column?.kind === "loading" && (
                    <div data-testid="column-loading" className="animate-pulse space-y-3">
                      {Array.from({ length: 3 }, (_, i) => (
                        <div key={i} className="h-40 rounded-lg bg-slate-200" />
                      ))}
                    </div>
                  )}
                  {column?.kind === "error" && <ErrorBanner message={column.message} />}
                  {column?.kind === "ok" && (
                    <>
                      <WarningsList warnings={column.response.warnings} />
                      {column.response.results.length === 0 ? (
                        <EmptyState message="No result in this mode." />
                      ) : (
                        column.response.results.map((r) => (
                          <ResultCard
                            key={`${r.rank}-${r.study_id}`}
                            result={r}
                            compact
                            onOpen={(result) =>
                              setSelected({ result, mode, buildId: column.response.build_id })
                            }
                          />
                        ))
                      )}
                    </>
                  )}
                </section>
              );
            })}
          </div>
        </div>
      )}

      {selected && (
        <StudyDetailPanel
          result={selected.result}
          mode={selected.mode}
          buildId={selected.buildId}
          onClose={() => setSelected(null)}
        />
      )}
    </div>
  );
}

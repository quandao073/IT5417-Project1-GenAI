"use client";

import type { SearchResult } from "@/lib/types";
import { ResultCard } from "./ResultCard";

interface Props {
  results: SearchResult[];
  onOpen: (result: SearchResult) => void;
  dimmed?: boolean;
}

export function ResultGrid({ results, onOpen, dimmed = false }: Props) {
  return (
    <div
      data-testid="result-grid"
      aria-busy={dimmed}
      className={`grid gap-4 sm:grid-cols-2 lg:grid-cols-3 ${dimmed ? "opacity-50" : ""}`}
    >
      {results.map((r) => (
        <ResultCard key={`${r.rank}-${r.study_id}`} result={r} onOpen={onOpen} />
      ))}
    </div>
  );
}

"use client";

import { imageUrl } from "@/lib/api";
import { formatScore, SCORE_CAPTION } from "@/lib/modes";
import type { SearchResult } from "@/lib/types";
import { EvidenceList } from "./Evidence";
import { LabelBadge } from "./LabelBadge";

interface Props {
  result: SearchResult;
  onOpen: (result: SearchResult) => void;
  compact?: boolean;
}

export function ConstraintsIcon({ value }: { value: boolean | null }) {
  const text =
    value === true
      ? "constraints satisfied"
      : value === false
        ? "constraints violated"
        : "no constraint to check";
  const glyph = value === true ? "✓" : value === false ? "✗" : "–";
  const style =
    value === true
      ? "bg-emerald-100 text-emerald-800"
      : value === false
        ? "bg-rose-100 text-rose-800"
        : "bg-slate-100 text-slate-500";
  return (
    <span
      data-testid="constraints"
      data-value={value === null ? "null" : String(value)}
      title={text}
      aria-label={text}
      className={`inline-flex h-5 w-5 items-center justify-center rounded-full text-xs font-bold ${style}`}
    >
      {glyph}
    </span>
  );
}

export function ResultCard({ result, onOpen, compact = false }: Props) {
  const projection = result.projection ?? null;
  return (
    <article
      data-testid="result-card"
      data-rank={result.rank}
      data-study-id={result.study_id}
      className="flex flex-col gap-2 rounded-lg border border-slate-200 bg-white p-3 shadow-sm"
    >
      <button
        type="button"
        data-testid="open-detail"
        onClick={() => onOpen(result)}
        aria-label={`Open study ${result.study_id}`}
        className="group relative block overflow-hidden rounded-md bg-black"
      >
        <img
          src={imageUrl(result.image_url)}
          alt={`Chest X-ray, study ${result.study_id}`}
          loading="lazy"
          className="aspect-square w-full object-contain transition group-hover:opacity-90"
        />
        <span
          data-testid="result-rank"
          className="absolute left-2 top-2 rounded bg-slate-900/80 px-1.5 py-0.5 font-mono text-xs text-white"
        >
          #{result.rank}
        </span>
        {projection && (
          <span
            data-testid="projection-badge"
            className="absolute right-2 top-2 rounded bg-sky-700/90 px-1.5 py-0.5 text-xs font-semibold text-white"
          >
            {projection}
          </span>
        )}
      </button>

      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <p className="truncate font-mono text-xs text-slate-500" title={result.study_id}>
            {result.study_id}
          </p>
          {result.score !== null && (
            <p className="leading-tight">
              <span data-testid="score" className="font-mono text-sm font-semibold text-slate-900">
                {formatScore(result.score)}
              </span>
              <span data-testid="score-caption" className="ml-1.5 text-[11px] text-slate-500">
                {SCORE_CAPTION}
              </span>
            </p>
          )}
        </div>
        <ConstraintsIcon value={result.score_breakdown.constraints_satisfied} />
      </div>

      <div className="flex flex-wrap gap-1">
        {result.labels.length === 0 ? (
          <span className="text-xs text-slate-500">no constrained concept</span>
        ) : (
          result.labels.map((a) => <LabelBadge key={a.concept} assertion={a} />)
        )}
      </div>

      {!compact && <EvidenceList items={result.report_evidence} limit={3} />}
    </article>
  );
}

"use client";

import { useEffect, useState } from "react";
import { API_BASE, errorMessage, imageUrl, study } from "@/lib/api";
import { formatScore, modeLabel, SCORE_CAPTION } from "@/lib/modes";
import type { Mode, SearchResult, StudyDetail } from "@/lib/types";
import { EvidenceList } from "./Evidence";
import { ConstraintsIcon } from "./ResultCard";
import { sourceText, STATUS_STYLES, statusText } from "./LabelBadge";

interface Props {
  result: SearchResult;
  mode: Mode;
  buildId: string;
  onClose: () => void;
}

type DetailState =
  | { kind: "loading" }
  | { kind: "ok"; detail: StudyDetail }
  | { kind: "error"; message: string };

function Section({ title, children, testId }: { title: string; children: React.ReactNode; testId?: string }) {
  return (
    <section data-testid={testId} className="space-y-2">
      <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-500">{title}</h3>
      {children}
    </section>
  );
}

function Row({ name, value }: { name: string; value: React.ReactNode }) {
  return (
    <div className="flex justify-between gap-4 border-b border-slate-100 py-1 text-sm last:border-0">
      <dt className="text-slate-500">{name}</dt>
      <dd className="font-mono text-slate-900">{value}</dd>
    </div>
  );
}

function show(value: number | boolean | null | undefined): string {
  if (value === null || value === undefined) return "-";
  if (typeof value === "boolean") return value ? "true" : "false";
  return Number.isInteger(value) ? String(value) : value.toFixed(4);
}

export function StudyDetailPanel({ result, mode, buildId, onClose }: Props) {
  const [state, setState] = useState<DetailState>({ kind: "loading" });

  useEffect(() => {
    const controller = new AbortController();
    setState({ kind: "loading" });
    study(result.study_id, controller.signal).then(
      (detail) => setState({ kind: "ok", detail }),
      (err: unknown) => {
        if (!controller.signal.aborted) setState({ kind: "error", message: errorMessage(err) });
      },
    );
    return () => controller.abort();
  }, [result.study_id]);

  useEffect(() => {
    function onKey(event: KeyboardEvent) {
      if (event.key === "Escape") onClose();
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const detail = state.kind === "ok" ? state.detail : null;
  const projection = detail?.projection ?? result.projection ?? null;
  const labels = detail?.labels ?? result.labels;
  const breakdown = result.score_breakdown;
  const conceptScores = breakdown.concept_scores ?? null;

  return (
    <div className="fixed inset-0 z-50 flex justify-end">
      <button type="button" aria-label="Close study detail" onClick={onClose} className="flex-1 bg-slate-900/40" />
      <aside
        role="dialog"
        aria-modal="true"
        aria-labelledby="study-detail-title"
        data-testid="study-detail"
        className="flex h-full w-full max-w-2xl flex-col overflow-y-auto bg-white shadow-xl"
      >
        <header className="flex items-start justify-between gap-4 border-b border-slate-200 px-5 py-4">
          <div>
            <h2 id="study-detail-title" className="font-mono text-base font-semibold text-slate-900">
              {result.study_id}
            </h2>
            <p className="text-xs text-slate-500">
              image {result.image_id}
              {projection && (
                <>
                  {" / "}
                  <span data-testid="detail-projection" className="font-semibold text-sky-800">
                    {projection}
                  </span>
                </>
              )}
              {" / rank #"}
              {result.rank}
            </p>
          </div>
          <button
            type="button"
            data-testid="close-detail"
            onClick={onClose}
            className="rounded-md border border-slate-300 px-2 py-1 text-sm text-slate-700 hover:bg-slate-50"
          >
            Close
          </button>
        </header>

        <div className="space-y-6 px-5 py-4 pb-24">
          <img
            src={imageUrl(result.image_url)}
            alt={`Chest X-ray, study ${result.study_id}`}
            className="mx-auto max-h-[60vh] rounded-md bg-black object-contain"
          />

          <Section title="Report" testId="detail-report">
            {state.kind === "loading" && <p className="text-sm text-slate-500">Loading study detail...</p>}
            {state.kind === "error" && (
              <p data-testid="detail-fallback" className="text-xs text-amber-800">
                Study detail unavailable ({state.message}); showing the evidence from the search result only.
              </p>
            )}
            {detail && detail.report === null && (
              <p className="text-sm text-amber-800">This study has no report.</p>
            )}
            {detail?.report && (
              <dl className="space-y-2 text-sm">
                <div>
                  <dt className="font-medium text-slate-700">Impression</dt>
                  <dd data-testid="detail-impression" className="whitespace-pre-wrap text-slate-800">
                    {detail.report.impression ?? <span className="text-slate-400">-</span>}
                  </dd>
                </div>
                <div>
                  <dt className="font-medium text-slate-700">Findings</dt>
                  <dd data-testid="detail-findings" className="whitespace-pre-wrap text-slate-800">
                    {detail.report.findings ?? <span className="text-slate-400">-</span>}
                  </dd>
                </div>
              </dl>
            )}
            <div>
              <p className="mb-1 text-xs font-medium text-slate-600">Evidence for the query</p>
              <EvidenceList items={result.report_evidence} />
            </div>
          </Section>

          <Section title={detail ? "Label assertions (14)" : "Constrained labels"} testId="detail-labels">
            <table className="w-full text-sm">
              <thead className="text-left text-xs uppercase tracking-wide text-slate-500">
                <tr>
                  <th className="py-1 font-medium">Concept</th>
                  <th className="py-1 font-medium">Status</th>
                  <th className="py-1 font-medium">Source</th>
                </tr>
              </thead>
              <tbody>
                {labels.map((a) => (
                  <tr key={a.concept} data-testid="detail-label-row" className="border-t border-slate-100">
                    <td className="py-1 text-slate-800">{a.concept}</td>
                    <td className="py-1">
                      <span className={`rounded-full border px-2 py-0.5 text-xs ${STATUS_STYLES[a.status]}`}>
                        {statusText(a.status)}
                      </span>
                    </td>
                    <td className="py-1 font-mono text-xs text-slate-600" title={sourceText(a.source)}>
                      {a.source ?? "-"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Section>

          <Section title="Score breakdown" testId="detail-breakdown">
            <dl>
              <Row
                name="score"
                value={
                  result.score === null ? (
                    "- (label mode does not rank)"
                  ) : (
                    <>
                      {formatScore(result.score)} <span className="text-xs text-slate-500">({SCORE_CAPTION})</span>
                    </>
                  )
                }
              />
              <Row name="vector_rank" value={show(breakdown.vector_rank)} />
              <Row name="report_rank" value={show(breakdown.report_rank)} />
              <Row name="image_rank" value={show(breakdown.image_rank)} />
              <Row name="rrf_score" value={show(breakdown.rrf_score)} />
              <Row name="label_match" value={show(breakdown.label_match)} />
              <Row
                name="constraints_satisfied"
                value={
                  <span className="inline-flex items-center gap-1">
                    <ConstraintsIcon value={breakdown.constraints_satisfied} /> {show(breakdown.constraints_satisfied)}
                  </span>
                }
              />
            </dl>
            {conceptScores && Object.keys(conceptScores).length > 0 && (
              <table data-testid="concept-scores" className="mt-2 w-full text-sm">
                <thead className="text-left text-xs uppercase tracking-wide text-slate-500">
                  <tr>
                    <th className="py-1 font-medium">Concept</th>
                    <th className="py-1 text-right font-medium">log-prob term</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(conceptScores).map(([concept, value]) => (
                    <tr key={concept} className="border-t border-slate-100">
                      <td className="py-1 text-slate-800">{concept}</td>
                      <td className="py-1 text-right font-mono text-slate-900">{value.toFixed(4)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </Section>

          <Section title="Provenance" testId="detail-provenance">
            <dl>
              <Row name="mode" value={modeLabel(mode)} />
              <Row name="build_id" value={detail?.build_id ?? buildId} />
              <Row name="study_id" value={result.study_id} />
              <Row name="image_id" value={result.image_id} />
              <Row name="api" value={API_BASE || "same origin"} />
            </dl>
          </Section>
        </div>
      </aside>
    </div>
  );
}

import type { LabelAssertion, LabelStatus } from "@/lib/types";

export const STATUS_STYLES: Record<LabelStatus, string> = {
  PRESENT: "border-emerald-300 bg-emerald-100 text-emerald-900",
  ABSENT: "border-slate-300 bg-slate-100 text-slate-700",
  UNCERTAIN: "border-amber-300 bg-amber-100 text-amber-900",
  NOT_MENTIONED: "border-dashed border-slate-300 bg-white text-slate-500",
};

export function statusText(status: LabelStatus): string {
  return status.toLowerCase().replace("_", " ");
}

export function sourceText(source: string | null): string {
  return source ? `source: ${source}` : "source: none (no label source mentions this concept)";
}

export function LabelBadge({ assertion }: { assertion: LabelAssertion }) {
  return (
    <span
      data-testid="label-badge"
      data-status={assertion.status}
      title={sourceText(assertion.source)}
      className={`inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-xs font-medium ${STATUS_STYLES[assertion.status]}`}
    >
      {assertion.concept}
      <span className="font-normal opacity-75">· {statusText(assertion.status)}</span>
    </span>
  );
}

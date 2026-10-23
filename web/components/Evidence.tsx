import type { Evidence, Polarity } from "@/lib/types";

const POLARITY_STYLES: Record<Polarity, string> = {
  affirmed: "bg-emerald-100 text-emerald-900",
  negated: "bg-rose-100 text-rose-900",
  uncertain: "bg-amber-100 text-amber-900",
  mention: "bg-slate-100 text-slate-700",
};

export function EvidenceList({ items, limit }: { items: Evidence[]; limit?: number }) {
  if (items.length === 0) {
    return (
      <p data-testid="evidence-empty" className="text-xs text-amber-800">
        No report evidence for this study (no report, or the report index is disabled).
      </p>
    );
  }
  const shown = limit === undefined ? items : items.slice(0, limit);
  return (
    <ul data-testid="evidence-list" className="space-y-1.5">
      {shown.map((e, i) => (
        <li key={`${e.start}-${e.end}-${i}`} className="text-xs leading-snug text-slate-700">
          <span
            data-testid="evidence-polarity"
            className={`mr-1.5 rounded px-1.5 py-0.5 font-medium ${POLARITY_STYLES[e.polarity]}`}
          >
            {e.polarity}
          </span>
          <span className="mr-1.5 text-slate-500">{e.concept ?? "free text"}</span>
          <q className="italic">{e.text}</q>
        </li>
      ))}
      {shown.length < items.length && (
        <li className="text-xs text-slate-500">+{items.length - shown.length} more in the study detail</li>
      )}
    </ul>
  );
}

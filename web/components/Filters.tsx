"use client";

import { ABSENCE_POLICIES, TOP_K_OPTIONS, UNCERTAINTY_POLICIES } from "@/lib/modes";
import type { AbsencePolicy, UncertaintyPolicy, View } from "@/lib/types";

export interface FilterValues {
  /** null = let the parser decide */
  views: View[] | null;
  absence_policy: AbsencePolicy;
  uncertainty_policy: UncertaintyPolicy;
  top_k: number;
}

export const DEFAULT_FILTERS: FilterValues = {
  views: null,
  absence_policy: "exclude_present",
  uncertainty_policy: "EXCLUDE",
  top_k: 10,
};

interface Props {
  value: FilterValues;
  onChange: (patch: Partial<FilterValues>) => void;
}

const VIEWS: View[] = ["AP", "PA"];
const select =
  "rounded-md border border-slate-300 bg-white px-2 py-1.5 text-sm shadow-sm focus:border-sky-500 focus:outline-none";
const label = "flex flex-col gap-1 text-xs font-medium uppercase tracking-wide text-slate-500";

export function Filters({ value, onChange }: Props) {
  const selected = value.views ?? [];

  function toggleView(view: View) {
    const next = selected.includes(view) ? selected.filter((v) => v !== view) : [...selected, view];
    onChange({ views: next.length > 0 ? next : null });
  }

  return (
    <div className="flex flex-wrap items-end gap-4">
      <fieldset>
        <legend className="mb-1 text-xs font-medium uppercase tracking-wide text-slate-500">
          View filter
        </legend>
        <div className="flex gap-1.5" title="Overrides the views found by the parser; none = parser decides">
          {VIEWS.map((view) => {
            const active = selected.includes(view);
            return (
              <button
                key={view}
                type="button"
                aria-pressed={active}
                data-testid={`view-${view}`}
                onClick={() => toggleView(view)}
                className={`rounded-md border px-3 py-1.5 text-sm ${
                  active
                    ? "border-sky-700 bg-sky-700 text-white"
                    : "border-slate-300 bg-white text-slate-700 hover:border-sky-400"
                }`}
              >
                {view}
              </button>
            );
          })}
        </div>
      </fieldset>
      <label className={label}>
        Absence policy
        <select
          data-testid="absence-policy"
          className={select}
          value={value.absence_policy}
          onChange={(event) => onChange({ absence_policy: event.target.value as AbsencePolicy })}
        >
          {ABSENCE_POLICIES.map((policy) => (
            <option key={policy.value} value={policy.value} title={policy.hint}>
              {policy.label}
            </option>
          ))}
        </select>
      </label>
      <label className={label}>
        Uncertainty policy
        <select
          data-testid="uncertainty-policy"
          className={select}
          value={value.uncertainty_policy}
          onChange={(event) => onChange({ uncertainty_policy: event.target.value as UncertaintyPolicy })}
        >
          {UNCERTAINTY_POLICIES.map((policy) => (
            <option key={policy.value} value={policy.value} title={policy.hint}>
              {policy.label}
            </option>
          ))}
        </select>
      </label>
      <label className={label}>
        Top-k
        <select
          data-testid="top-k"
          className={select}
          value={value.top_k}
          onChange={(event) => onChange({ top_k: Number(event.target.value) })}
        >
          {TOP_K_OPTIONS.map((k) => (
            <option key={k} value={k}>
              {k}
            </option>
          ))}
        </select>
      </label>
    </div>
  );
}

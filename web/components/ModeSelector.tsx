"use client";

import { MODES } from "@/lib/modes";
import type { Mode } from "@/lib/types";

interface Props {
  value: Mode;
  onChange: (mode: Mode) => void;
}

export function ModeSelector({ value, onChange }: Props) {
  return (
    <fieldset>
      <legend className="mb-1 text-xs font-medium uppercase tracking-wide text-slate-500">Mode</legend>
      <div role="radiogroup" className="flex flex-wrap gap-1.5">
        {MODES.map((mode) => {
          const active = mode.value === value;
          return (
            <button
              key={mode.value}
              type="button"
              role="radio"
              aria-checked={active}
              data-testid={`mode-${mode.value}`}
              title={mode.hint}
              onClick={() => onChange(mode.value)}
              className={`rounded-md border px-3 py-1.5 text-sm transition ${
                active
                  ? "border-sky-700 bg-sky-700 text-white"
                  : "border-slate-300 bg-white text-slate-700 hover:border-sky-400"
              }`}
            >
              {mode.label}
            </button>
          );
        })}
      </div>
    </fieldset>
  );
}

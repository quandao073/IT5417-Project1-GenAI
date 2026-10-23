"use client";

import { useState } from "react";
import { EXAMPLE_QUERIES } from "@/lib/modes";

interface Props {
  value: string;
  onChange: (value: string) => void;
  onSubmit: (query: string) => void;
  busy?: boolean;
  inputTestId?: string;
}

export function SearchBar({ value, onChange, onSubmit, busy = false, inputTestId = "query-input" }: Props) {
  const [touched, setTouched] = useState(false);
  const trimmed = value.trim();
  const invalid = touched && trimmed.length === 0;

  return (
    <div className="space-y-3">
      <form
        className="flex gap-2"
        onSubmit={(event) => {
          event.preventDefault();
          setTouched(true);
          if (trimmed.length > 0) onSubmit(trimmed);
        }}
      >
        <input
          data-testid={inputTestId}
          type="text"
          value={value}
          maxLength={500}
          onChange={(event) => onChange(event.target.value)}
          placeholder="e.g. Find edema with no pleural effusion"
          aria-label="Search query"
          aria-invalid={invalid}
          className="flex-1 rounded-md border border-slate-300 bg-white px-3 py-2 text-base shadow-sm focus:border-sky-500 focus:outline-none focus:ring-2 focus:ring-sky-200"
        />
        <button
          data-testid="search-button"
          type="submit"
          disabled={busy}
          className="rounded-md bg-sky-700 px-4 py-2 text-sm font-medium text-white shadow-sm hover:bg-sky-800 disabled:opacity-60"
        >
          {busy ? "Searching…" : "Search"}
        </button>
      </form>
      {invalid && <p className="text-sm text-rose-700">Enter a query first.</p>}
      <div className="flex flex-wrap items-center gap-2 text-xs">
        <span className="text-slate-500">Examples:</span>
        {EXAMPLE_QUERIES.map((example) => (
          <button
            key={example.query}
            type="button"
            data-testid="example-query"
            title={example.title}
            onClick={() => {
              onChange(example.query);
              onSubmit(example.query);
            }}
            className="rounded-full border border-slate-300 bg-white px-2.5 py-1 text-slate-700 hover:border-sky-400 hover:text-sky-800"
          >
            {example.query}
          </button>
        ))}
      </div>
    </div>
  );
}

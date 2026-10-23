import type { ParsedQuery } from "@/lib/types";

const chip = "inline-flex items-center gap-1 rounded-full border px-2.5 py-0.5 text-xs font-medium";

export function ParsedQueryChips({ parsed }: { parsed: ParsedQuery }) {
  const noConstraint = parsed.include.length === 0 && parsed.exclude.length === 0;
  const parserTitle =
    parsed.llm_ms === null ? undefined : `LLM ${Math.round(parsed.llm_ms)} ms${parsed.cached ? ", cached" : ""}`;
  return (
    <div data-testid="parsed-query" className="space-y-1.5">
      <div className="flex flex-wrap items-center gap-1.5">
        <span className="text-xs uppercase tracking-wide text-slate-500">Parsed</span>
        {parsed.include.map((c) =>
          c.status === "UNCERTAIN" ? (
            <span
              key={`inc-${c.concept}`}
              data-testid="chip-uncertain"
              title="included with status UNCERTAIN"
              className={`${chip} border-amber-300 bg-amber-100 text-amber-900`}
            >
              {c.concept} <span className="font-normal opacity-80">· uncertain</span>
            </span>
          ) : (
            <span
              key={`inc-${c.concept}`}
              data-testid="chip-include"
              title="included (PRESENT)"
              className={`${chip} border-emerald-300 bg-emerald-100 text-emerald-900`}
            >
              {c.concept}
            </span>
          ),
        )}
        {parsed.exclude.map((concept) => (
          <span
            key={`exc-${concept}`}
            data-testid="chip-exclude"
            title="excluded (negated in the query)"
            className={`${chip} border-rose-300 bg-rose-100 text-rose-900`}
          >
            no {concept}
          </span>
        ))}
        {noConstraint && (
          <span className={`${chip} border-slate-300 bg-slate-100 text-slate-600`}>no concept recognised</span>
        )}
        {parsed.views.map((view) => (
          <span
            key={view}
            data-testid="chip-view"
            title="view requested in the query"
            className={`${chip} border-sky-300 bg-sky-100 text-sky-900`}
          >
            {view} view
          </span>
        ))}
        <span data-testid="chip-language" className={`${chip} border-slate-300 bg-white text-slate-700`}>
          lang: {parsed.language}
        </span>
        <span
          data-testid="parser-badge"
          title={parserTitle}
          className={`${chip} border-slate-300 bg-white font-mono text-slate-700`}
        >
          parser: {parsed.parser}
        </span>
        {parsed.fallback_used && (
          <span
            data-testid="fallback-flag"
            title="the LLM parser failed; the rule parser answered"
            className={`${chip} border-amber-300 bg-amber-50 text-amber-900`}
          >
            rule fallback
          </span>
        )}
      </div>
      {parsed.semantic_query && (
        <p className="text-xs text-slate-500">
          encoded text: <span className="font-mono text-slate-700">{parsed.semantic_query}</span>
        </p>
      )}
    </div>
  );
}

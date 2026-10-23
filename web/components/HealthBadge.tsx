"use client";

import { useEffect, useState } from "react";
import { errorMessage, health } from "@/lib/api";
import type { HealthResponse } from "@/lib/types";

type State =
  | { kind: "loading" }
  | { kind: "ok"; health: HealthResponse }
  | { kind: "error"; message: string };

function summary(h: HealthResponse): string {
  const lines = [
    `build ${h.build_id}`,
    `encoder ${h.model_id} (${h.revision.slice(0, 7)})`,
    `${h.studies} studies, ${h.vectors} vectors`,
    `report index: ${h.report_index}`,
    `image probe: ${h.image_probe}`,
    `image_joint: ${h.image_joint ?? "not in this API"}`,
    `image_decomposed: ${h.image_decomposed ?? "not in this API"}`,
    `llm: ${h.llm} (${h.llm_model})`,
  ];
  if (h.image_encoder) lines.push(`image encoder ${h.image_encoder.name}, build ${h.image_encoder.build_id}`);
  return lines.join("\n");
}

export function HealthBadge() {
  const [state, setState] = useState<State>({ kind: "loading" });

  useEffect(() => {
    const controller = new AbortController();
    health(controller.signal).then(
      (h) => setState({ kind: "ok", health: h }),
      (err: unknown) => {
        if (!controller.signal.aborted) setState({ kind: "error", message: errorMessage(err) });
      },
    );
    return () => controller.abort();
  }, []);

  const dot =
    state.kind === "ok" ? "bg-emerald-500" : state.kind === "error" ? "bg-rose-500" : "bg-slate-300";
  const text =
    state.kind === "ok"
      ? `API ready · build ${state.health.build_id}`
      : state.kind === "error"
        ? "API unreachable"
        : "Checking API…";
  const title = state.kind === "ok" ? summary(state.health) : state.kind === "error" ? state.message : "";

  return (
    <span
      data-testid="health-badge"
      data-state={state.kind}
      title={title}
      className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-slate-50 px-3 py-1 font-mono text-xs text-slate-700"
    >
      <span className={`h-2 w-2 rounded-full ${dot}`} aria-hidden />
      {text}
    </span>
  );
}

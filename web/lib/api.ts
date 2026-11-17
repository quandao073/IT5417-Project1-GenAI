// The API the demo reads (src/api/routes): types of what it returns, and the calls.

export const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000";

export type View = "AP" | "PA";

export interface Column {
  cui: string;
  column: string;
  weight: number;
  quality: number | null;
  weak: boolean;
  borderline: boolean;
  name: string;
  name_vi: string | null;
}

export interface Intent {
  term: string;
  cui: string;
  name: string;
  name_vi: string | null;
  polarity: "PRESENT" | "ABSENT";
  kind: "direct" | "wider" | "narrower" | "disease" | "anatomy" | "none";
  combine: "single" | "noisy_or" | "weighted" | "weighted_noisy_or";
  columns: Column[];
  path: string[];
  warnings: string[];
}

/** Per result, per intent with columns: the probability of the intent and of each column. */
export interface IntentScore {
  intent: number;
  p: number;
  columns: { cui: string; p: number }[];
}

export interface KGResult {
  rank: number;
  study_id: string;
  image_id: string;
  image_url: string;
  view: string;
  score: number;
  intents: IntentScore[];
  kg_z?: number;        // a fused score: score = kg_z + clip_part (z(KG) + 0.2 z(cosine))
  clip_part?: number;
}

export interface TraceTerm {
  term: string;
  skipped?: string;
  choice?: { cui: string | null; via: string; reason: string };
}

export interface ResultsEvent {
  event: "results";
  request_id: string;
  kg_build_id: string | null;
  matrix_version: string | null;
  build_id: string | null;
  versions: Record<string, string | number | null>;
  language: string;
  intents: Intent[];
  warnings: string[];
  degraded: string[];
  results: KGResult[];
  timings_ms: Record<string, number>;
  trace: {
    terms?: TraceTerm[];
    fallback?: { route: string; text: string };
    fusion?: { weight: number; text: string };
    english?: string | null;
  };
  disclaimer: string;
}

export interface ExplanationEvent {
  event: "explanation";
  request_id: string;
  status: "ok" | "partial" | "ungrounded" | "failed" | "disabled" | "no_results";
  summary?: string;
  images?: { rank: number; line: string }[];
  knowledge?: string;
  knowledge_items?: { id: string; text: string }[];
  dropped?: unknown[];
  fallback: "path" | null;
  error?: string | null;
}

export type KGEvent = ResultsEvent | ExplanationEvent;

export interface SearchResult {
  rank: number;
  study_id: string;
  image_id: string;
  image_url: string;
  score: number;
}

export interface SearchResponse {
  system: "B0" | "B1";
  query_text: string | null;
  results: SearchResult[];
  warnings: string[];
}

export interface Health {
  status: string;
  build_id: string;
  neo4j: string;
  llm: string;
  kg_build_id: string | null;
  show_report: boolean;
}

export const imageSrc = (url: string) => `${API_BASE}${url}`;

/** One JSON object per line of a byte stream, as each line arrives. */
export async function* readNdjson<T>(body: ReadableStream<Uint8Array>): AsyncGenerator<T> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { done, value } = await reader.read();
    buffer += decoder.decode(value, { stream: !done });
    let newline: number;
    while ((newline = buffer.indexOf("\n")) >= 0) {
      const line = buffer.slice(0, newline).trim();
      buffer = buffer.slice(newline + 1);
      if (line) yield JSON.parse(line) as T;
    }
    if (done) break;
  }
  if (buffer.trim()) yield JSON.parse(buffer) as T;
}

async function failure(r: Response): Promise<Error> {
  let detail = r.statusText;
  try {
    detail = JSON.stringify((await r.json()).detail);
  } catch {
    /* not JSON */
  }
  return new Error(`API ${r.status}: ${detail}`);
}

export interface KGRequest {
  query: string;
  top_k: number;
  views: View[];
  explain: boolean;
}

/** POST /api/v1/search/kg as a stream: the results event first, the explanation after. */
export async function* searchKG(req: KGRequest, signal?: AbortSignal): AsyncGenerator<KGEvent> {
  const r = await fetch(`${API_BASE}/api/v1/search/kg`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ...req, stream: true }),
    signal,
  });
  if (!r.ok || !r.body) throw await failure(r);
  yield* readNdjson<KGEvent>(r.body);
}

/** B2 in one object, without the explanation (the comparison screen). */
export async function searchKGOnce(req: Omit<KGRequest, "explain">): Promise<ResultsEvent> {
  const r = await fetch(`${API_BASE}/api/v1/search/kg`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ...req, stream: false, explain: false }),
  });
  if (!r.ok) throw await failure(r);
  return r.json();
}

/** B1 (parser, template, CLIP) or B0 (the question's English, CLIP): the evaluation's ranks. */
export async function searchBaseline(system: "B0" | "B1", query: string, top_k: number,
                                     views: View[]): Promise<SearchResponse> {
  const r = await fetch(`${API_BASE}/api/v1/search`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, top_k, system, views: views.length ? views : null }),
  });
  if (!r.ok) throw await failure(r);
  return r.json();
}

export async function health(): Promise<Health> {
  const r = await fetch(`${API_BASE}/api/v1/health`);
  if (!r.ok) throw await failure(r);
  return r.json();
}

/** Display only; the API answers when configs/demo.yaml sets show_report (D25). */
export async function studyReport(studyId: string): Promise<string> {
  const path = studyId.split("/").map(encodeURIComponent).join("/");
  const r = await fetch(`${API_BASE}/api/v1/studies/${path}/report`);
  if (!r.ok) throw await failure(r);
  return (await r.json()).text;
}

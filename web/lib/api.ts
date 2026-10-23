import type {
  HealthResponse,
  ParsedQuery,
  SearchRequest,
  SearchResponse,
  StudyDetail,
} from "./types";

const configured = process.env.NEXT_PUBLIC_API_BASE_URL;
/** Base URL of the FastAPI service; an empty string means same origin. Inlined at build time. */
export const API_BASE = (configured === undefined ? "http://localhost:8000" : configured).replace(
  /\/+$/,
  "",
);
export const TIMEOUT_MS = 30_000;
export const RETRY_DELAY_MS = 2_000;

export class ApiError extends Error {
  readonly status: number;
  readonly detail: unknown;
  readonly notReady: boolean;

  constructor(status: number, detail: unknown, message: string, notReady = false) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
    this.notReady = notReady;
  }
}

export function imageUrl(path: string): string {
  return `${API_BASE}${path}`;
}

export function errorMessage(err: unknown): string {
  if (err instanceof Error) return err.message;
  return "Unknown error";
}

export function search(req: SearchRequest, signal?: AbortSignal): Promise<SearchResponse> {
  return request<SearchResponse>("/api/v1/search", jsonPost(req), signal);
}

export function parse(query: string, signal?: AbortSignal): Promise<ParsedQuery> {
  return request<ParsedQuery>("/api/v1/query/parse", jsonPost({ query }), signal);
}

export function health(signal?: AbortSignal): Promise<HealthResponse> {
  return request<HealthResponse>("/api/v1/health", {}, signal);
}

export function study(studyId: string, signal?: AbortSignal): Promise<StudyDetail> {
  return request<StudyDetail>(`/api/v1/studies/${encodeURIComponent(studyId)}`, {}, signal);
}

function jsonPost(body: unknown): RequestInit {
  return {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/** The reason when the body is a 503 not_ready payload (bare as /health sends it, or under `detail`). */
function notReadyReason(body: unknown): string | null {
  const payload = isRecord(body) && isRecord(body.detail) ? body.detail : body;
  if (isRecord(payload) && payload.status === "not_ready") {
    return typeof payload.reason === "string" ? payload.reason : "not ready";
  }
  return null;
}

function describe(status: number, body: unknown): string {
  if (isRecord(body)) {
    const detail = body.detail;
    if (typeof detail === "string") return detail;
    if (isRecord(detail) && typeof detail.reason === "string") return detail.reason;
    if (Array.isArray(detail)) return "invalid request";
    if (typeof body.reason === "string") return body.reason;
  }
  return `HTTP ${status}`;
}

async function readBody(res: Response): Promise<unknown> {
  try {
    return await res.json();
  } catch {
    return null;
  }
}

function abortError(signal?: AbortSignal): unknown {
  return signal?.reason ?? new DOMException("The operation was aborted", "AbortError");
}

function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(abortError(signal));
      return;
    }
    const onAbort = () => {
      clearTimeout(timer);
      reject(abortError(signal));
    };
    const timer = setTimeout(() => {
      signal?.removeEventListener("abort", onAbort);
      resolve();
    }, ms);
    signal?.addEventListener("abort", onAbort, { once: true });
  });
}

async function request<T>(
  path: string,
  init: RequestInit,
  signal?: AbortSignal,
  retried = false,
): Promise<T> {
  const controller = new AbortController();
  let timedOut = false;
  const timer = setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, TIMEOUT_MS);
  const forward = () => controller.abort(abortError(signal));
  if (signal?.aborted) forward();
  else signal?.addEventListener("abort", forward, { once: true });

  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`, { ...init, signal: controller.signal });
  } catch (err) {
    if (timedOut) {
      throw new ApiError(0, null, `The API did not answer within ${TIMEOUT_MS / 1000} s`);
    }
    if (controller.signal.aborted) throw err; // cancelled by the caller
    throw new ApiError(0, null, `Cannot reach the API at ${API_BASE || "this origin"}`);
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener("abort", forward);
  }

  if (res.ok) return (await res.json()) as T;
  const body = await readBody(res);
  const reason = res.status === 503 ? notReadyReason(body) : null;
  if (reason !== null && !retried) {
    await sleep(RETRY_DELAY_MS, signal);
    return request<T>(path, init, signal, true);
  }
  throw new ApiError(
    res.status,
    body,
    reason !== null ? `API not ready: ${reason}` : describe(res.status, body),
    reason !== null,
  );
}

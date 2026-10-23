import type { Page, Route } from "@playwright/test";
import type {
  Evidence,
  HealthResponse,
  LabelAssertion,
  Mode,
  ParsedQuery,
  SearchRequest,
  SearchResponse,
  SearchResult,
  StudyDetail,
} from "../../lib/types";

export const BUILD_ID = "20261011_chexpert25k_biomedclip_v1";

// 1x1 black PNG, enough for <img> to load.
const PNG_1X1 = Buffer.from(
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==",
  "base64",
);

export const CHEXPERT_LABELS = [
  "No Finding",
  "Enlarged Cardiomediastinum",
  "Cardiomegaly",
  "Lung Opacity",
  "Lung Lesion",
  "Edema",
  "Consolidation",
  "Pneumonia",
  "Atelectasis",
  "Pneumothorax",
  "Pleural Effusion",
  "Pleural Other",
  "Fracture",
  "Support Devices",
];

export function parsed(overrides: Partial<ParsedQuery> = {}): ParsedQuery {
  return {
    semantic_query: "",
    include: [],
    exclude: [],
    views: [],
    language: "en",
    parser: "llm",
    fallback_used: false,
    warnings: [],
    cached: false,
    llm_ms: 812,
    ...overrides,
  };
}

export function label(concept: string, status: LabelAssertion["status"], source: string | null = "CHEXBERT_IMPRESSION"): LabelAssertion {
  return { concept, status, source };
}

export function evidence(concept: string | null, text: string, polarity: Evidence["polarity"]): Evidence {
  return { concept, text, start: 0, end: text.length, polarity };
}

export function result(rank: number, overrides: Partial<SearchResult> = {}): SearchResult {
  return {
    rank,
    study_id: `patient0000${rank}/study1`,
    image_id: `img${rank}`,
    image_url: `/api/v1/images/img${rank}`,
    score: Number((0.42 - rank * 0.01).toFixed(3)),
    score_breakdown: {
      vector_rank: rank,
      report_rank: null,
      image_rank: null,
      rrf_score: null,
      label_match: null,
      constraints_satisfied: true,
    },
    labels: [],
    report_evidence: [],
    projection: rank % 2 === 0 ? "AP" : "PA",
    ...overrides,
  };
}

export function response(
  mode: Mode,
  parsedQuery: ParsedQuery,
  results: SearchResult[],
  overrides: Partial<SearchResponse> = {},
): SearchResponse {
  return {
    request_id: `req-${mode}`,
    build_id: BUILD_ID,
    parsed_query: parsedQuery,
    results,
    total_matches: mode === "label" || mode === "hybrid" ? 123 : null,
    warnings: [],
    latency_ms: 42,
    ...overrides,
  };
}

export function studyDetail(studyId: string, overrides: Partial<StudyDetail> = {}): StudyDetail {
  return {
    study_id: studyId,
    image_id: "img1",
    projection: "PA",
    labels: CHEXPERT_LABELS.map((c) => label(c, c === "Cardiomegaly" ? "PRESENT" : "NOT_MENTIONED", c === "Cardiomegaly" ? "CHEXBERT_IMPRESSION" : null)),
    report: {
      impression: "Cardiomegaly without pulmonary edema.",
      findings: "The heart is enlarged. The lungs are clear. No pleural effusion.",
      search_text: "The heart is enlarged. The lungs are clear. No pleural effusion. Cardiomegaly without pulmonary edema.",
    },
    build_id: BUILD_ID,
    ...overrides,
  };
}

export const HEALTH: HealthResponse = {
  status: "ready",
  build_id: BUILD_ID,
  model_id: "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
  revision: "abcdef0123456789",
  dimension: 512,
  vectors: 25000,
  studies: 25000,
  qdrant: "ok",
  report_index: "ok",
  report_studies: 25000,
  image_probe: "ok",
  image_probe_id: "probe1",
  llm: "ok",
  llm_model: "gpt-5-mini",
  image_joint: "ok",
  image_decomposed: "disabled",
  image_encoder: { name: "cxrclip", model_id: "kakaobrain/cxr-clip-swint-m", revision: "r1", build_id: "20261101_chexpert25k_cxrclip_v1", collection: "cxr_images_20261101_chexpert25k_cxrclip_v1" },
};

export interface Status {
  status: number;
  body: unknown;
}

function isStatus(value: unknown): value is Status {
  return typeof value === "object" && value !== null && typeof (value as Status).status === "number";
}

export interface MockApi {
  /** Called per /search call with the request body and the 0-based call index. */
  search: (req: SearchRequest, call: number) => SearchResponse | Status;
  study?: (studyId: string) => StudyDetail | Status;
  health?: HealthResponse | Status;
  /** Delay every /search answer, to observe the loading state. */
  delayMs?: number;
}

export async function mockApi(page: Page, api: MockApi) {
  const requests: SearchRequest[] = [];
  let calls = 0;
  await page.route("**/api/v1/**", async (route: Route) => {
    const path = new URL(route.request().url()).pathname;
    if (path.endsWith("/api/v1/health")) return json(route, api.health ?? HEALTH);
    if (path.endsWith("/api/v1/search")) {
      const body = route.request().postDataJSON() as SearchRequest;
      requests.push(body);
      if (api.delayMs) await new Promise((resolve) => setTimeout(resolve, api.delayMs));
      return json(route, api.search(body, calls++));
    }
    if (path.includes("/api/v1/images/")) {
      return route.fulfill({ status: 200, contentType: "image/png", body: PNG_1X1 });
    }
    const study = path.match(/\/api\/v1\/studies\/(.+)$/);
    if (study) {
      const id = decodeURIComponent(study[1]);
      return json(route, api.study ? api.study(id) : { status: 404, body: { detail: "Not Found" } });
    }
    return route.fulfill({ status: 404, contentType: "application/json", body: JSON.stringify({ detail: `not mocked: ${path}` }) });
  });
  return { requests };
}

function json(route: Route, out: unknown) {
  if (isStatus(out)) {
    return route.fulfill({ status: out.status, contentType: "application/json", body: JSON.stringify(out.body) });
  }
  return route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(out) });
}

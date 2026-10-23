// Mirrors src/api/schemas/search.py. Fields marked optional are added by the serving package
// (WP5, modes image_joint / image_decomposed, /studies/{id}); older APIs omit them.

export type Mode = "vector" | "label" | "hybrid" | "image" | "image_joint" | "image_decomposed";
export type View = "AP" | "PA";
export type Language = "en" | "vi";
export type ParserName = "llm" | "rule";
export type AbsencePolicy = "exclude_present" | "explicit_absence_required";
export type UncertaintyPolicy = "EXCLUDE" | "INCLUDE";
export type ConstraintStatus = "PRESENT" | "UNCERTAIN";
export type LabelStatus = "PRESENT" | "ABSENT" | "UNCERTAIN" | "NOT_MENTIONED";
export type Polarity = "affirmed" | "negated" | "uncertain" | "mention";

export interface Constraint {
  concept: string;
  status: ConstraintStatus;
}

export interface SearchRequest {
  query: string;
  mode: Mode;
  top_k: number;
  /** AP/PA facet; overrides the views the parser found when given. */
  views?: View[] | null;
  absence_policy: AbsencePolicy;
  uncertainty_policy: UncertaintyPolicy;
}

export interface ParsedQuery {
  semantic_query: string;
  include: Constraint[];
  exclude: string[];
  views: View[];
  language: Language;
  parser: ParserName;
  fallback_used: boolean;
  warnings: string[];
  cached: boolean;
  llm_ms: number | null;
}

export interface LabelAssertion {
  concept: string;
  status: LabelStatus;
  /** winning label source; null if no source mentions the concept */
  source: string | null;
}

export interface ScoreBreakdown {
  vector_rank: number | null;
  report_rank: number | null;
  image_rank: number | null;
  rrf_score: number | null;
  label_match: boolean | null;
  /** hybrid/label: true by construction; vector/image: informational; null without constraints */
  constraints_satisfied: boolean | null;
  /** image_decomposed only: per-concept log-probability terms */
  concept_scores?: Record<string, number> | null;
}

export interface Evidence {
  /** null = free-text match */
  concept: string | null;
  text: string;
  start: number;
  end: number;
  polarity: Polarity;
}

export interface SearchResult {
  rank: number;
  study_id: string;
  image_id: string;
  /** relative, e.g. /api/v1/images/{image_id}; prefix with the API base */
  image_url: string;
  /** ranking signal, never a probability; null in label mode */
  score: number | null;
  score_breakdown: ScoreBreakdown;
  /** resolved status of every constrained concept */
  labels: LabelAssertion[];
  report_evidence: Evidence[];
  projection?: View | null;
}

export interface SearchResponse {
  request_id: string;
  build_id: string;
  parsed_query: ParsedQuery;
  results: SearchResult[];
  /** label/hybrid: studies matching after the view filter; null otherwise */
  total_matches: number | null;
  warnings: string[];
  latency_ms: number;
}

export interface StudyReport {
  impression: string | null;
  findings: string | null;
  search_text: string;
}

export interface StudyDetail {
  study_id: string;
  image_id: string;
  projection: View | null;
  /** all 14 CheXpert labels */
  labels: LabelAssertion[];
  report: StudyReport | null;
  build_id: string;
}

export interface ImageEncoderInfo {
  name: string;
  model_id: string;
  revision: string;
  build_id: string;
  collection: string;
}

export type ServiceStatus = "ok" | "disabled";

export interface HealthResponse {
  status: "ready";
  build_id: string;
  model_id: string;
  revision: string;
  dimension: number;
  vectors: number;
  studies: number;
  qdrant: "ok";
  report_index: ServiceStatus;
  report_studies: number | null;
  image_probe: ServiceStatus;
  image_probe_id: string | null;
  llm: "ok" | "disabled" | "unreachable" | "missing_model";
  llm_model: string;
  image_joint?: ServiceStatus;
  image_decomposed?: ServiceStatus;
  image_encoder?: ImageEncoderInfo | null;
}

export interface NotReady {
  status: "not_ready";
  reason: string;
}

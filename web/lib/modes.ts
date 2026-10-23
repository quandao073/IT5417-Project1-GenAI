import type { AbsencePolicy, Mode, UncertaintyPolicy } from "./types";

export interface ModeOption {
  value: Mode;
  label: string;
  hint: string;
}

export const MODES: readonly ModeOption[] = [
  { value: "vector", label: "Vector (BiomedCLIP)", hint: "text-image cosine over the corpus, no label filter" },
  { value: "label", label: "Label", hint: "label filter only; unranked, fixed pseudo-random order" },
  { value: "hybrid", label: "Hybrid", hint: "label filter, then RRF of the vector and report legs" },
  { value: "image", label: "Image (probe)", hint: "14 image probes on BiomedCLIP image vectors" },
  { value: "image_joint", label: "Image CXR-CLIP joint", hint: "template sentence vs CXR-CLIP image vectors" },
  { value: "image_decomposed", label: "Image CXR-CLIP decomposed", hint: "per-concept CXR-CLIP scores, summed" },
];

export function modeLabel(mode: Mode): string {
  return MODES.find((m) => m.value === mode)?.label ?? mode;
}

/** The six mandatory use cases of the plan (section 4.2). */
export const EXAMPLE_QUERIES: readonly { title: string; query: string }[] = [
  { title: "Single pathology", query: "Find chest X-rays with cardiomegaly." },
  { title: "Multiple pathologies", query: "Find cardiomegaly with pulmonary edema." },
  { title: "Explicit absence", query: "Find edema with no pleural effusion." },
  { title: "Uncertainty", query: "Find cases where pneumonia is uncertain." },
  { title: "View", query: "Show frontal PA images with lung opacity." },
  { title: "Vietnamese", query: "Tìm ảnh X-quang thẳng có tim to và phù phổi." },
];

export const TOP_K_OPTIONS = [5, 10, 20] as const;

export const ABSENCE_POLICIES: readonly { value: AbsencePolicy; label: string; hint: string }[] = [
  {
    value: "exclude_present",
    label: "Exclude present (default)",
    hint: "a negated finding passes when it is ABSENT or not mentioned",
  },
  {
    value: "explicit_absence_required",
    label: "Explicit absence required",
    hint: "a negated finding passes only when a label source says ABSENT",
  },
];

export const UNCERTAINTY_POLICIES: readonly { value: UncertaintyPolicy; label: string; hint: string }[] = [
  { value: "EXCLUDE", label: "Exclude uncertain (default)", hint: "UNCERTAIN does not count as present" },
  { value: "INCLUDE", label: "Include uncertain", hint: "UNCERTAIN counts as present" },
];

export const DEFAULT_COMPARE_MODES: readonly Mode[] = ["vector", "label", "hybrid", "image_joint"];
export const MIN_COMPARE_MODES = 3;
export const MAX_COMPARE_MODES = 4;

export const SCORE_CAPTION = "ranking signal, not a probability";
export const DISCLAIMER =
  "Research and educational prototype. Not for diagnosis or clinical decision-making.";

export function formatScore(score: number): string {
  return score.toFixed(3);
}

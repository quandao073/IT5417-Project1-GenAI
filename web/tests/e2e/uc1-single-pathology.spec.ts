import { expect, test } from "@playwright/test";
import { BUILD_ID, evidence, label, mockApi, parsed, response, result, studyDetail } from "./fixtures";

// Use case 1: one pathology.
const QUERY = "Find chest X-rays with cardiomegaly.";
const PARSED = parsed({
  semantic_query: "cardiomegaly",
  include: [{ concept: "Cardiomegaly", status: "PRESENT" }],
});

const RESULTS = [
  result(1, {
    labels: [label("Cardiomegaly", "PRESENT")],
    report_evidence: [evidence("Cardiomegaly", "The heart is enlarged.", "affirmed")],
  }),
  result(2, { labels: [label("Cardiomegaly", "PRESENT", "CHEXPERT_V1")] }),
  result(3, {
    labels: [label("Cardiomegaly", "UNCERTAIN")],
    score_breakdown: { ...result(3).score_breakdown, constraints_satisfied: false },
  }),
];

test("single pathology: chip, cards, score caption, evidence and disclaimer", async ({ page }) => {
  const { requests } = await mockApi(page, { search: (req) => response(req.mode, PARSED, RESULTS) });
  await page.goto("/");

  await expect(page.getByTestId("disclaimer")).toHaveText(
    "Research and educational prototype. Not for diagnosis or clinical decision-making.",
  );
  await expect(page.getByTestId("empty-state")).toBeVisible();
  await expect(page.getByTestId("health-badge")).toContainText(`build ${BUILD_ID}`);

  await page.getByTestId("query-input").fill(QUERY);
  await page.getByTestId("search-button").click();

  const chip = page.getByTestId("chip-include");
  await expect(chip).toHaveText("Cardiomegaly");
  await expect(chip).toHaveClass(/emerald/);
  await expect(page.getByTestId("chip-exclude")).toHaveCount(0);
  await expect(page.getByTestId("chip-language")).toHaveText("lang: en");
  await expect(page.getByTestId("parser-badge")).toHaveText("parser: llm");
  await expect(page.getByTestId("fallback-flag")).toHaveCount(0);

  const cards = page.getByTestId("result-card");
  await expect(cards).toHaveCount(3);
  await expect(cards.nth(0).getByTestId("result-rank")).toHaveText("#1");
  await expect(cards.nth(0).getByTestId("score")).toHaveText("0.410");
  await expect(cards.nth(0).getByTestId("score-caption")).toHaveText("ranking signal, not a probability");
  await expect(page.getByText("%")).toHaveCount(0);
  await expect(cards.nth(0).getByTestId("evidence-polarity")).toHaveText("affirmed");
  await expect(cards.nth(1).getByTestId("evidence-empty")).toBeVisible();
  await expect(cards.nth(1).getByTestId("label-badge")).toHaveAttribute("title", "source: CHEXPERT_V1");
  await expect(cards.nth(0).getByTestId("constraints")).toHaveAttribute("data-value", "true");
  await expect(cards.nth(2).getByTestId("constraints")).toHaveAttribute("data-value", "false");
  await expect(cards.nth(2).getByTestId("label-badge")).toHaveAttribute("data-status", "UNCERTAIN");
  await expect(page.getByTestId("total-matches")).toHaveCount(0); // null in vector mode

  expect(requests).toHaveLength(1);
  expect(requests[0]).toMatchObject({
    query: QUERY,
    mode: "vector",
    top_k: 10,
    views: null,
    absence_policy: "exclude_present",
    uncertainty_policy: "EXCLUDE",
  });
});

test("study detail drawer: report, 14 label assertions, breakdown and provenance", async ({ page }) => {
  await mockApi(page, {
    search: (req) => response(req.mode, PARSED, RESULTS),
    study: (id) => studyDetail(id),
  });
  await page.goto("/");
  await page.getByTestId("example-query").first().click();
  await page.getByTestId("result-card").first().getByTestId("open-detail").click();

  const drawer = page.getByTestId("study-detail");
  await expect(drawer).toBeVisible();
  await expect(drawer.getByTestId("detail-impression")).toHaveText("Cardiomegaly without pulmonary edema.");
  await expect(drawer.getByTestId("detail-findings")).toContainText("The heart is enlarged.");
  await expect(drawer.getByTestId("detail-label-row")).toHaveCount(14);
  await expect(drawer.getByTestId("detail-label-row").filter({ hasText: "Cardiomegaly" }).first()).toContainText(
    "CHEXBERT_IMPRESSION",
  );
  await expect(drawer.getByTestId("detail-breakdown")).toContainText("vector_rank");
  await expect(drawer.getByTestId("detail-breakdown")).toContainText("ranking signal, not a probability");
  await expect(drawer.getByTestId("detail-provenance")).toContainText(BUILD_ID);
  await expect(drawer.getByTestId("detail-provenance")).toContainText("Vector (BiomedCLIP)");
  await expect(drawer.getByTestId("detail-projection")).toHaveText("PA");

  await page.keyboard.press("Escape");
  await expect(drawer).toHaveCount(0);
});

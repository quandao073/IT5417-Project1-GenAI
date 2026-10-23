import { expect, test } from "@playwright/test";
import { evidence, label, mockApi, parsed, response, result } from "./fixtures";

// Use case 2: several pathologies, compared in vector and hybrid.
const QUERY = "Find cardiomegaly with pulmonary edema.";
const PARSED = parsed({
  semantic_query: "cardiomegaly and pulmonary edema",
  include: [
    { concept: "Cardiomegaly", status: "PRESENT" },
    { concept: "Edema", status: "PRESENT" },
  ],
});

function hybridResult(rank: number) {
  return result(rank, {
    score: 0.0328 - rank * 0.001,
    score_breakdown: {
      vector_rank: rank + 2,
      report_rank: rank,
      image_rank: null,
      rrf_score: 0.0328 - rank * 0.001,
      label_match: true,
      constraints_satisfied: true,
    },
    labels: [label("Cardiomegaly", "PRESENT"), label("Edema", "PRESENT", "CHEXBERT_FINDINGS")],
    report_evidence: [
      evidence("Cardiomegaly", "Enlarged cardiac silhouette.", "affirmed"),
      evidence("Edema", "Mild pulmonary edema.", "affirmed"),
    ],
  });
}

test("multiple pathologies: two chips, both labels on each card, hybrid re-runs with total_matches", async ({
  page,
}) => {
  const { requests } = await mockApi(page, {
    search: (req) =>
      req.mode === "hybrid"
        ? response("hybrid", PARSED, [hybridResult(1), hybridResult(2)], { total_matches: 57 })
        : response(req.mode, PARSED, [result(1, { labels: [label("Cardiomegaly", "PRESENT"), label("Edema", "ABSENT")] })]),
  });
  await page.goto("/");
  await page.getByTestId("query-input").fill(QUERY);
  await page.getByTestId("search-button").click();

  await expect(page.getByTestId("chip-include")).toHaveText(["Cardiomegaly", "Edema"]);
  await expect(page.getByTestId("result-card")).toHaveCount(1);
  await expect(page.getByTestId("result-card").first().getByTestId("label-badge")).toHaveCount(2);

  // Changing the mode re-runs the same query.
  await page.getByTestId("mode-hybrid").click();
  await expect(page.getByTestId("mode-hybrid")).toHaveAttribute("aria-checked", "true");
  await expect(page.getByTestId("result-card")).toHaveCount(2);
  await expect(page.getByTestId("total-matches")).toHaveText("57 studies match the constraints");

  const first = page.getByTestId("result-card").first();
  await expect(first.getByTestId("label-badge")).toHaveCount(2);
  await expect(first.getByTestId("label-badge").nth(0)).toHaveAttribute("data-status", "PRESENT");
  await expect(first.getByTestId("label-badge").nth(1)).toHaveAttribute("data-status", "PRESENT");
  await expect(first.getByTestId("evidence-polarity")).toHaveCount(2);
  await expect(first.getByTestId("score")).toHaveText("0.032");
  await expect(first.getByTestId("score-caption")).toBeVisible();

  expect(requests.map((r) => r.mode)).toEqual(["vector", "hybrid"]);
  expect(requests[1].query).toBe(QUERY);

  // The breakdown of a hybrid hit names both legs.
  await first.getByTestId("open-detail").click();
  const breakdown = page.getByTestId("detail-breakdown");
  await expect(breakdown).toContainText("report_rank");
  await expect(breakdown).toContainText("rrf_score");
  await expect(breakdown).toContainText("0.0318");
});

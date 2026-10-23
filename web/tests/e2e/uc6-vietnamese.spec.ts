import { expect, test } from "@playwright/test";
import { label, mockApi, parsed, response, result } from "./fixtures";

// Use case 6: a Vietnamese query answered by the rule parser after an LLM fallback, in label mode.
const QUERY = "Tìm ảnh X-quang thẳng có tim to và phù phổi.";
const PARSED = parsed({
  semantic_query: "",
  include: [
    { concept: "Cardiomegaly", status: "PRESENT" },
    { concept: "Edema", status: "PRESENT" },
  ],
  language: "vi",
  parser: "rule",
  fallback_used: true,
  llm_ms: null,
});

function labelResult(rank: number) {
  return result(rank, {
    score: null,
    score_breakdown: {
      vector_rank: null,
      report_rank: null,
      image_rank: null,
      rrf_score: null,
      label_match: true,
      constraints_satisfied: true,
    },
    labels: [label("Cardiomegaly", "PRESENT"), label("Edema", "PRESENT")],
  });
}

test("vietnamese: language and parser chips, fallback flag, label mode hides the score", async ({ page }) => {
  const { requests } = await mockApi(page, {
    search: (req) =>
      req.mode === "label"
        ? response("label", PARSED, [labelResult(1), labelResult(2)], { total_matches: 88 })
        : response(req.mode, PARSED, [result(1, { labels: [label("Cardiomegaly", "PRESENT"), label("Edema", "PRESENT")] })]),
  });
  await page.goto("/");
  await page.getByTestId("example-query").nth(5).click();

  await expect(page.getByTestId("query-input")).toHaveValue(QUERY);
  await expect(page.getByTestId("chip-language")).toHaveText("lang: vi");
  await expect(page.getByTestId("parser-badge")).toHaveText("parser: rule");
  await expect(page.getByTestId("fallback-flag")).toBeVisible();
  await expect(page.getByTestId("chip-include")).toHaveText(["Cardiomegaly", "Edema"]);
  await expect(page.getByTestId("score")).toHaveCount(1);

  await page.getByTestId("mode-label").click();
  await expect(page.getByTestId("result-card")).toHaveCount(2);
  await expect(page.getByTestId("score")).toHaveCount(0);
  await expect(page.getByTestId("score-caption")).toHaveCount(0);
  await expect(page.getByTestId("total-matches")).toHaveText("88 studies match the constraints");
  await expect(page.getByTestId("result-card").first().getByTestId("constraints")).toHaveAttribute("data-value", "true");

  expect(requests.map((r) => r.mode)).toEqual(["vector", "label"]);
  expect(requests[1].query).toBe(QUERY);
});

test("older API without /studies: the drawer falls back to the search evidence", async ({ page }) => {
  await mockApi(page, {
    search: (req) => response(req.mode, PARSED, [labelResult(1)]),
  });
  await page.goto("/");
  await page.getByTestId("example-query").nth(5).click();
  await page.getByTestId("open-detail").first().click();

  const drawer = page.getByTestId("study-detail");
  await expect(drawer.getByTestId("detail-fallback")).toBeVisible();
  await expect(drawer.getByTestId("detail-label-row")).toHaveCount(2);
  await expect(drawer.getByTestId("evidence-empty")).toBeVisible();
  await expect(drawer.getByTestId("detail-breakdown")).toContainText("label mode does not rank");
  await drawer.getByTestId("close-detail").click();
  await expect(drawer).toHaveCount(0);
});

test("no match shows the empty state", async ({ page }) => {
  await mockApi(page, { search: (req) => response(req.mode, PARSED, [], { total_matches: 0 }) });
  await page.goto("/");
  await page.getByTestId("mode-label").click();
  await page.getByTestId("example-query").nth(5).click();
  await expect(page.getByTestId("empty-state")).toContainText("No study matches this query");
  await expect(page.getByTestId("total-matches")).toHaveText("0 studies match the constraints");
});

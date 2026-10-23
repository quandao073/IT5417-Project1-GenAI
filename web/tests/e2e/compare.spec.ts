import { expect, test } from "@playwright/test";
import type { Mode, SearchRequest } from "../../lib/types";
import { evidence, label, mockApi, parsed, response, result } from "./fixtures";

// Use case 7: the same query across modes, side by side.
const QUERY = "Find edema with no pleural effusion.";
const PARSED = parsed({
  semantic_query: "pulmonary edema",
  include: [{ concept: "Edema", status: "PRESENT" }],
  exclude: ["Pleural Effusion"],
});

function column(req: SearchRequest) {
  const labels = [label("Edema", "PRESENT"), label("Pleural Effusion", "ABSENT")];
  switch (req.mode) {
    case "label":
      return response("label", PARSED, [result(1, { score: null, labels }), result(2, { score: null, labels })], {
        total_matches: 311,
        latency_ms: 12,
      });
    case "hybrid":
      return response("hybrid", PARSED, [result(1, { score: 0.0325, labels, report_evidence: [evidence("Pleural Effusion", "No effusion.", "negated")] })], {
        total_matches: 311,
        latency_ms: 180,
      });
    case "image_joint":
      return response("image_joint", PARSED, [], {
        warnings: ["image mode unavailable: build_manifest_image.json not found; run cxr index --role image"],
        latency_ms: 3,
      });
    case "image_decomposed":
      return response("image_decomposed", PARSED, [
        result(1, {
          score: -1.234,
          score_breakdown: {
            vector_rank: null,
            report_rank: null,
            image_rank: 1,
            rrf_score: null,
            label_match: null,
            constraints_satisfied: true,
            concept_scores: { Edema: -0.4, "Pleural Effusion": -0.834 },
          },
          labels,
        }),
      ]);
    default:
      return response(req.mode, PARSED, [result(1, { labels }), result(2, { labels }), result(3, { labels })], { latency_ms: 95 });
  }
}

test("compare: ?q runs the four default columns, deselecting one leaves three", async ({ page }) => {
  const { requests } = await mockApi(page, { search: column });
  await page.goto(`/compare?q=${encodeURIComponent(QUERY)}`);

  const columns = page.getByTestId("compare-column");
  await expect(columns).toHaveCount(4);
  await expect(columns.nth(0)).toHaveAttribute("data-mode", "vector");
  await expect(columns.nth(1)).toHaveAttribute("data-mode", "label");
  await expect(columns.nth(2)).toHaveAttribute("data-mode", "hybrid");
  await expect(columns.nth(3)).toHaveAttribute("data-mode", "image_joint");

  // Shared chips, once.
  await expect(page.getByTestId("parsed-query")).toHaveCount(1);
  await expect(page.getByTestId("chip-exclude")).toHaveText("no Pleural Effusion");

  // Per-column content.
  await expect(columns.nth(0).getByTestId("result-card")).toHaveCount(3);
  await expect(columns.nth(0).getByTestId("column-latency")).toHaveText("95 ms");
  await expect(columns.nth(1).getByTestId("column-total-matches")).toHaveText("311 match");
  await expect(columns.nth(1).getByTestId("score")).toHaveCount(0);
  await expect(columns.nth(2).getByTestId("score").first()).toHaveText("0.033");
  await expect(columns.nth(3).getByTestId("warnings")).toContainText("image mode unavailable");
  await expect(columns.nth(3).getByTestId("empty-state")).toBeVisible();
  await expect(page.getByTestId("disclaimer")).toBeVisible();

  const modes = (from: number) => requests.slice(from).map((r) => r.mode).sort();
  expect(modes(0)).toEqual((["vector", "label", "hybrid", "image_joint"] as Mode[]).sort());
  expect(requests.every((r) => r.query === QUERY)).toBe(true);

  // With four columns no fifth can be added; dropping one re-runs the remaining three.
  await expect(page.getByTestId("compare-mode-image")).toBeDisabled();
  await page.getByTestId("compare-mode-image_joint").uncheck();
  await expect(columns).toHaveCount(3);
  await expect(columns.nth(2)).toHaveAttribute("data-mode", "hybrid");
  expect(modes(4)).toEqual((["vector", "label", "hybrid"] as Mode[]).sort());
  // At three columns none can be removed.
  await expect(page.getByTestId("compare-mode-vector")).toBeDisabled();
});

test("compare: the decomposed column exposes concept scores in the drawer", async ({ page }) => {
  await mockApi(page, { search: column });
  await page.goto("/compare");
  await expect(page.getByTestId("empty-state")).toBeVisible();
  await page.getByTestId("compare-mode-image_joint").uncheck();
  await page.getByTestId("compare-mode-image_decomposed").check();
  await page.getByTestId("compare-query-input").fill(QUERY);
  await page.getByTestId("search-button").click();

  const col = page.locator('[data-testid="compare-column"][data-mode="image_decomposed"]');
  await expect(col).toHaveCount(1);
  await expect(col.getByTestId("score")).toHaveText("-1.234");
  await col.getByTestId("open-detail").click();
  const scores = page.getByTestId("concept-scores");
  await expect(scores).toContainText("Pleural Effusion");
  await expect(scores).toContainText("-0.8340");
  await expect(page.getByTestId("detail-provenance")).toContainText("Image CXR-CLIP decomposed");
});

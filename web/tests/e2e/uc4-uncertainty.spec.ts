import { expect, test } from "@playwright/test";
import { evidence, label, mockApi, parsed, response, result } from "./fixtures";

// Use case 4: an UNCERTAIN constraint and the uncertainty policy.
const QUERY = "Find cases where pneumonia is uncertain.";
const PARSED = parsed({
  semantic_query: "pneumonia",
  include: [{ concept: "Pneumonia", status: "UNCERTAIN" }],
});
const WARNING = "include Pneumonia is UNCERTAIN but uncertainty_policy is EXCLUDE; the constraint matches nothing";

test("uncertainty: amber chip, warning, uncertainty policy in the request", async ({ page }) => {
  const { requests } = await mockApi(page, {
    search: (req) =>
      req.uncertainty_policy === "INCLUDE"
        ? response(req.mode, PARSED, [
            result(1, {
              labels: [label("Pneumonia", "UNCERTAIN")],
              report_evidence: [evidence("Pneumonia", "Opacity may represent pneumonia.", "uncertain")],
            }),
          ])
        : response(req.mode, PARSED, [], { warnings: [WARNING] }),
  });
  await page.goto("/");
  await page.getByTestId("mode-label").click();
  await page.getByTestId("query-input").fill(QUERY);
  await page.getByTestId("search-button").click();

  const chip = page.getByTestId("chip-uncertain");
  await expect(chip).toContainText("Pneumonia");
  await expect(chip).toContainText("uncertain");
  await expect(chip).toHaveClass(/amber/);
  await expect(page.getByTestId("chip-include")).toHaveCount(0);
  await expect(page.getByTestId("warnings")).toContainText(WARNING);
  await expect(page.getByTestId("empty-state")).toBeVisible();
  await expect(page.getByTestId("result-card")).toHaveCount(0);

  await expect(page.getByTestId("uncertainty-policy")).toHaveValue("EXCLUDE");
  await page.getByTestId("uncertainty-policy").selectOption("INCLUDE");

  const card = page.getByTestId("result-card");
  await expect(card).toHaveCount(1);
  await expect(card.getByTestId("label-badge")).toHaveAttribute("data-status", "UNCERTAIN");
  await expect(card.getByTestId("evidence-polarity")).toHaveText("uncertain");
  await expect(page.getByTestId("warnings")).toHaveCount(0);

  expect(requests.map((r) => [r.mode, r.uncertainty_policy])).toEqual([
    ["label", "EXCLUDE"],
    ["label", "INCLUDE"],
  ]);
});

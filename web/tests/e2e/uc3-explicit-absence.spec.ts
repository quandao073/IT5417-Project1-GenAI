import { expect, test } from "@playwright/test";
import { evidence, label, mockApi, parsed, response, result } from "./fixtures";

// Use case 3: a negated finding and the absence policy.
const QUERY = "Find edema with no pleural effusion.";
const PARSED = parsed({
  semantic_query: "pulmonary edema",
  include: [{ concept: "Edema", status: "PRESENT" }],
  exclude: ["Pleural Effusion"],
});

const RESULTS = [
  result(1, {
    labels: [label("Edema", "PRESENT"), label("Pleural Effusion", "ABSENT", "CHEXBERT_FINDINGS")],
    report_evidence: [
      evidence("Edema", "Interstitial pulmonary edema.", "affirmed"),
      evidence("Pleural Effusion", "No pleural effusion.", "negated"),
    ],
  }),
  result(2, {
    labels: [label("Edema", "PRESENT"), label("Pleural Effusion", "NOT_MENTIONED", null)],
    report_evidence: [evidence("Edema", "Pulmonary edema is present.", "affirmed")],
  }),
];

test("explicit absence: red exclude chip, negated evidence, absence policy in the request", async ({ page }) => {
  const { requests } = await mockApi(page, {
    search: (req) =>
      response(
        req.mode,
        PARSED,
        req.absence_policy === "explicit_absence_required" ? RESULTS.slice(0, 1) : RESULTS,
      ),
  });
  await page.goto("/");
  await page.getByTestId("query-input").fill(QUERY);
  await page.getByTestId("search-button").click();

  await expect(page.getByTestId("chip-include")).toHaveText("Edema");
  const exclude = page.getByTestId("chip-exclude");
  await expect(exclude).toHaveText("no Pleural Effusion");
  await expect(exclude).toHaveClass(/rose/);

  const cards = page.getByTestId("result-card");
  await expect(cards).toHaveCount(2);
  const effusion = cards.nth(0).getByTestId("label-badge").filter({ hasText: "Pleural Effusion" });
  await expect(effusion).toHaveAttribute("data-status", "ABSENT");
  await expect(effusion).toHaveAttribute("title", "source: CHEXBERT_FINDINGS");
  await expect(cards.nth(0).getByTestId("evidence-polarity").nth(1)).toHaveText("negated");
  await expect(cards.nth(1).getByTestId("label-badge").nth(1)).toHaveAttribute("data-status", "NOT_MENTIONED");
  await expect(cards.nth(1).getByTestId("label-badge").nth(1)).toHaveAttribute("title", /source: none/);

  await expect(page.getByTestId("absence-policy")).toHaveValue("exclude_present");
  await page.getByTestId("absence-policy").selectOption("explicit_absence_required");
  await expect(cards).toHaveCount(1);

  expect(requests.map((r) => r.absence_policy)).toEqual(["exclude_present", "explicit_absence_required"]);
});

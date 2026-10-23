import { expect, test } from "@playwright/test";
import { label, mockApi, parsed, response, result } from "./fixtures";

// Use case 5: a view in the query, and the AP/PA filter overriding it.
const QUERY = "Show frontal PA images with lung opacity.";
const PARSED = parsed({
  semantic_query: "lung opacity",
  include: [{ concept: "Lung Opacity", status: "PRESENT" }],
  views: ["PA"],
});

function withProjection(projection: "AP" | "PA") {
  return [1, 2, 3].map((rank) =>
    result(rank, { projection, labels: [label("Lung Opacity", "PRESENT")] }),
  );
}

test("view: PA chip from the parser, PA badges, AP filter overrides the parsed view", async ({ page }) => {
  const { requests } = await mockApi(page, {
    search: (req) => response(req.mode, PARSED, withProjection(req.views?.includes("AP") ? "AP" : "PA")),
  });
  await page.goto("/");
  await page.getByTestId("query-input").fill(QUERY);
  await page.getByTestId("search-button").click();

  await expect(page.getByTestId("chip-view")).toHaveText("PA view");
  await expect(page.getByTestId("chip-include")).toHaveText("Lung Opacity");
  await expect(page.getByTestId("projection-badge")).toHaveText(["PA", "PA", "PA"]);
  expect(requests[0].views).toBeNull();

  await page.getByTestId("view-AP").click();
  await expect(page.getByTestId("view-AP")).toHaveAttribute("aria-pressed", "true");
  await expect(page.getByTestId("projection-badge")).toHaveText(["AP", "AP", "AP"]);
  // The parsed chip still reports what the query said; the request carried the override.
  await expect(page.getByTestId("chip-view")).toHaveText("PA view");
  expect(requests[1].views).toEqual(["AP"]);

  await page.getByTestId("view-AP").click();
  await expect(page.getByTestId("view-AP")).toHaveAttribute("aria-pressed", "false");
  expect(requests[2].views).toBeNull();
});

test("a result without projection shows no badge (older API)", async ({ page }) => {
  await mockApi(page, {
    search: (req) => {
      const r = result(1, { labels: [label("Lung Opacity", "PRESENT")] });
      delete r.projection;
      return response(req.mode, PARSED, [r]);
    },
  });
  await page.goto("/");
  await page.getByTestId("query-input").fill(QUERY);
  await page.getByTestId("search-button").click();
  await expect(page.getByTestId("result-card")).toHaveCount(1);
  await expect(page.getByTestId("projection-badge")).toHaveCount(0);
});

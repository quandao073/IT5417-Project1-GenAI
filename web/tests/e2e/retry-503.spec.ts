import { expect, test } from "@playwright/test";
import { label, mockApi, parsed, response, result } from "./fixtures";

const PARSED = parsed({ semantic_query: "cardiomegaly", include: [{ concept: "Cardiomegaly", status: "PRESENT" }] });
const NOT_READY = { status: 503, body: { detail: { status: "not_ready", reason: "qdrant: connection refused" } } };

test("503 not_ready: one automatic retry after 2 s, then an error banner whose Retry works", async ({ page }) => {
  const { requests } = await mockApi(page, {
    search: (req, call) =>
      call < 2 ? NOT_READY : response(req.mode, PARSED, [result(1, { labels: [label("Cardiomegaly", "PRESENT")] })]),
  });
  await page.goto("/");
  const started = Date.now();
  await page.getByTestId("example-query").first().click();

  const banner = page.getByTestId("error-banner");
  await expect(banner).toBeVisible({ timeout: 15_000 });
  await expect(banner).toContainText("API not ready: qdrant: connection refused");
  expect(Date.now() - started).toBeGreaterThanOrEqual(2_000);
  expect(requests).toHaveLength(2); // the call and its single retry
  await expect(page.getByTestId("result-card")).toHaveCount(0);

  await page.getByTestId("retry-button").click();
  await expect(page.getByTestId("result-card")).toHaveCount(1);
  await expect(banner).toHaveCount(0);
  expect(requests).toHaveLength(3);
});

test("other errors show the detail without retrying automatically", async ({ page }) => {
  const { requests } = await mockApi(page, {
    search: () => ({ status: 422, body: { detail: [{ msg: "query must not be blank" }] } }),
  });
  await page.goto("/");
  await page.getByTestId("example-query").first().click();
  await expect(page.getByTestId("error-banner")).toContainText("invalid request");
  expect(requests).toHaveLength(1);
});

test("the loading skeleton shows while the API answers", async ({ page }) => {
  await mockApi(page, {
    delayMs: 1_500,
    search: (req) => response(req.mode, PARSED, [result(1)]),
  });
  await page.goto("/");
  await page.getByTestId("example-query").first().click();
  await expect(page.getByTestId("loading")).toBeVisible();
  await expect(page.getByTestId("search-button")).toBeDisabled();
  await expect(page.getByTestId("result-card")).toHaveCount(1);
  await expect(page.getByTestId("loading")).toHaveCount(0);
});

test("API unreachable at load: the health badge says so", async ({ page }) => {
  await mockApi(page, {
    health: { status: 503, body: { status: "not_ready", reason: "loading encoder" } },
    search: (req) => response(req.mode, PARSED, []),
  });
  await page.goto("/");
  await expect(page.getByTestId("health-badge")).toHaveAttribute("data-state", "error", { timeout: 15_000 });
  await expect(page.getByTestId("health-badge")).toHaveAttribute("title", /loading encoder/);
});

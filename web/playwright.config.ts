import { defineConfig, devices } from "@playwright/test";

// The production build is served on its own port so a running `pnpm dev` (3000) is not touched.
// NEXT_PUBLIC_API_BASE_URL is emptied so the page calls /api/v1/* on its own origin, which the
// specs intercept with page.route; no API server is involved.
const PORT = 3100;

export default defineConfig({
  testDir: "./tests/e2e",
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  reporter: [["list"]],
  use: {
    baseURL: `http://localhost:${PORT}`,
    trace: "retain-on-failure",
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: {
    command: `pnpm build && pnpm start -p ${PORT}`,
    url: `http://localhost:${PORT}`,
    reuseExistingServer: !process.env.CI,
    timeout: 240_000,
    env: { NEXT_PUBLIC_API_BASE_URL: "" },
  },
});

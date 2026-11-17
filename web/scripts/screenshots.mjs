// Screenshots of the running demo (README, "Chạy demo"): results, detail and comparison pages for
// a few questions, into docs/demo/. They show CheXpert images, so they stay on this machine
// (docs/demo/README.md). Uses the installed Microsoft Edge: no browser download.
//   pnpm screenshots            (the web on :3000 and the API on :8000 must be running)
import { mkdirSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";

const WEB = process.env.WEB_URL ?? "http://localhost:3000";
const OUT = fileURLToPath(new URL("../../docs/demo/", import.meta.url));
const QUESTIONS = [
  ["vi", "ảnh X-quang có khí phế thũng"],
  ["negation", "no pneumothorax and no pleural effusion"],
  ["disease", "suy tim"],
  ["disease-signs", "heart failure with small pleural effusions"],
  ["unrecognised", "tràn dịch màng phổi và sỏi thận"],
  ["text-route", "sỏi thận"],
];

mkdirSync(OUT, { recursive: true });
const browser = await chromium.launch({ channel: process.env.PW_CHANNEL ?? "msedge" });
const page = await browser.newPage({ viewport: { width: 1600, height: 1000 } });
const summary = [];

// The page grown to its full height, every image loaded, then one shot: a fullPage shot stitches
// viewport-sized pieces, which leaves the sticky disclaimer mid-page and lazy images black.
async function shoot(file, scroller = "html") {
  const height = await page.locator(scroller).evaluate((el) => el.scrollHeight);
  await page.setViewportSize({ width: 1600, height: Math.max(1000, height + 40) });
  await page.waitForLoadState("networkidle");
  await page.waitForFunction(() => [...document.images].every((i) => i.complete));
  await page.screenshot({ path: `${OUT}${file}` });
  await page.setViewportSize({ width: 1600, height: 1000 });
}
for (const [name, q] of QUESTIONS) {
  const kg = page.waitForResponse((r) => r.url().endsWith("/api/v1/search/kg"), { timeout: 60_000 });
  await page.goto(`${WEB}/?q=${encodeURIComponent(q)}&k=10`);
  await kg;
  await page.waitForSelector(".grid, .panel.muted", { timeout: 60_000 });
  // the explanation streams in after the results; the text route has none
  await page.waitForSelector(".explain", { timeout: 45_000 }).catch(() => {});
  await shoot(`results-${name}.png`);
  summary.push({
    name, q,
    results: await page.locator(".grid .card").count(),
    misses: await page.locator(".tag.miss").allTextContents(),
    warnings: await page.locator(".tag.warn").allTextContents(),
    explanation: await page.locator(".explain .tag").first().textContent().catch(() => null),
  });
  if (name === "disease-signs") {
    await page.locator(".grid .card").first().click();
    await page.waitForSelector(".dialog img.big");
    await shoot(`detail-${name}.png`, ".overlay");
    await page.keyboard.press("Escape");
  }
}
for (const [name, q] of [QUESTIONS[2], QUESTIONS[1]]) {
  await page.goto(`${WEB}/compare?q=${encodeURIComponent(q)}&k=10`);
  await page.waitForSelector(".compare section:nth-child(3) .grid .card", { timeout: 90_000 });
  await shoot(`compare-${name}.png`);
  summary.push({ name: `compare-${name}`, q,
                 shared: await page.locator(".card.shared").count() });
}
await browser.close();
writeFileSync(`${OUT}screenshots.json`, JSON.stringify(summary, null, 1));
console.log(JSON.stringify(summary, null, 1));

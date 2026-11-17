// Screenshots of the demo script (docs/demo/kich-ban-demo.md), in its order, into docs/demo/.
// They show CheXpert images and UMLS names, so they stay on this machine (docs/demo/README.md).
// Uses the installed Microsoft Edge: no browser download. screenshots.json records, per question,
// what the system said (intents, warnings, explanation), to check the script against.
//   pnpm screenshots            (the web on :3000 and the API on :8000 must be running)
import { mkdirSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";

const WEB = process.env.WEB_URL ?? "http://localhost:3000";
const OUT = fileURLToPath(new URL("../../docs/demo/", import.meta.url));
// [file name, question, page]: the order of the demo script
const SCRIPT = [
  ["1-phu-dinh", "no pneumothorax and no pleural effusion", "compare"],
  ["2-benh", "suy tim", "search"],
  ["3-tieng-viet", "tràn khí màng phổi bên phải", "search"],
  ["4-ngoai-14-nhan", "phim có nốt phổi ở thùy trên", "search"],
  ["5-vung-giai-phau", "tổn thương ở góc sườn hoành", "search"],
  ["6-khong-nhan-ra", "tràn dịch màng phổi và sỏi thận", "search"],
  ["7-ngoai-pham-vi", "ảnh chó mèo", "search"],          // the spare question: a refusal (D79)
  ["7-ngoai-pham-vi-so-sanh", "ảnh chó mèo", "compare"],
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

for (const [name, q, kind] of SCRIPT) {
  const enc = encodeURIComponent(q);
  if (kind === "compare") {
    await page.goto(`${WEB}/compare?q=${enc}&k=10`);
    await page.waitForSelector(".compare section:nth-child(3) .grid .card, .compare .out-of-scope",
                               { timeout: 90_000 });
    await page.waitForSelector(".compare section:nth-child(3) .grid .card", { timeout: 30_000 });
    await shoot(`${name}.png`);
    summary.push({ name, q, shared: await page.locator(".card.shared").count(),
                   scales: await page.locator(".scale-note").allTextContents() });
    continue;
  }
  const kg = page.waitForResponse((r) => r.url().endsWith("/api/v1/search/kg"), { timeout: 60_000 });
  await page.goto(`${WEB}/?q=${enc}&k=10`);
  const events = (await (await kg).text()).trim().split("\n").map((l) => JSON.parse(l));
  await page.waitForSelector(".grid, .panel.muted, .out-of-scope", { timeout: 60_000 });
  await page.waitForSelector(".explain", { timeout: 45_000 }).catch(() => {});
  await shoot(`${name}.png`);
  const [res, ex] = events;
  summary.push({
    name, q,
    intents: res.intents.map((i) => ({
      term: i.term, cui: i.cui, name: i.name, vi: i.name_vi, kind: i.kind, polarity: i.polarity,
      columns: i.columns.map((c) => `${c.name_vi ?? c.name} ${Math.round(c.weight * 100)}%`) })),
    status: res.status, message: res.message ?? null, scope: res.trace.scope ?? null,
    translation: res.trace.translation ?? null, fusion: res.trace.fusion ?? null,
    warnings: res.warnings, degraded: res.degraded, fallback: res.trace.fallback ?? null,
    top: res.results.slice(0, 3).map((r) => ({ rank: r.rank, score: r.score, kg_z: r.kg_z,
                                              clip_part: r.clip_part })),
    explanation: ex && { status: ex.status, summary: ex.summary, dropped: ex.dropped },
  });
  if (name === "2-benh") {
    await page.locator(".grid .card").first().click();
    await page.waitForSelector(".dialog img.big");
    await shoot(`${name}-chi-tiet.png`, ".overlay");
    await page.keyboard.press("Escape");
  }
}
await browser.close();
writeFileSync(`${OUT}screenshots.json`, JSON.stringify(summary, null, 1));
console.log(JSON.stringify(summary, null, 1));

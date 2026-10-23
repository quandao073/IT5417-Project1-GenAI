# WP8 - web UI (Phase 6), branch `feat/lair-ui`

## What runs

- `web/`: Next.js 15.5 (App Router, React 19, TypeScript, Tailwind 4), pnpm 12, `output: "standalone"`,
  `engines.node >= 20`. Pages: `/` (search) and `/compare` (3-4 modes side by side, `?q=` prefills).
- `pnpm dev` (port 3000) talks to `NEXT_PUBLIC_API_BASE_URL` (default `http://localhost:8000`, see
  `web/.env.example`). The value is inlined at build time: set it before `pnpm build` / `docker build`.
  The browser calls the API directly, so the API must allow the web origin (CORS) unless both share an
  origin; `src/api/main.py` adds no CORS middleware today (see open questions).
- `pnpm test:e2e`: builds with `NEXT_PUBLIC_API_BASE_URL=""`, serves on port 3100, and every spec mocks
  `**/api/v1/**` with `page.route` - no API server is needed. 16 tests: one spec per use case of plan
  section 4.2 (uc1-uc6), `retry-503.spec.ts` (503 not_ready -> one retry after 2 s -> error banner ->
  Retry), `compare.spec.ts` (4 default columns, 3 after deselecting one, decomposed concept scores).
  Running it leaves `.next` built against an empty API base; run `pnpm build` again before `pnpm start`.
- `docker build -f docker/web.Dockerfile web` (2026-10-23): **succeeds**, image 196 MB; `docker run -p
  3199:3000` answered `GET /` 200 with the disclaimer in the HTML.

## Needs the real API (not exercised here)

- Every result, image and study detail. Without `GET /api/v1/studies/{id}` (older API, before WP5) the
  drawer shows the search result's constrained labels and evidence with a "Study detail unavailable" note.
- Optional fields from WP5 the UI reads when present: `SearchResult.projection` (AP/PA badge),
  `ScoreBreakdown.concept_scores` (drawer table), `/studies/{id}`, `HealthResponse.image_joint`,
  `.image_decomposed`, `.image_encoder` (health badge tooltip). Modes `image_joint` / `image_decomposed`
  are always offered; an older API rejects them with 422, shown in the error banner.

## Choices worth knowing

- `web/pnpm-workspace.yaml` holds two pnpm settings, so the Dockerfile copies it next to the lockfile
  (only Dockerfile change): `allowBuilds: {unrs-resolver: false}` (pnpm 12 fails the install on an
  unapproved build script) and `nodeLinker: hoisted` (`next build` standalone re-creates pnpm symlinks and
  fails with `EPERM: operation not permitted, symlink` on Windows without Developer Mode).
- `next.config.ts` pins `outputFileTracingRoot` to `web/` so `server.js` sits at the standalone root.
- `pnpm` is not on PATH by default (T5) and `corepack` is not shipped with Node 25: installed with
  `npm i -g pnpm` (12.15.0, `packageManager` in `package.json`; corepack in the Node 20 image honours it).
- `next start` prints a warning under `output: standalone` but serves the build; the e2e web server uses it.
- Score is shown as a raw number with "ranking signal, not a probability", never as a percentage; hidden
  when null (label mode).

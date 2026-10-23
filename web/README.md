# web

Next.js UI for the CXR semantic retrieval API (plan Phase 6, WP8).

```bash
pnpm install
cp .env.example .env.local        # NEXT_PUBLIC_API_BASE_URL, default http://localhost:8000
pnpm dev                          # http://localhost:3000
pnpm lint && pnpm build
pnpm exec playwright install chromium
pnpm test:e2e                     # mocked API; rebuilds .next with an empty API base, run `pnpm build` again before `pnpm start`
```

`docker build -f ../docker/web.Dockerfile .` from this folder builds the standalone image.
See `notes/lair-ui.md` at the repository root for what runs without the API.

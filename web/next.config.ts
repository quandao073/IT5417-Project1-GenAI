import type { NextConfig } from "next";
import { PHASE_DEVELOPMENT_SERVER } from "next/constants";

// standalone: docker/web.Dockerfile (Linux) copies .next/standalone. Off on Windows, where it
// needs symlink rights a normal account lacks; `pnpm dev` and `pnpm build` do not need it.
// `pnpm dev` writes to .next-dev: a `pnpm build` while the demo runs would otherwise overwrite
// the dev server's chunks ("Cannot find module './280.js'").
export default function config(phase: string): NextConfig {
  return {
    distDir: phase === PHASE_DEVELOPMENT_SERVER ? ".next-dev" : ".next",
    output: process.platform === "win32" ? undefined : "standalone",
    outputFileTracingRoot: process.cwd(),   // web/, not a lockfile found in a parent folder
    reactStrictMode: true,
    devIndicators: false,                   // no Next.js badge on the projected page
  };
}

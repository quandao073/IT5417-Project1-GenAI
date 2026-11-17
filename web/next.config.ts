import type { NextConfig } from "next";

// standalone: docker/web.Dockerfile (Linux) copies .next/standalone. Off on Windows, where it
// needs symlink rights a normal account lacks; `pnpm dev` and `pnpm build` do not need it.
const config: NextConfig = {
  output: process.platform === "win32" ? undefined : "standalone",
  outputFileTracingRoot: process.cwd(),     // web/, not a lockfile found in a parent folder
  reactStrictMode: true,
  devIndicators: false,                     // no Next.js badge on the projected page
};

export default config;

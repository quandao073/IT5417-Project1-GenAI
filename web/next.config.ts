import path from "node:path";
import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // docker/web.Dockerfile copies .next/standalone (server.js at its root) and .next/static.
  output: "standalone",
  // Keep server.js at the standalone root instead of inferring a workspace root above web/.
  outputFileTracingRoot: path.join(__dirname),
  reactStrictMode: true,
};

export default nextConfig;

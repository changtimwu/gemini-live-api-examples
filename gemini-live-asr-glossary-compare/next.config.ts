import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone",
  serverExternalPackages: ["@livekit/rtc-node", "ws"],
  // Allow dev requests (HMR, dev assets) from LAN clients and the Cloudflare tunnel host.
  allowedDevOrigins: ["192.168.1.146", "glossary-asr.wormhole.work"],
};

export default nextConfig;

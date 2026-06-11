import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone",
  serverExternalPackages: ["@livekit/rtc-node", "ws"],
  // Allow dev requests (HMR, dev assets) from LAN clients hitting this host by IP.
  allowedDevOrigins: ["192.168.1.146"],
};

export default nextConfig;

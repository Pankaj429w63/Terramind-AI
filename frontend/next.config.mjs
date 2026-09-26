/** @type {import('next').NextConfig} */
const backendOrigin = (process.env.TERRAMIND_BACKEND_API_URL || process.env.NEXT_PUBLIC_API_URL || "").replace(/\/+$/, "");

const nextConfig = {
  output: "standalone",
  async rewrites() {
    if (!backendOrigin) return [];
    return [
      { source: "/health", destination: `${backendOrigin}/health` },
      { source: "/api/:path*", destination: `${backendOrigin}/api/:path*` },
    ];
  },
};
export default nextConfig;

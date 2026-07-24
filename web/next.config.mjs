/** @type {import('next').NextConfig} */
const nextConfig = {
  // Fully static: no server, no API routes, no runtime data fetching. `next build`
  // emits web/out/, which Vercel (or any static host) serves as-is.
  output: 'export',
  images: { unoptimized: true },   // the image optimiser needs a server; we have none
  reactStrictMode: true,
};
export default nextConfig;

import type { NextConfig } from "next";

const config: NextConfig = {
  // The API host is read at build time so the same image can point at a local
  // backend or at the deployed Space without a code change.
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000",
  },
};

export default config;

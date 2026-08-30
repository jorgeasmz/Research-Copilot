import type { Metadata } from "next";

import "./globals.css";

export const metadata: Metadata = {
  title: "Research Copilot",
  description:
    "Grounded answers over the quantum cryptography literature on arXiv, citing the paragraph each claim comes from.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}

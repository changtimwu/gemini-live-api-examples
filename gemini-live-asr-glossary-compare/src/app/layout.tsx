import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "ASR Glossary Compare",
  description:
    "Side-by-side live transcription: glossary vs general system instruction (Gemini Live API + LiveKit).",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}

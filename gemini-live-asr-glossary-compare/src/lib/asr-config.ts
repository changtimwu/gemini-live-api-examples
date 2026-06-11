/**
 * Fixed A/B configuration for the ASR glossary comparison.
 *
 * Both variants stream the same mic audio into a separate Gemini Live session
 * (gemini-3.5-live-translate-preview). We read `input_transcription` as the
 * Chinese ASR and `output_transcription` as the English translation. The only
 * difference between the two arms is the systemInstruction:
 *
 *   - "glossary": pins the TWSE company names/tickers (see glossary.ts), or a
 *                 show-specific instruction file via GLOSSARY_SI_FILE (below)
 *   - "general":  same domain framing, but no specific company list
 *
 * Same domain context on both sides isolates the variable under test (the
 * glossary terms), matching the offline harness in ../../stock-asr-eval.
 *
 * Edit these instructions to taste, then restart `npm run dev` to apply.
 *
 * This module is server-only (imported solely by asr-bridge / asr-session-manager),
 * so reading a file at startup is safe — it never ships to the client bundle.
 */
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { buildGlossaryInstruction } from "./glossary";

export type Variant = "glossary" | "general";

export const VARIANTS: Variant[] = ["glossary", "general"];

export const MODEL = "gemini-3.5-live-translate-preview";
export const TARGET_LANGUAGE = "en"; // translate Chinese → English

/**
 * The glossary arm's system instruction. Defaults to the built-in TWSE list,
 * but set GLOSSARY_SI_FILE to a text file to load a show-specific instruction
 * instead — e.g. the output of ../stock-asr-eval/glossary_llm.py:
 *   GLOSSARY_SI_FILE=../stock-asr-eval/results/<videoId>.si.txt
 * Relative paths resolve against the dev-server cwd. Read once at startup; on
 * any error it falls back to the built-in glossary so the app still runs.
 */
function resolveGlossaryInstruction(): string {
  const file = process.env.GLOSSARY_SI_FILE?.trim();
  if (!file) return buildGlossaryInstruction();
  try {
    const text = readFileSync(resolve(process.cwd(), file), "utf-8").trim();
    if (!text) throw new Error("file is empty");
    console.log(`[asr-config] glossary instruction loaded from ${file} (${text.length} chars)`);
    return text;
  } catch (err) {
    console.warn(
      `[asr-config] GLOSSARY_SI_FILE=${file} unreadable (${(err as Error).message}); ` +
        "using built-in glossary"
    );
    return buildGlossaryInstruction();
  }
}

export const SYSTEM_INSTRUCTIONS: Record<Variant, string> = {
  glossary: resolveGlossaryInstruction(),
  general:
    "這是一段台灣股市分析的廣播，內容可能會提到台灣上市櫃公司的名稱與股票代號，" +
    "請盡量正確辨識並轉寫所有內容。",
};

export const VARIANT_LABELS: Record<Variant, string> = {
  glossary: "With glossary",
  general: "General",
};

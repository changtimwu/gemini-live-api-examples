/**
 * Fixed A/B configuration for the ASR glossary comparison.
 *
 * Both variants stream the same mic audio into a separate Gemini Live session
 * (gemini-3.5-live-translate-preview). We read `input_transcription` as the
 * Chinese ASR and `output_transcription` as the English translation. The only
 * difference between the two arms is the systemInstruction:
 *
 *   - "glossary": pins the TWSE company names/tickers (see glossary.ts)
 *   - "general":  same domain framing, but no specific company list
 *
 * Same domain context on both sides isolates the variable under test (the
 * glossary terms), matching the offline harness in ../../stock-asr-eval.
 *
 * Edit these instructions to taste, then restart `npm run dev` to apply.
 */
import { buildGlossaryInstruction } from "./glossary";

export type Variant = "glossary" | "general";

export const VARIANTS: Variant[] = ["glossary", "general"];

export const MODEL = "gemini-3.5-live-translate-preview";
export const TARGET_LANGUAGE = "en"; // translate Chinese → English

export const SYSTEM_INSTRUCTIONS: Record<Variant, string> = {
  glossary: buildGlossaryInstruction(),
  general:
    "這是一段台灣股市分析的廣播，內容可能會提到台灣上市櫃公司的名稱與股票代號，" +
    "請盡量正確辨識並轉寫所有內容。",
};

export const VARIANT_LABELS: Record<Variant, string> = {
  glossary: "With glossary",
  general: "General",
};

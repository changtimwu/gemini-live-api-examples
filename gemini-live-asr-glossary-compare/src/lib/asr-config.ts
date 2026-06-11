/**
 * Fixed A/B configuration for the ASR glossary comparison.
 *
 * Both variants stream the same mic audio into a separate Gemini Live session
 * (gemini-3.5-live-translate-preview) with the SAME light system instruction, so
 * both transcribe identically. The arms differ only afterwards:
 *
 *   - "general":  publish the raw Chinese ASR as-is.
 *   - "glossary": run each completed turn's Chinese ASR through a cheap rephrase
 *                 agent (gemini-3.1-flash-lite) that fixes domain-term mishearings
 *                 using the glossary knowledge — names, examples, and the phonetic/
 *                 negative cues that used to live in the translate model's SI.
 *
 * Moving the glossary rules from the recognizer's prompt to a post-transcribe
 * correction pass: a model that sees the whole phrase fixes true homophones
 * (外溢/外意, 精材/精彩) far better than priming the streaming recognizer does.
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
 * The glossary knowledge (company names + tickers + jargon + example sentences +
 * phonetic/negative cues). Defaults to the built-in TWSE list, but set
 * GLOSSARY_SI_FILE to a show-specific file — e.g. ../stock-asr-eval/glossary_llm.py's
 * output (../stock-asr-eval/results/<videoId>.si.txt). Read once at startup; on any
 * error it falls back to the built-in glossary. This text now drives the rephrase
 * agent (below) rather than the translate model's prompt.
 */
function resolveGlossaryKnowledge(): string {
  const file = process.env.GLOSSARY_SI_FILE?.trim();
  if (!file) return buildGlossaryInstruction();
  try {
    const text = readFileSync(resolve(process.cwd(), file), "utf-8").trim();
    if (!text) throw new Error("file is empty");
    console.log(`[asr-config] glossary knowledge loaded from ${file} (${text.length} chars)`);
    return text;
  } catch (err) {
    console.warn(
      `[asr-config] GLOSSARY_SI_FILE=${file} unreadable (${(err as Error).message}); ` +
        "using built-in glossary"
    );
    return buildGlossaryInstruction();
  }
}

// Light recognizer prompt — same for both arms, so they transcribe identically.
const BASE_INSTRUCTION =
  "這是一段台灣股市分析的廣播，內容可能會提到台灣上市櫃公司的名稱與股票代號，" +
  "請盡量正確辨識並轉寫所有內容。";

export const SYSTEM_INSTRUCTIONS: Record<Variant, string> = {
  glossary: BASE_INSTRUCTION,
  general: BASE_INSTRUCTION,
};

// --- Rephrase agent ---------------------------------------------------------
// A cheap text model re-reads each completed turn's Chinese transcription and
// corrects domain-term mishearings using the glossary knowledge. Only the arms
// listed here run it; the rest publish the raw ASR.
export const REPHRASE_MODEL = "gemini-3.1-flash-lite";
export const REPHRASE_VARIANTS: Variant[] = ["glossary"];
export const REPHRASE_INSTRUCTION =
  "你是台灣股市節目的『中文逐字稿校正員』。下面的『詞彙知識』列出節目會提到的公司名稱、股票代號、" +
  "專有名詞、用法範例，以及常見的同音誤聽與讀音提示。請依據它，只把使用者輸入的逐字稿中被誤聽的" +
  "名稱／專有名詞更正為正確寫法；其餘文字、語序、口語風格、標點一律原樣保留，不要翻譯、不要解釋、" +
  "不要新增或刪除其他內容。若沒有需要更正的地方，就原樣輸出。只輸出更正後的中文逐字稿本身。\n\n" +
  "詞彙知識：\n" +
  resolveGlossaryKnowledge();

export const VARIANT_LABELS: Record<Variant, string> = {
  glossary: "Tuned with glossary",
  general: "General",
};

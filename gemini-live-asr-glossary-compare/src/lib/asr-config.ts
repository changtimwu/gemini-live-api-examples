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
// The rephrase agent only needs correct names/tickers/terms + the precise negative cues. Drop:
//  - the example-sentence section (it regurgitates the examples),
//  - the English-translation map (it appends English glosses, e.g. "封測 (Packaging and Testing)"),
//  - the 讀音提示 (phonetic) section — "always write 欣銓 for this sound" is far too broad for a
//    context-aware agent and pulls in homophones (晶圓/安全/投信 → 欣銓). The exact-wrong-form
//    negative cues (特別注意) stay; they're precise.
function glossaryForRephrase(): string {
  return resolveGlossaryKnowledge()
    .split(/(?<=。)/)
    .filter(
      (s) =>
        !s.includes("用法範例") &&
        !s.includes("翻譯成英文") &&
        !s.includes("讀音提示")
    )
    .join("");
}

export const REPHRASE_INSTRUCTION =
  "你是台灣股市節目的『中文逐字稿校正員』。下面的『詞彙知識』列出節目會提到的公司名稱、股票代號、" +
  "專有名詞，以及常見的同音誤聽與讀音提示。請『只』針對我提供的這段逐字稿做兩件事：\n" +
  "1. 把明顯被誤聽、其實是詞彙知識中公司／專有名詞的地方，整個誤聽的詞一併換成正確寫法——" +
  "包含誤聽殘留的多餘字也要一起換掉，絕不可留下原本誤聽的任何字。只有在明確是在講某家公司／某個" +
  "專有名詞的語境下才更正；一般常用詞（例如「新聞」「測試」「需求」「基本面」「漲停」「投信」" +
  "「安全」「信心」）即使發音接近某個名稱，也絕不可改成公司名。讀音提示只在這種情況下參考使用。\n" +
  "2. 只在『公司名稱』後面加上『 (代號)』，代號必須是詞彙知識『公司清單』裡該公司自己標註的數字代號，" +
  "不可張冠李戴，例如「台積電」→「台積電 (2330)」、「日月光投控」→「日月光投控 (3711)」。" +
  "詞彙知識『專有名詞清單』裡的詞（例如封測、矽光子、CoWoS、先進封裝、ASIC、外溢、台指期）都不是公司，" +
  "一律不可加代號、也不可加任何括號——例如「封測族群」維持「封測族群」，" +
  "絕不可變成「封測 (3711)」或「封測 (封測)」。英文縮寫與一般詞同樣不加。" +
  "沒有代號的公司（例如外國公司）維持原樣。\n" +
  "範例（特別注意要清掉誤聽殘留字）：「台積電電它的」→「台積電 (2330) 它的」；" +
  "「那欣銓權來講」→「那欣銓 (3264) 來講」；「日月光投控光族群」→「日月光投控 (3711) 族群」。\n" +
  "嚴格要求：除上述兩點外，其他文字、語序、口語、標點一律原樣保留；不可翻譯、不可附加任何英文、" +
  "不可解釋、不可自行造句、不可加入詞彙知識裡的例句或清單。" +
  "詞彙知識只是查字典用的參考，絕對不可把輸入逐字稿中『沒有出現』的公司或名詞補進輸出——" +
  "例如輸入是「盟立、大銀微系統」，就只能輸出「盟立 (2464)、大銀微系統 (4576)」，" +
  "絕不可多出「穎崴 (6515)」等沒被提到的標的。" +
  "若這段只是片段或標點，直接原樣輸出，不要回問或說明。" +
  "輸出長度必須與輸入相近，只輸出處理後的這段逐字稿本身。\n\n" +
  "詞彙知識：\n" +
  glossaryForRephrase();

export const VARIANT_LABELS: Record<Variant, string> = {
  glossary: "Tuned with glossary",
  general: "General",
};

/**
 * A/B configuration for the ASR glossary comparison.
 *
 * Each arm ("general" and "glossary") is configured independently in the ARMS
 * table at the bottom of this file, on four axes:
 *
 *   1. model               — which Gemini Live recognizer model to stream into.
 *   2. systemInstruction    — the system instruction sent to that recognizer.
 *   3. rephraseModel         — the model for the post-transcribe correction pass,
 *                              or null for no rephrasing (publish the raw ASR).
 *   4. rephraseInstruction   — the system instruction for that rephrase agent.
 *
 * By default both arms use the same translate model + light instruction, so they
 * transcribe identically; the glossary arm then runs each completed turn through a
 * cheap rephrase agent (gemini-3.1-flash-lite) that fixes domain-term mishearings
 * using the glossary knowledge — names, examples, and the phonetic/negative cues.
 * A model that sees the whole phrase fixes true homophones (外溢/外意, 精材/精彩)
 * far better than priming the streaming recognizer does.
 *
 * Edit the ARMS table (and the instructions it references), then restart
 * `npm run dev` to apply.
 *
 * This module is server-only (imported solely by asr-bridge / asr-session-manager),
 * so reading a file at startup is safe — it never ships to the client bundle.
 */
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { buildGlossaryInstruction } from "./glossary";

export type Variant = "glossary" | "general";

export const VARIANTS: Variant[] = ["glossary", "general"];

// Recognizer model options — the new flash model sits beside the original translate model.
//   translate → does ASR *and* English translation natively (via translationConfig below).
//   flash     → does ASR; it can also translate, but only when its systemInstruction tells
//               it to (it has no translationConfig). See ARMS + asr-bridge's sendGeminiSetup.
export const LIVE_MODELS = {
  translate: "gemini-3.5-live-translate-preview",
  flash: "gemini-3.1-flash-live-preview",
} as const;

// translationConfig is specific to the translate model; only emit it for that model.
export function isTranslateModel(model: string): boolean {
  return model.includes("translate");
}

export const TARGET_LANGUAGE = "en"; // translate Chinese → English (translate model only)

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

// Light recognizer prompt — the default for both arms, so they transcribe identically.
const BASE_INSTRUCTION =
  "這是一段台灣股市分析的廣播，內容可能會提到台灣上市櫃公司的名稱與股票代號，" +
  "請盡量正確辨識並轉寫所有內容。";

// --- Rephrase agent ---------------------------------------------------------
// A cheap text model re-reads each completed turn's Chinese transcription and
// corrects domain-term mishearings using the glossary knowledge. Which arms run
// it (and with what model / instruction) is set per-arm in ARMS below; an arm
// with rephraseModel: null publishes the raw ASR unchanged.
// The rephrase agent only needs correct names/tickers/terms + the precise negative cues. Drop:
//  - the example-sentence section (it regurgitates the examples) — UNLESS GLOSSARY_KEEP_EXAMPLES=1,
//    which keeps 用法範例 because its in-context sentences disambiguate look-alike names the negative
//    cues alone miss (e.g. 創意→欣銓 in "…京元電子精材跟欣銓", where 創意 is itself a real company),
//  - the English-translation map (it appends English glosses, e.g. "封測 (Packaging and Testing)"),
//  - the 讀音提示 (phonetic) section — "always write 欣銓 for this sound" is far too broad for a
//    context-aware agent and pulls in homophones (晶圓/安全/投信 → 欣銓). The exact-wrong-form
//    negative cues (特別注意) stay; they're precise.
function glossaryForRephrase(): string {
  const keepExamples = process.env.GLOSSARY_KEEP_EXAMPLES === "1";
  return resolveGlossaryKnowledge()
    .split(/(?<=。)/)
    .filter(
      (s) =>
        (keepExamples || !s.includes("用法範例")) &&
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

// --- Per-arm configuration --------------------------------------------------
// The single source of truth for how each arm runs. Edit this table (and the
// instructions it references) and restart `npm run dev` to apply.
export interface ArmConfig {
  /** Recognizer (Gemini Live) model. Pick from LIVE_MODELS, or any Live model id. */
  model: string;
  /**
   * System instruction sent to the recognizer. For a non-translate model
   * (e.g. LIVE_MODELS.flash), add a "translate to English" directive here if you
   * want the English column populated — it won't translate on its own.
   */
  systemInstruction: string;
  /** Post-transcribe correction model, or null for no rephrasing (raw ASR is published). */
  rephraseModel: string | null;
  /** System instruction for the rephrase agent (ignored when rephraseModel is null). */
  rephraseInstruction: string;
}

export const ARMS: Record<Variant, ArmConfig> = {
  // General arm: plain ASR, no glossary correction.
  general: {
    model: LIVE_MODELS.translate,
    systemInstruction: BASE_INSTRUCTION,
    rephraseModel: null,
    rephraseInstruction: "",
  },
  // Glossary arm: same recognizer, but each turn is corrected by the rephrase
  // agent using the glossary knowledge.
  glossary: {
    model: LIVE_MODELS.translate,
    systemInstruction: BASE_INSTRUCTION,
    rephraseModel: "gemini-3.1-flash-lite",
    rephraseInstruction: REPHRASE_INSTRUCTION,
  },
};

export const VARIANT_LABELS: Record<Variant, string> = {
  glossary: "Tuned with glossary",
  general: "General",
};

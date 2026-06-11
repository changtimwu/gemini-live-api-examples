/**
 * Curated TWSE / TPEx glossary for the A/B comparison.
 *
 * These are well-known Taiwan-listed companies whose Chinese names are commonly
 * mis-heard by a general recognizer (homophones, abbreviations) but snap into
 * place once the model is told the name↔ticker mapping. The list is intentionally
 * small and hand-picked — edit it freely to match the broadcast you're testing.
 *
 * The full TWSE dictionary lives in ../../stock-asr-eval/tw_stocks.json; this is a
 * demo-sized subset. `english` is reference-only (the translate model uses the
 * Chinese name + ticker for ASR pinning).
 */
export interface StockTerm {
  name: string; // canonical Chinese name (as spoken)
  ticker: string; // TWSE/TPEx code
  english: string; // internationally recognized name
}

export const GLOSSARY: StockTerm[] = [
  { name: "台積電", ticker: "2330", english: "TSMC" },
  { name: "聯發科", ticker: "2454", english: "MediaTek" },
  { name: "鴻海", ticker: "2317", english: "Hon Hai / Foxconn" },
  { name: "聯電", ticker: "2303", english: "UMC" },
  { name: "日月光", ticker: "3711", english: "ASE Technology" },
  { name: "華邦電", ticker: "2344", english: "Winbond" },
  { name: "南亞科", ticker: "2408", english: "Nanya Technology" },
  { name: "威剛", ticker: "3260", english: "ADATA" },
  { name: "宜鼎", ticker: "5289", english: "Innodisk" },
  { name: "穩懋", ticker: "3105", english: "WIN Semiconductors" },
  { name: "創意", ticker: "3443", english: "Global Unichip (GUC)" },
  { name: "世芯", ticker: "3661", english: "Alchip" },
  { name: "智原", ticker: "3035", english: "Faraday" },
  { name: "緯創", ticker: "3231", english: "Wistron" },
  { name: "廣達", ticker: "2382", english: "Quanta" },
  { name: "大立光", ticker: "3008", english: "Largan Precision" },
  { name: "台達電", ticker: "2308", english: "Delta Electronics" },
  { name: "中華電", ticker: "2412", english: "Chunghwa Telecom" },
  { name: "富邦金", ticker: "2881", english: "Fubon Financial" },
  { name: "長榮", ticker: "2603", english: "Evergreen Marine" },
];

/**
 * Build the Chinese ASR glossary system instruction.
 * Mirrors stock-asr-eval/glossary.py:build_system_instruction so the on-screen
 * comparison matches the offline eval harness.
 */
export function buildGlossaryInstruction(terms: StockTerm[] = GLOSSARY): string {
  const items = terms.map((t) => `${t.name}（${t.ticker}）`).join("、");
  return (
    "這是一段台灣股市分析的廣播。內容會提到下列台灣上市櫃公司，" +
    "請正確辨識並轉寫這些公司的中文名稱與股票代號：" +
    items +
    "。"
  );
}

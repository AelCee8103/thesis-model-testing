// Dependency-free RFC-4180 CSV parsing + serialization.
// Handles quoted fields containing commas, escaped quotes (""), and CRLF/LF newlines.

import type { Dimension } from "@/types/api";

// Encodings tried (in order) when there is no BOM and the bytes are not UTF-16.
// UTF-8 is preferred; the Chinese encodings cover the CVAT dataset; windows-1252
// is the last-resort single-byte fallback. UTF-16 is detected separately below.
const CANDIDATE_ENCODINGS = ["utf-8", "big5", "gb18030", "windows-1252"];

/** Count chars that signal a wrong decoding: replacement chars + stray control bytes. */
function decodeBadness(text: string): number {
  let bad = 0;
  for (let i = 0; i < text.length; i++) {
    const c = text.charCodeAt(i);
    if (c === 0xfffd) bad += 1; // U+FFFD replacement
    else if (c === 0x7f || c < 0x09 || (c > 0x0d && c < 0x20)) bad += 1; // control (keep \t \n \r)
  }
  return bad;
}

/**
 * Decode raw file bytes into text, sniffing the encoding instead of assuming UTF-8
 * (File.text() always assumes UTF-8, which mangles UTF-16 / legacy-Chinese CSVs).
 * Returns the decoded text and the chosen encoding label.
 */
export function decodeBytes(buffer: ArrayBuffer): { text: string; encoding: string } {
  const bytes = new Uint8Array(buffer);

  // 1. BOM sniffing — authoritative when present.
  if (bytes[0] === 0xff && bytes[1] === 0xfe)
    return { text: new TextDecoder("utf-16le").decode(buffer), encoding: "utf-16le" };
  if (bytes[0] === 0xfe && bytes[1] === 0xff)
    return { text: new TextDecoder("utf-16be").decode(buffer), encoding: "utf-16be" };
  if (bytes[0] === 0xef && bytes[1] === 0xbb && bytes[2] === 0xbf)
    return { text: new TextDecoder("utf-8").decode(buffer), encoding: "utf-8" };

  // 2. UTF-16 without BOM — detected via a high ratio of NUL bytes.
  let zeros = 0;
  let evenZeros = 0;
  const scanLen = Math.min(bytes.length, 4096);
  for (let i = 0; i < scanLen; i++) {
    if (bytes[i] === 0) {
      zeros += 1;
      if (i % 2 === 0) evenZeros += 1;
    }
  }
  if (scanLen > 0 && zeros / scanLen > 0.1) {
    const enc = evenZeros > zeros - evenZeros ? "utf-16be" : "utf-16le";
    return { text: new TextDecoder(enc).decode(buffer), encoding: enc };
  }

  // 3. Single/multi-byte candidates — pick the decoding with the fewest bad chars.
  let best = { text: "", encoding: "utf-8", score: Number.POSITIVE_INFINITY };
  for (const enc of CANDIDATE_ENCODINGS) {
    let text: string;
    try {
      text = new TextDecoder(enc).decode(buffer);
    } catch {
      continue;
    }
    const score = decodeBadness(text);
    if (score < best.score) best = { text, encoding: enc, score };
    if (score === 0) break; // perfect decode, stop early (UTF-8 wins ties)
  }
  return { text: best.text, encoding: best.encoding };
}

/** Parse CSV text into a 2D array of string cells. Strips a leading BOM. */
export function parseCsv(input: string): string[][] {
  if (input.charCodeAt(0) === 0xfeff) input = input.slice(1); // strip BOM

  const rows: string[][] = [];
  let row: string[] = [];
  let field = "";
  let inQuotes = false;
  let i = 0;

  const pushField = () => {
    row.push(field);
    field = "";
  };
  const pushRow = () => {
    pushField();
    rows.push(row);
    row = [];
  };

  while (i < input.length) {
    const c = input[i];

    if (inQuotes) {
      if (c === '"') {
        if (input[i + 1] === '"') {
          field += '"';
          i += 2;
          continue;
        }
        inQuotes = false;
        i++;
        continue;
      }
      field += c;
      i++;
      continue;
    }

    if (c === '"') {
      inQuotes = true;
      i++;
      continue;
    }
    if (c === ",") {
      pushField();
      i++;
      continue;
    }
    if (c === "\r") {
      if (input[i + 1] === "\n") i++;
      pushRow();
      i++;
      continue;
    }
    if (c === "\n") {
      pushRow();
      i++;
      continue;
    }
    field += c;
    i++;
  }

  // Flush the trailing field/row (file may not end in a newline).
  if (field.length > 0 || row.length > 0) pushRow();
  return rows;
}

/**
 * Extract a single column of texts from CSV input.
 * If the first row contains a `text` cell (case-insensitive) that column is used,
 * otherwise the first column of a headerless file is used. Empty cells are skipped.
 */
export function parseTextColumn(input: string): string[] {
  const rows = parseCsv(input).filter((r) => r.some((c) => c.trim().length > 0));
  if (rows.length === 0) return [];

  const header = rows[0];
  const headerIdx = header.findIndex((h) => h.trim().toLowerCase() === "text");

  const start = headerIdx >= 0 ? 1 : 0;
  const col = headerIdx >= 0 ? headerIdx : 0;

  const out: string[] = [];
  for (let r = start; r < rows.length; r++) {
    const cell = (rows[r][col] ?? "").trim();
    if (cell.length > 0) out.push(cell);
  }
  return out;
}

const NEEDS_QUOTE = /[",\r\n]/;

/** Serialize a single value as a CSV cell, quoting only when required. */
export function csvCell(value: string | number): string {
  const s = String(value);
  return NEEDS_QUOTE.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

/** Serialize a 2D array of values into RFC-4180 CSV text (CRLF line endings). */
export function toCsv(rows: (string | number)[][]): string {
  return rows.map((r) => r.map(csvCell).join(",")).join("\r\n");
}

export interface EvalRow {
  text: string;
  truth: number;
}

export interface EvalParse {
  rows: EvalRow[];
  dimension: Dimension | null; // derived from which true_* column is present
  skipped: number; // rows dropped for empty text or non-numeric truth
  error: string | null;
}

/**
 * Parse an evaluation CSV: requires a `text` column plus a `true_valence` OR
 * `true_arousal` column (case-insensitive). The present truth column sets the
 * dimension. All other columns are ignored.
 */
export function parseEvalRows(input: string): EvalParse {
  const table = parseCsv(input).filter((r) => r.some((c) => c.trim().length > 0));
  if (table.length === 0) {
    return { rows: [], dimension: null, skipped: 0, error: "File is empty." };
  }

  const header = table[0].map((h) => h.trim().toLowerCase());
  const textIdx = header.indexOf("text");
  if (textIdx < 0) {
    return { rows: [], dimension: null, skipped: 0, error: "CSV needs a `text` column." };
  }

  const valenceIdx = header.indexOf("true_valence");
  const arousalIdx = header.indexOf("true_arousal");
  let dimension: Dimension;
  let truthIdx: number;
  if (valenceIdx >= 0) {
    dimension = "valence";
    truthIdx = valenceIdx;
  } else if (arousalIdx >= 0) {
    dimension = "arousal";
    truthIdx = arousalIdx;
  } else {
    return {
      rows: [],
      dimension: null,
      skipped: 0,
      error: "CSV needs a `true_valence` or `true_arousal` column.",
    };
  }

  const rows: EvalRow[] = [];
  let skipped = 0;
  for (let r = 1; r < table.length; r++) {
    const text = (table[r][textIdx] ?? "").trim();
    const truth = Number((table[r][truthIdx] ?? "").trim());
    if (text.length === 0 || !Number.isFinite(truth)) {
      skipped += 1;
      continue;
    }
    rows.push({ text, truth });
  }
  return { rows, dimension, skipped, error: null };
}

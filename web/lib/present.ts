// How the page words what the API returns. No diagnosis: a score says how well an image fits
// the question (docs/v2-kg-design.md, "Rào chắn").

import type { Intent, IntentScore } from "./api";

// Fixed on every page (docs/v2-kg-design.md, "Rào chắn"): kept word for word, in English.
export const DISCLAIMER =
  "Research and educational prototype. Not for diagnosis or clinical decision-making.";

// From docs/v2-6-smoke.txt, never from the benchmark: EN and VI, a negation, a disease,
// a question with a phrase the KG does not know.
export const EXAMPLES: { text: string; note: string }[] = [
  { text: "ảnh X-quang có khí phế thũng", note: "tiếng Việt" },
  { text: "no pneumothorax and no pleural effusion", note: "phủ định" },
  { text: "suy tim", note: "bệnh, suy từ dấu hiệu" },
  { text: "heart failure with small pleural effusions", note: "bệnh + dấu hiệu" },
  { text: "không có nốt phổi nhưng có khí phế thũng", note: "phủ định, tiếng Việt" },
  { text: "tràn dịch màng phổi và sỏi thận", note: "có cụm không nhận ra" },
  { text: "tổn thương ở góc sườn hoành", note: "vùng giải phẫu" },
  { text: "pacemaker leads", note: "thiết bị" },
];

export const KIND: Record<Intent["kind"], string> = {
  direct: "chấm thẳng bằng cột của nó",
  wider: "mở rộng xuống các khái niệm con",
  narrower: "hạ về khái niệm cha có cột",
  disease: "suy từ các dấu hiệu gợi ý",
  anatomy: "theo các dấu hiệu thường gặp ở vùng",
  none: "không chấm được",
};

export const COMBINE: Record<Intent["combine"], string> = {
  single: "một cột",
  noisy_or: "gộp: có một trong các cột (noisy-OR)",
  weighted: "gộp có trọng số",
  weighted_noisy_or: "gộp: noisy-OR có trọng số",
};

export const pct = (p: number) => `${Math.round(p * 100)}%`;

/** "tên VI (EN)" when the KG has a Vietnamese form, else the English name. */
export function both(vi: string | null | undefined, en: string): string {
  return vi && vi.toLowerCase() !== en.toLowerCase() ? `${vi} (${en})` : en;
}

export interface Part {
  label: string;      // what the column stands for, with "không" for an excluded concept
  value: number;      // how well the image fits that part: p, or 1 - p when excluded
  weight: number;
}

/** The columns that weigh most for one result: p x weight for a wanted concept, 1 - p for an
 * excluded one (the image fits "no X" when X scores low). */
export function topParts(intents: Intent[], scores: IntentScore[], n = 3): Part[] {
  const parts: (Part & { key: number })[] = [];
  for (const s of scores) {
    const intent = intents[s.intent];
    if (!intent) continue;
    for (const c of s.columns) {
      const col = intent.columns.find((x) => x.cui === c.cui);
      if (!col) continue;
      const absent = intent.polarity === "ABSENT";
      const name = col.name_vi ?? col.name;
      parts.push({
        label: absent ? `không ${name}` : name,
        value: absent ? 1 - c.p : c.p,
        weight: col.weight,
        key: absent ? 1 - c.p : c.p * col.weight,
      });
    }
  }
  return parts.sort((a, b) => b.key - a.key).slice(0, n)
    .map(({ label, value, weight }) => ({ label, value, weight }));
}

/** Split explanation text on its [K#] citations: text and citation pieces, in order. */
export function citations(text: string): { text: string; cite?: string }[] {
  const out: { text: string; cite?: string }[] = [];
  let last = 0;
  for (const m of text.matchAll(/\[(K\d+)\]/g)) {
    if (m.index! > last) out.push({ text: text.slice(last, m.index) });
    out.push({ text: m[0], cite: m[1] });
    last = m.index! + m[0].length;
  }
  if (last < text.length) out.push({ text: text.slice(last) });
  return out;
}

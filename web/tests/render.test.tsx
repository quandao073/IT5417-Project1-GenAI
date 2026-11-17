import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import { Explanation } from "@/components/Explanation";
import { ResultCard } from "@/components/ResultGrid";
import { Understanding } from "@/components/Understanding";
import type { ExplanationEvent, Intent, ResultsEvent } from "@/lib/api";
import { citations, DISCLAIMER, EXAMPLES, topParts } from "@/lib/present";

afterEach(cleanup);

const col = (cui: string, name: string, name_vi: string | null, weight = 1) =>
  ({ cui, column: name, weight, quality: 0.8, weak: false, borderline: false, name, name_vi });

const INTENTS: Intent[] = [
  { term: "suy tim", cui: "C1", name: "Heart failure", name_vi: "suy tim", polarity: "PRESENT",
    kind: "disease", combine: "weighted",
    columns: [col("C2", "Cardiomegaly", "tim to", 0.6), col("C3", "Edema", "phù phổi", 0.4)],
    path: ["Cardiomegaly —SUGGESTS(p=0.0062)→ Heart failure",
           "Edema —SUGGESTS(p=0.0030)→ Heart failure"],
    warnings: ["Suy từ dấu hiệu (Heart failure)."] },
  { term: "no pneumothorax", cui: "C4", name: "Pneumothorax", name_vi: "tràn khí màng phổi",
    polarity: "ABSENT", kind: "direct", combine: "single",
    columns: [col("C4", "Pneumothorax", "tràn khí màng phổi")], path: ["Pneumothorax"], warnings: [] },
  { term: "sỏi thận", cui: "", name: "sỏi thận", name_vi: null, polarity: "PRESENT", kind: "none",
    combine: "single", columns: [], path: [], warnings: ["Không chấm được \"sỏi thận\""] },
];

const RESULTS: ResultsEvent = {
  event: "results", request_id: "r", kg_build_id: "kg_x", matrix_version: "m", build_id: "b",
  versions: {}, language: "vi", intents: INTENTS,
  warnings: ["Suy từ dấu hiệu (Heart failure).", "Không chấm được \"sỏi thận\""], degraded: [],
  results: [{ rank: 1, study_id: "p/s1", image_id: "img1", image_url: "/api/v1/images/img1",
              view: "AP", score: 1.5,
              intents: [{ intent: 0, p: 0.8, columns: [{ cui: "C2", p: 0.9 }, { cui: "C3", p: 0.5 }] },
                        { intent: 1, p: 0.1, columns: [{ cui: "C4", p: 0.1 }] }] }],
  timings_ms: {}, trace: { terms: [{ term: "trái", skipped: "modifier" }] }, disclaimer: "",
};

describe("presentation", () => {
  it("ranks the parts by weight x p, an excluded one by 1 - p", () => {
    const parts = topParts(INTENTS, RESULTS.results[0].intents);
    expect(parts.map((p) => p.label)).toEqual(["không tràn khí màng phổi", "tim to", "phù phổi"]);
    expect(parts[0].value).toBeCloseTo(0.9);
  });

  it("splits [K#] citations out of the text", () => {
    expect(citations("A [K1] b [K12].")).toEqual([
      { text: "A " }, { text: "[K1]", cite: "K1" }, { text: " b " }, { text: "[K12]", cite: "K12" },
      { text: "." }]);
  });

  it("keeps the disclaimer word for word and has EN and VI examples", () => {
    expect(DISCLAIMER).toBe(
      "Research and educational prototype. Not for diagnosis or clinical decision-making.");
    expect(EXAMPLES.length).toBeGreaterThanOrEqual(6);
    expect(EXAMPLES.length).toBeLessThanOrEqual(8);
  });
});

describe("components", () => {
  it("the understanding panel shows concepts, columns with weights, warnings and misses", () => {
    render(<Understanding data={RESULTS} />);
    expect(screen.getByText("suy tim (Heart failure)")).toBeTruthy();
    expect(screen.getByText("C1")).toBeTruthy();
    expect(screen.getByText("loại trừ")).toBeTruthy();
    expect(screen.getByText(/60%/)).toBeTruthy();
    expect(screen.getByText(/Không nhận ra/)).toBeTruthy();
    expect(screen.getByText("Suy từ dấu hiệu (Heart failure).")).toBeTruthy();
    expect(screen.getByText(/Không xét: “trái”/)).toBeTruthy();
  });

  it("a fused B2 card shows the two terms of its score, a B1 card its cosine", () => {
    render(<ResultCard r={{ ...RESULTS.results[0], score: 2.13, kg_z: 1.93, clip_part: 0.2 }}
                       intents={INTENTS} />);
    expect(screen.getByText("điểm trộn 2.13").getAttribute("title")).toMatch(/z\(KG\) \+ 0,2·z\(CLIP\)/);
    expect(screen.getByText("z(KG) 1.93")).toBeTruthy();
    expect(screen.getByText("CLIP +0.20")).toBeTruthy();
    cleanup();
    render(<ResultCard r={{ ...RESULTS.results[0], intents: [] }} intents={[]} scale="cosine" />);
    expect(screen.getByText("cosine 1.50")).toBeTruthy();
    expect(screen.queryByText(/z\(KG\)/)).toBeNull();
  });

  it("a result card shows rank, view, score and the top parts", () => {
    render(<ResultCard r={RESULTS.results[0]} intents={INTENTS} />);
    expect(screen.getByText("#1")).toBeTruthy();
    expect(screen.getByText("AP")).toBeTruthy();
    expect(screen.getByText("điểm 1.50")).toBeTruthy();
    expect(screen.getByText("tim to")).toBeTruthy();
  });

  it("the explanation shows each [K#] item on hover and the raw path when it fell back", () => {
    const ev: ExplanationEvent = {
      event: "explanation", request_id: "r", status: "partial", summary: "Tóm tắt.",
      images: [{ rank: 1, line: "Ảnh 1 khớp." }], knowledge: "Tim to hay đi cùng [K1].",
      knowledge_items: [{ id: "K1", text: "Mục K1 nói về tim to." }], fallback: null,
    };
    render(<Explanation event={ev} loading={false} intents={INTENTS} />);
    expect(screen.getByText("Tóm tắt.")).toBeTruthy();
    expect(screen.getByText("Ảnh 1 khớp.")).toBeTruthy();
    expect(screen.getByRole("tooltip").textContent).toBe("Mục K1 nói về tim to.");
    // partial: the path too, edges in words, the normalised weight instead of the raw p
    const path = document.querySelector(".path")!.textContent!;
    expect(path).toContain("Cardiomegaly —gợi ý→ Heart failure");
    expect(path).not.toMatch(/p=|SUGGESTS/);
    const weight = screen.getByText(/trọng số 60%/);
    expect(weight.getAttribute("title")).toMatch(/cùng xuất hiện trong report/);
  });

  it("while the explanation is on its way, the panel says so", () => {
    render(<Explanation event={null} loading intents={INTENTS} />);
    expect(screen.getByText(/Đang viết diễn giải/)).toBeTruthy();
  });
});

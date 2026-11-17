"use client";

import { useEffect, useRef, useState } from "react";

import type { KGResult, ResultsEvent } from "@/lib/api";
import { imageSrc, studyReport } from "@/lib/api";
import { both, KIND, pct } from "@/lib/present";

const VERSION_KEYS: [string, string][] = [
  ["kg_build_id", "KG build"],
  ["matrix_version", "Ma trận điểm"],
  ["build_id", "Build ảnh"],
  ["encoder", "Encoder ảnh"],
  ["parser_model", "Model parser"],
  ["parser_prompt_sha256", "sha prompt parser"],
  ["select_model", "Model chọn khái niệm"],
  ["select_prompt_sha256", "sha prompt chọn"],
  ["explain_model", "Model diễn giải"],
  ["explain_prompt_sha256", "sha prompt diễn giải"],
  ["fusion_weight", "Trọng số trộn cosine"],
];

export function Detail({ r, data, line, showReport, onClose }: {
  r: KGResult; data: ResultsEvent; line?: string; showReport: boolean; onClose: () => void;
}) {
  const [report, setReport] = useState<string | null>(null);
  const [reportError, setReportError] = useState<string | null>(null);
  const close = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    close.current?.focus();
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const byIntent = new Map((r.intents ?? []).map((s) => [s.intent, s]));
  return (
    <div className="overlay" role="dialog" aria-modal="true" aria-labelledby="detail-title"
         onClick={(e) => e.target === e.currentTarget && onClose()}>
      <div className="dialog">
        <header>
          <h2 id="detail-title">Ảnh hạng #{r.rank} · view {r.view} · điểm {r.score.toFixed(2)}</h2>
          <button ref={close} type="button" onClick={onClose}>Đóng (Esc)</button>
        </header>
        <div>
          <img className="big" src={imageSrc(r.image_url)} alt={`Ảnh X-quang hạng ${r.rank}`} />
          <p className="small muted">study {r.study_id} · image {r.image_id.slice(0, 16)}…</p>
          {showReport && (
            <div>
              <button type="button" onClick={() => studyReport(r.study_id).then(setReport, (e) => setReportError(String(e)))}>
                Xem report gốc (chỉ để đối chiếu)
              </button>
              {report && (
                <>
                  <p className="small muted">Report không dùng để xếp hạng; chỉ hiện để đối chiếu.</p>
                  <div className="report">{report}</div>
                </>
              )}
              {reportError && <p className="error">{reportError}</p>}
            </div>
          )}
        </div>
        <div>
          {line && (<><h3>Diễn giải cho ảnh này</h3><p>{line}</p></>)}
          <h3>Điểm từng cột</h3>
          {data.trace.fallback ? (
            <p className="muted small">Trang này xếp theo văn bản (không qua KG), nên không có điểm cột.</p>
          ) : (
            <table>
              <thead><tr><th>Ý của câu hỏi</th><th>Cột</th><th>Trọng số</th><th>Mức khớp</th></tr></thead>
              <tbody>
                {data.intents.map((it, i) => {
                  const s = byIntent.get(i);
                  if (!s) return null;
                  return it.columns.map((c, j) => {
                    const p = s.columns.find((x) => x.cui === c.cui)?.p;
                    return (
                      <tr key={`${i}-${c.cui}`}>
                        {j === 0 && (
                          <td rowSpan={it.columns.length}>
                            {it.polarity === "ABSENT" ? "không: " : ""}{both(it.name_vi, it.name)}
                            <div className="small muted">{KIND[it.kind]} · cả ý {pct(s.p)}</div>
                          </td>
                        )}
                        <td>{c.name_vi ?? c.name}{c.weak && <span className="tag warn">tin cậy thấp</span>}</td>
                        <td>{pct(c.weight)}</td>
                        <td>{p === undefined ? "?" : pct(p)}</td>
                      </tr>
                    );
                  });
                })}
              </tbody>
            </table>
          )}
          <p className="small muted">
            Mức khớp là xác suất đã hiệu chỉnh của model ảnh cho từng cột; với ý “không”, ảnh càng khớp khi
            mức này càng thấp. Đây là độ khớp truy vấn, không phải kết luận về ảnh.
          </p>
          <h3>Đường đi trên KG</h3>
          {data.intents.filter((i) => i.path.length).map((it, i) => (
            <div key={i} className="path">{`“${it.term}”\n  ${it.path.join("\n  ")}`}</div>
          ))}
          <h3>Phiên bản</h3>
          <dl className="versions">
            {VERSION_KEYS.filter(([k]) => data.versions[k] != null).map(([k, label]) => (
              <div key={k} style={{ display: "contents" }}>
                <dt>{label}</dt>
                <dd>{String(data.versions[k]).slice(0, k.endsWith("sha256") ? 12 : undefined)}</dd>
              </div>
            ))}
          </dl>
        </div>
      </div>
    </div>
  );
}

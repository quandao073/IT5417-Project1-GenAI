import type { ResultsEvent } from "@/lib/api";
import { both, COMBINE, KIND, pct } from "@/lib/present";

/** "Hệ thống hiểu câu hỏi": each phrase -> the chosen concept -> how it was expanded -> the
 * columns scored, with their weights; warnings as tags; phrases not recognised stand out. */
export function Understanding({ data }: { data: ResultsEvent }) {
  const own = new Set(data.intents.flatMap((i) => i.warnings));
  const pageWarnings = data.warnings.filter((w) => !own.has(w));
  const skipped = (data.trace.terms ?? []).filter((t) => t.skipped);
  return (
    <section className="panel" aria-labelledby="understanding">
      <h2 id="understanding">Hệ thống hiểu câu hỏi</h2>
      {data.intents.length === 0 && <p className="muted">Không tách được cụm nào có khái niệm.</p>}
      {data.intents.map((it, i) => (
        <div key={i} className={`intent ${it.kind === "none" ? "none" : ""}`}>
          <div className="flow">
            <span className="term">“{it.term}”</span>
            <span className="arrow" aria-hidden>→</span>
            {it.kind === "none" ? (
              <span className="tag miss">Không nhận ra: không khái niệm nào của KG khớp</span>
            ) : (
              <>
                <span>{both(it.name_vi, it.name)}</span>
                <span className="cui">{it.cui}</span>
                <span className={`tag ${it.polarity === "ABSENT" ? "info" : "ok"}`}>
                  {it.polarity === "ABSENT" ? "loại trừ" : "cần có"}
                </span>
                <span className="arrow" aria-hidden>→</span>
                <span>{KIND[it.kind]}</span>
                {it.columns.length > 1 && <span className="muted small">· {COMBINE[it.combine]}</span>}
              </>
            )}
          </div>
          {it.columns.length > 0 && (
            <div className="cols" aria-label="Các cột được chấm">
              {it.columns.map((c) => (
                <span key={c.cui} className="col" title={`AUROC kiểm định ${c.quality?.toFixed(2) ?? "?"}`}>
                  {c.name_vi ?? c.name}
                  {it.columns.length > 1 && <strong> · {pct(c.weight)}</strong>}
                  {c.weak && <span className="tag warn"> độ tin cậy thấp</span>}
                </span>
              ))}
            </div>
          )}
          {it.warnings.map((w) => <span key={w} className="tag warn">{w}</span>)}
        </div>
      ))}
      {skipped.length > 0 && (
        <p className="small">Không xét: {skipped.map((t) => `“${t.term}”`).join(", ")}</p>
      )}
      {pageWarnings.length > 0 && (
        <div aria-label="Cảnh báo">
          {pageWarnings.map((w) => (
            <span key={w} className={`tag ${/not recogni|Không nhận ra|cannot be scored|Không chấm/i.test(w) ? "miss" : "warn"}`}>
              {w}
            </span>
          ))}
        </div>
      )}
      {data.trace.fallback && (
        <p className="small">
          <span className="tag miss">Đường văn bản</span> xếp theo độ giống của “{data.trace.fallback.text}”
          với ảnh (như B1), không qua KG.
        </p>
      )}
      {data.trace.fusion && (
        <p className="small muted">
          Điểm KG được trộn với cosine CLIP của “{data.trace.fusion.text}”, trọng số {data.trace.fusion.weight}.
        </p>
      )}
      {data.degraded.length > 0 && (
        <p className="small"><span className="tag warn">Chạy giảm cấp</span> {data.degraded.join("; ")}</p>
      )}
    </section>
  );
}

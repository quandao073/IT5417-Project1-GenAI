import type { ExplanationEvent, Intent } from "@/lib/api";
import { both, citations, pathLines, pct, WEIGHT_TIP } from "@/lib/present";

function Cited({ text, items }: { text: string; items: Map<string, string> }) {
  return (
    <>
      {citations(text).map((piece, i) =>
        piece.cite ? (
          <span key={i} className="cite" tabIndex={0} aria-label={items.get(piece.cite) ?? piece.cite}>
            {piece.text}
            <span className="tip" role="tooltip">{items.get(piece.cite) ?? "mục không có"}</span>
          </span>
        ) : (
          <span key={i}>{piece.text}</span>
        ))}
    </>
  );
}

/** The KG path of every intent with columns: edges in words ("gợi ý"), each weighed column's
 * normalised weight (61%) with what it means on hover; never the raw p. */
export function PathView({ intents }: { intents: Intent[] }) {
  const shown = intents.filter((i) => i.columns.length > 0 && i.path.length > 0);
  if (shown.length === 0) return <p className="muted small">Không có đường đi KG.</p>;
  return (
    <div>
      {shown.map((it, i) => (
        <div key={i} className="path">
          <div>{`“${it.term}” → ${both(it.name_vi, it.name)}`}</div>
          {pathLines(it).map((l, j) => (
            <div key={j}>
              {`  ${l.text}`}
              {l.weight !== undefined && (
                <span className="weight" tabIndex={0} title={WEIGHT_TIP[it.kind]}
                      aria-label={`trọng số ${pct(l.weight)}: ${WEIGHT_TIP[it.kind]}`}>
                  {` · trọng số ${pct(l.weight)}`}
                </span>
              )}
            </div>
          ))}
        </div>
      ))}
    </div>
  );
}

const STATUS: Record<ExplanationEvent["status"], string> = {
  ok: "đầy đủ",
  partial: "một phần (có câu bị bỏ vì không có căn cứ)",
  ungrounded: "bị bỏ: không câu nào có căn cứ",
  failed: "lỗi model ngôn ngữ",
  disabled: "đã tắt",
  no_results: "không có kết quả",
};

export function Explanation({ event, loading, intents, textRoute = false }: {
  event: ExplanationEvent | null; loading: boolean; intents: Intent[]; textRoute?: boolean;
}) {
  if (loading && !event) {
    return (
      <section className="panel" aria-busy="true">
        <h2>Diễn giải</h2>
        <p className="muted small">Đang viết diễn giải (kết quả đã hiện bên trái)…</p>
        <div className="skeleton" style={{ height: 120 }} />
      </section>
    );
  }
  if (!event) return null;
  if (textRoute) {
    return (
      <section className="panel explain" aria-labelledby="explain">
        <h2 id="explain">Diễn giải</h2>
        <p className="muted">Không có diễn giải: trang này xếp theo độ giống văn bản, không có đường đi KG
          nào để diễn giải.</p>
      </section>
    );
  }
  const items = new Map((event.knowledge_items ?? []).map((k) => [k.id, k.text]));
  const showPath = event.fallback === "path" || event.status === "partial";
  return (
    <section className="panel explain" aria-labelledby="explain">
      <h2 id="explain">Diễn giải</h2>
      <p className="small">
        <span className={`tag ${event.status === "ok" ? "ok" : "warn"}`}>{STATUS[event.status]}</span>
      </p>
      {event.summary && <p>{event.summary}</p>}
      {(event.images ?? []).length > 0 && (
        <>
          <h3>Theo từng ảnh</h3>
          <ol>
            {event.images!.map((im) => <li key={im.rank} value={im.rank}>{im.line}</li>)}
          </ol>
        </>
      )}
      {event.knowledge && (
        <>
          <h3>Kiến thức liên quan</h3>
          <p><Cited text={event.knowledge} items={items} /></p>
          <p className="muted small">Đưa chuột hoặc Tab tới [K#] để xem mục kiến thức được dẫn.</p>
        </>
      )}
      {showPath && (
        <>
          <h3>Đường đi trên KG (thô)</h3>
          <PathView intents={intents} />
        </>
      )}
    </section>
  );
}

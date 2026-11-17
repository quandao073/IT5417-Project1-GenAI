import type { Intent, KGResult } from "@/lib/api";
import { imageSrc } from "@/lib/api";
import { FUSION_TIP, fusionParts, pct, topParts } from "@/lib/present";

export function ResultCard({ r, intents, onOpen, badge, scale = "điểm" }: {
  r: KGResult; intents: Intent[]; onOpen?: (r: KGResult) => void; badge?: string;
  scale?: string;     // what the score is when it is not fused: "cosine" for B0 and B1
}) {
  const top = topParts(intents, r.intents ?? []);
  const parts = fusionParts(r);
  return (
    <button type="button" className={`card ${badge ? "shared" : ""}`} onClick={() => onOpen?.(r)}
            aria-label={`Ảnh hạng ${r.rank}, xem chi tiết`}>
      <img src={imageSrc(r.image_url)} alt={`Ảnh X-quang hạng ${r.rank}, view ${r.view}`}
           loading="lazy" width={256} height={256} />
      <div className="meta">
        <div className="row">
          <span className="rank">#{r.rank}</span>
          {r.view && <span className="tag info">{r.view}</span>}
          <span className="score" title={parts ? FUSION_TIP : `${scale}: thang xếp hạng, không phải xác suất`}>
            {parts ? "điểm trộn" : scale} {r.score.toFixed(2)}
          </span>
        </div>
        {parts && (
          <div className="fusion" title={FUSION_TIP}>
            <span>z(KG) {parts.kg}</span>
            <span>CLIP {parts.clip}</span>
          </div>
        )}
        {badge && <span className="tag ok">{badge}</span>}
        {top.length > 0 && (
          <ul className="parts" aria-label="Độ khớp của các cột đóng góp nhiều nhất">
            {top.map((p) => (
              <li key={p.label}>
                <span>{p.label}</span>
                <strong>{pct(p.value)}</strong>
                <span className="bar" aria-hidden><span style={{ width: pct(p.value) }} /></span>
              </li>
            ))}
          </ul>
        )}
      </div>
    </button>
  );
}

export function ResultGrid({ results, intents, onOpen }: {
  results: KGResult[]; intents: Intent[]; onOpen: (r: KGResult) => void;
}) {
  if (results.length === 0) {
    return <p className="panel muted">Không có ảnh nào: câu hỏi không có gì để chấm.</p>;
  }
  return (
    <div className="grid">
      {results.map((r) => <ResultCard key={r.image_id} r={r} intents={intents} onOpen={onOpen} />)}
    </div>
  );
}

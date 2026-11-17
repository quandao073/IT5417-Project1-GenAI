import type { ResultsEvent } from "@/lib/api";

/** The refusal of a request for no chest X-ray (D79): the message, and questions to try. */
export function OutOfScope({ data, onTry }: { data: ResultsEvent; onTry?: (q: string) => void }) {
  return (
    <section className="panel out-of-scope" role="status" aria-labelledby="oos">
      <h2 id="oos">{data.message ?? "Hệ thống chỉ tìm ảnh X-quang ngực."}</h2>
      <p className="muted">
        Câu hỏi này không hỏi về ảnh X-quang ngực, nên không trả ảnh nào và không viết diễn giải.
        {data.scope_reason ? ` (Lý do của parser: ${data.scope_reason}.)` : ""}
      </p>
      {(data.examples ?? []).length > 0 && (
        <>
          <p className="small">Thử một câu như:</p>
          <div className="examples">
            {data.examples!.map((q) =>
              onTry ? <button key={q} type="button" onClick={() => onTry(q)}>{q}</button>
                    : <span key={q} className="tag info">{q}</span>)}
          </div>
        </>
      )}
    </section>
  );
}

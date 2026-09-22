# Scope và non-goals

> Chốt phạm vi V1. Xem §3 của [kế hoạch](ke-hoach-du-an-nho-cxr-semantic-retrieval-v1.md).

Dự án từng được scope kèm một Knowledge Graph suy ra từ RadGraph-XL lưu trong Neo4j.
Phần đó đã được hoãn sang V2 (đồ án tốt nghiệp). V1 là một sản phẩm độc lập và
đánh giá được, đồng thời tạo ra data contracts, stable IDs, vector index, query
schema, tool interfaces, API và benchmark mà V2 sẽ dùng lại.

## V1 — bắt buộc hoàn thành

| Hạng mục | Ghi chú |
|---|---|
| Data audit và canonical corpus 25.000 study | Join rate đo được 1,0000 |
| Image embedding và Qdrant vector index | Model do Phase 2 gate chọn |
| Text-to-image semantic search | |
| LLM query parser có structured output | Ollama, kèm rule parser fallback |
| Mapping truy vấn tiếng Việt/Anh vào 14 pathology | `configs/concepts.yaml` |
| Include / exclude / explicit absence / uncertainty policies | `NOT_MENTIONED` không bao giờ là `ABSENT` |
| Report full-text search | SQLite FTS5 |
| Hybrid ranking | RRF + constraint reranker |
| Web UI, FastAPI và Docker Compose | Gồm màn so sánh ba mode |
| Evaluation định lượng và error analysis | Ba trục, xem [evaluation-design](evaluation-design.md) |

## V1 — không thực hiện

Neo4j · RadGraph entity/relation ingestion vào runtime · `LOCATED_AT` /
`MODIFIES` / `SUGGESTIVE_OF` reasoning · UMLS, SNOMED CT, RadLex ·
truy vấn theo vị trí giải phẫu chi tiết ("left lower lobe opacity") ·
graph visualization · fine-tune VLM · tự động chẩn đoán hoặc đề xuất điều trị ·
PACS/HIS/FHIR integration · cloud deployment.

## Hoãn sang V2 (đồ án tốt nghiệp)

| V1 | V2 |
|---|---|
| Query schema 14 nhãn | Observation + anatomy + relation schema |
| `labels.parquet` | Canonical KG facts |
| SQLite report search | Report embeddings + evidence graph |
| Qdrant image vectors | Multimodal hybrid index |
| Deterministic workflow | Constrained tool-calling agent |
| Label constraints | KG constraints và reasoning |
| Giải thích bằng label/report | Graph provenance paths |
| Vector vs hybrid | Vector vs hybrid vs KG ablation |

Câu hỏi nghiên cứu V2: *Knowledge Graph có cải thiện retrieval với các truy vấn
chứa vị trí giải phẫu, phủ định và quan hệ giữa các finding hay không?*

V1 phải thiết kế ID, data contracts và tool interfaces sao cho V2 thêm được
Knowledge Graph mà không phải viết lại image ingestion, vector retrieval, API
hoặc UI.

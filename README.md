# CXR Semantic Retrieval

Hệ thống truy vấn ảnh X-quang phổi theo ngữ nghĩa, kết hợp image embedding,
structured CheXpert labels và full-text search trên báo cáo.

> **Research/education POC — not for diagnosis.**
> Đây là công cụ tìm kiếm, không phải hệ thống chẩn đoán. Điểm số trả về không phải xác suất bệnh.

## Trạng thái

Đang xây. Trạng thái hiện tại, các quyết định đang có hiệu lực và những bẫy đã
biết nằm ở [WORKLOG.md](WORKLOG.md) — đọc file đó trước khi sửa code.

Xem thêm [kế hoạch triển khai](docs/ke-hoach-du-an-nho-cxr-semantic-retrieval-v1.md)
để biết lộ trình 8 tuần và thứ tự các phase, và [scope-and-non-goals](docs/scope-and-non-goals.md)
để biết phần nào đã được hoãn sang V2. Lưu ý kế hoạch viết trước khi có số đo nên
vài yêu cầu đã bị override; WORKLOG là nguồn đúng.

## Kiến trúc

```
web (Next.js) → api (FastAPI) → qdrant     (image vectors)
                              → SQLite FTS (reports)
                              → ollama     (query parser, có rule fallback)
```

Corpus build và embedding chạy trực tiếp ngoài Docker (`make build-corpus`, `make embed`);
Docker worker chỉ là đường thay thế khi muốn embed bằng GPU. API, web và Qdrant chạy trong
compose, thuần CPU vì embedding đã precompute.

## Dữ liệu

| Phần | Nguồn |
|---|---|
| Ảnh | CheXpert-v1.0-small (JPEG downsample) |
| Report và CheXbert labels | CheXpert Plus |
| Khóa join | `path_to_image` — đo được join rate 1,0000 trên toàn bộ 223.462 dòng |

Cả hai dataset đều yêu cầu đăng ký và chấp nhận điều khoản. Không có dữ liệu nào
được commit vào repo này.

## Yêu cầu

- Docker Desktop (Compose v2+), WSL2
- `uv` (Python) và `pnpm` (Node) nếu chạy ngoài Docker
- ~25 GB trống ngoài raw data (raw data chiếm 12,1 GB)
- GPU NVIDIA — không bắt buộc; embedding chạy được trên CPU, GPU chỉ để nhanh hơn
- Ollama 0.40+ (native) với model qwen2.5:3b-instruct-q4_K_M — chỉ cần cho LLM parser; không có thì API dùng rule parser.
  Cài: `winget install Ollama.Ollama` rồi `ollama pull qwen2.5:3b-instruct-q4_K_M`
- `make` (Git Bash hoặc WSL)

## Chạy nhanh

```bash
make setup               # tạo .env + thư mục data/artifacts + cài deps
make up                  # web + api + qdrant
make seed                # nạp fixture tổng hợp, không cần dataset
make smoke               # kiểm tra end-to-end
```

Mở http://localhost:3000.

## Pipeline đầy đủ (cần dataset)

```bash
make audit-data           # đo tỉ lệ join — CỔNG CHẶN
make build-corpus         # canonical Parquet tables + sample manifest
make embed                # embed 25k ảnh ra shard file, chạy trực tiếp trên CPU, resume được
make index                # upsert shard vào Qdrant + build_manifest.json (cần `docker compose up -d qdrant`)
make build-report-index   # SQLite FTS5
make validate-indexes     # kiểm tra build_id khớp nhau trước khi phục vụ
make eval-parser          # chấm parser trên benchmark held-out (cần Ollama)
make eval-retrieval       # chấm label / vector / hybrid trên benchmark held-out (cần Qdrant + report index; key OpenAI cho `auto`)
make up                   # web + api + qdrant
make evaluate             # benchmark ba trục
```

`make embed` không cần Docker hay GPU, mất ≈34 phút (đo từ mtime shard của lần chạy CPU dở,
chưa tính nạp model); đường GPU qua Docker worker nằm trong comment của `Makefile`. Số đo của vector baseline: [docs/vector-baseline.md](docs/vector-baseline.md).

`make eval-parser` (tức `cxr eval-parser`) **ghi đè** `artifacts/metrics/parser_eval.json`. Lần chạy
heldout ngày 2026-10-14 được đóng băng ở `artifacts/metrics/parser_eval_heldout_20261014.json`. Số dev
lấy từ `uv run cxr eval-parser --split dev --out artifacts/metrics/parser_eval_dev.json`.

Thứ tự dựng lại chỉ mục: `cxr embed` → `cxr index` → `cxr build-report-index` → `cxr validate-indexes`
(7 check; check thứ 7 so report index với `reports.parquet`: số dòng, tập `study_id`, `corpus_sha256`). Report index gắn với
corpus canonical chứ không với `build_id`, nên đổi encoder không phải dựng lại nó.

`make eval-retrieval` (tức `cxr eval-retrieval`; mặc định heldout, ba cấu hình parser `gold`/`rule`/`auto`
× ba mode, nhóm negation dưới hai policy) cũng **ghi đè** `artifacts/metrics/retrieval_eval.json`. Hai lần
chạy heldout ngày 2026-10-21 được đóng băng ở `artifacts/metrics/retrieval_eval_heldout_20261021.json` và
`artifacts/metrics/retrieval_eval_heldout_20261021_run2.json` (lần được trích dẫn). Lưới fusion chỉ chạy
trên dev: `uv run cxr eval-retrieval --split dev --parsers gold --modes hybrid --grid --out
artifacts/metrics/retrieval_dev.json`. `auto` cần key OpenAI trong `.env` (WORKLOG D22). Report index tắt,
hoặc có `auto` mà LLM không `ok`, thì lệnh thoát mã 1. Số đo: [docs/retrieval-evaluation.md](docs/retrieval-evaluation.md).

Chạy native (không qua Docker): `make setup` chép `.env.example` thành `.env` với
`QDRANT_URL=http://qdrant:6333`, `LLM_BASE_URL=http://host.docker.internal:11434` và đường dẫn
`/app/...`, chỉ đúng trong container. Các lệnh native (`cxr index`, `validate-indexes`,
`bench-vector`, `cxr parse`, `cxr eval-parser`, `cxr eval-retrieval`, uvicorn) cần hoặc không có `.env`,
hoặc ghi đè: `QDRANT_URL=http://localhost:6333`, `LLM_BASE_URL=http://localhost:11434`,
`IMAGE_ROOT=data/raw/CheXpert-v1.0-small`, `CANONICAL_DIR=data/canonical`, `INDEXES_DIR=data/indexes`,
`FTS_DB_PATH=data/indexes/report_search.sqlite` (thiếu dòng này thì API và `eval-retrieval` không thấy
report index).
Không ghi đè `LLM_BASE_URL` thì `cxr parse` và API chỉ chạy rule parser (chỉ có warning báo), còn
`cxr eval-parser` với `llm`/`auto` thoát mã 1.

`make help` liệt kê toàn bộ target.

## Truy vấn

`POST /api/v1/query/parse` trả `SearchQuery` đã validate kèm `parser` (llm | rule), `fallback_used`
và `warnings`. `POST /api/v1/search` nhận `mode` `vector` (mặc định), `label` hoặc `hybrid`,
`absence_policy` (`exclude_present` mặc định — `NOT_MENTIONED` không bị coi là có bệnh — hoặc
`explicit_absence_required`) và `uncertainty_policy` (`EXCLUDE` | `INCLUDE`). Kết quả mode `label`
không xếp hạng: trả theo thứ tự giả ngẫu nhiên cố định kèm `total_matches`. Mode `hybrid` lọc nhãn như
`label` (cùng `total_matches`) rồi xếp hạng trong tập đó bằng RRF của nhánh vector và nhánh report
(SQLite FTS5, bm25); `score_breakdown` ghi `vector_rank`, `report_rank`, `rrf_score`. Mọi mode trả
`report_evidence`: câu nguyên văn của report (offset vào `search_text`) cho mỗi concept bị ràng buộc, kèm
polarity (`affirmed`, `negated`, `uncertain`). Thiếu hoặc lệch report index thì API vẫn chạy: `/health`
báo `report_index: "disabled"`, hybrid chạy không có nhánh report và trả warning. Số đo parser:
[docs/parser-evaluation.md](docs/parser-evaluation.md), số đo retrieval:
[docs/retrieval-evaluation.md](docs/retrieval-evaluation.md). Thử nhanh: `uv run cxr parse "tim to nhưng không có tràn dịch"`.

## Cấu trúc

| Thư mục | Nội dung |
|---|---|
| `src/` | Package Python duy nhất: data, embeddings, query, retrieval, api, evaluation |
| `web/` | Next.js UI |
| `docker/` | Dockerfile và cấu hình service |
| `configs/` | data, models, retrieval, concepts, evaluation, logging |
| `docs/` | Thiết kế, data dictionary, ADR, limitations |
| `data/`, `artifacts/` | Gitignored |

## Giới hạn

Xem [docs/limitations.md](docs/limitations.md). Tóm tắt: ảnh đã downsample nên
không dùng cho chi tiết mảnh; nhãn sinh tự động từ report chứ không phải ground
truth lâm sàng; hai trong ba trục đánh giá đều suy ra từ report nên còn tương quan.

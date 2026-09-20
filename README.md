# CXR Semantic Retrieval

Hệ thống truy vấn ảnh X-quang phổi theo ngữ nghĩa, kết hợp image embedding, full-text search trên báo cáo và một Knowledge Graph suy ra từ RadGraph-XL.

> **Research/education POC — not for diagnosis.**
> Đây là công cụ tìm kiếm, không phải hệ thống chẩn đoán. Điểm số trả về không phải xác suất bệnh.

## Trạng thái

Khung dự án. Chưa có code thực thi — xem [kế hoạch triển khai](ke-hoach-POC-chexpert-plus-semantic-retrieval.md) để biết lộ trình 12 tuần và thứ tự các phase.

## Kiến trúc

```
web (Next.js) → api (FastAPI) → qdrant   (image vectors)
                              → SQLite FTS (reports)
                              → neo4j    (knowledge graph)
                              → ollama   (query parser, có rule fallback)
```

Ingestion chạy offline trên GPU qua profile `worker`; runtime thuần CPU vì embedding đã precompute.

## Dữ liệu

| Phần | Nguồn |
|---|---|
| Ảnh | CheXpert-v1.0-small (JPEG downsample) |
| Report, CheXbert labels, RadGraph-XL | CheXpert Plus |
| Khóa join | `path_to_image` |

Cả hai dataset đều yêu cầu đăng ký và chấp nhận điều khoản. Không có dữ liệu nào được commit vào repo này.

## Yêu cầu

- Docker Desktop (Compose v2+), WSL2
- ~120 GB trống cho corpus đầy đủ (~45 GB cho corpus 25k)
- GPU NVIDIA ≥ 8 GB VRAM cho bước embedding (không bắt buộc để chạy demo)
- `make` (Git Bash hoặc WSL)

## Chạy nhanh

```bash
make setup               # tạo .env + thư mục data/artifacts; sửa NEO4J_PASSWORD trong .env
make up                  # web + api + qdrant + neo4j
make seed                # nạp fixture tổng hợp, không cần dataset
make smoke               # kiểm tra end-to-end
```

Mở http://localhost:3000.

## Pipeline đầy đủ (cần dataset)

```bash
make audit       # Phase 1: đo tỉ lệ join, storage report — CỔNG CHẶN
make ingest      # master table + sample manifest
make preprocess  # chuẩn hóa ảnh 512 px
make embed       # embedding trên GPU
make index       # Qdrant + SQLite FTS
make graph       # RadGraph → study_facts.parquet → Neo4j
make evaluate    # benchmark ba trục
```

`make help` liệt kê toàn bộ target.

## Cấu trúc

| Thư mục | Nội dung |
|---|---|
| `src/cxr_retrieval/` | Package Python dùng chung cho API và worker |
| `web/` | Next.js UI |
| `docker/` | Dockerfile và cấu hình service |
| `configs/` | data, models, retrieval, ontology, evaluation, logging |
| `docs/` | Thiết kế, data dictionary, ADR, limitations |
| `data/`, `artifacts/` | Gitignored |

## Giới hạn

Xem [docs/limitations.md](docs/limitations.md). Tóm tắt: ảnh đã downsample nên không dùng cho chi tiết mảnh; nhãn sinh tự động từ report chứ không phải ground truth lâm sàng; hai trong ba trục đánh giá đều suy ra từ report nên còn tương quan.

# CXR Semantic Retrieval

Hệ thống truy vấn ảnh X-quang phổi theo ngữ nghĩa, kết hợp image embedding,
structured CheXpert labels và full-text search trên báo cáo.

> **Research/education POC — not for diagnosis.**
> Đây là công cụ tìm kiếm, không phải hệ thống chẩn đoán. Điểm số trả về không phải xác suất bệnh.

## Trạng thái

Khung dự án + data audit. Xem [kế hoạch triển khai](docs/ke-hoach-du-an-nho-cxr-semantic-retrieval-v1.md)
để biết lộ trình 8 tuần và thứ tự các phase, và [scope-and-non-goals](docs/scope-and-non-goals.md)
để biết phần nào đã được hoãn sang V2.

## Kiến trúc

```
web (Next.js) → api (FastAPI) → qdrant     (image vectors)
                              → SQLite FTS (reports)
                              → ollama     (query parser, có rule fallback)
```

Corpus build và embedding chạy offline qua profile `worker`; runtime thuần CPU
vì embedding đã precompute.

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
make embed                # embedding + index vào Qdrant
make build-report-index   # SQLite FTS5
make validate-indexes     # kiểm tra build_id khớp nhau trước khi phục vụ
make evaluate             # benchmark ba trục
```

`make help` liệt kê toàn bộ target.

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

# Vector baseline — Phase 3

**Ngày:** 2026-10-11 · **Build:** `20261011_chexpert25k_biomedclip_v1` · **Git:** `9f8aaa34d17ec771872513e09f7f1260307d200a` · **Model:** BiomedCLIP @ `9f341de` (dim 512)

Nguồn của mọi số dưới đây: `data/indexes/embeddings/<build_id>/shards.json` (embed),
`data/indexes/build_manifest.json` (index) và `artifacts/metrics/vector_baseline.json` (latency
từ `uv run cxr bench-vector --api-url http://localhost:8000`, chất lượng từ
`uv run cxr score-corpus --build-id 20261011_chexpert25k_biomedclip_v1`). Cả ba gitignored.
Embed, index, bench và score đều chạy ở commit trên. Cột spike lấy từ
`artifacts/metrics/model_spike.json` (Phase 2, mẫu 1.000 study).

**Thiết bị.** Embed chạy trên **GPU** (RTX 2000 Ada, `--device cuda --batch-size 64`, torch
`2.4.1+cu124` cài tạm vào venv bằng `uv pip`), rút thời gian từ ≈34 phút CPU xuống 293 s;
`shards.json` ghi `device: cuda`. Sau đó venv được đưa về lockfile bằng `uv sync --all-extras`, và mọi thứ còn lại
(validate, API, bench, score, integration test) chạy trên **CPU**: `torch.__version__` là
`2.4.1+cpu` (kiểm trực tiếp; `library_versions.torch` của `vector_baseline.json` ghi tên bản phân
phối `2.4.1`), 14 luồng torch (`latency.torch_threads` của `vector_baseline.json`). Riêng
`cxr index` chạy **trước** lần sync đó, nên `library_versions` của `build_manifest.json` ghi torch
và torchvision `+cu124`, numpy 2.5.2; index không dùng torch (xem WORKLOG T14).

## Embed và index

| | |
|---|---|
| Ảnh đã embed | 25.000 / 0 lỗi (`embedded` 25.000, `skipped` 0) |
| s/ảnh (GPU, batch 64) | 0,0117 · tổng 293 s (4,9 phút, `started_at` → `finished_at`, chưa tính nạp model) |
| s/ảnh CPU (lần chạy CPU bỏ dở, lệnh mặc định, batch 32) | 0,081 · 25k ≈34 phút (đo từ mtime shard của lần chạy CPU dở, chưa tính nạp model) |
| Shard | 25 × 1.000, 57,6 MB (57.612.950 B) |
| Qdrant | collection `cxr_images_20261011_chexpert25k_biomedclip_v1`, 25.000 vector cosine, server `v1.19.2`; volume 95 MiB thực chiếm (58 MiB segment + 37 MiB WAL) |
| `cxr index` | 13 s, kể cả khởi động `uv run` |
| `validate-indexes` | 6/6 PASS |

- s/ảnh tính trên `encode_images` theo đường dẫn: gồm đọc file, tiền xử lý và forward; không gồm
  ghi shard. 0,0117 × 25.000 = 293 s, khớp `started_at` → `finished_at`.
- s/ảnh CPU đo từ mtime của 17 shard mà lần chạy CPU kịp ghi trước khi dừng (16 khoảng liên
  tiếp, trung bình 81,0 s/shard; 79,0 s ở 10 khoảng đầu, 84,4 s ở 6 khoảng sau). Spike Phase 2
  đo 0,093 s/ảnh ở batch 16.
- **Vector GPU và CPU gần trùng nhau nhưng không trùng bit.** Trên 17.000 ảnh chung (shard
  0000–0016, cùng `image_ids`): cosine giữa vector GPU và CPU của cùng một ảnh nhỏ nhất 0,999995,
  sai lệch tuyệt đối lớn nhất trên từng phần tử 4,6e-4. Ảnh hưởng lên thứ hạng chưa đo. Một build
  embed trọn trên một thiết bị; lần chạy CPU dở được để riêng ở
  `data/indexes/embeddings/_cpu_partial_<build_id>/`, không trộn vào build.
- Volume Qdrant: `docker system df -v` báo 714,4 MB, đó là kích thước biểu kiến (682 MiB). WAL
  cấp trước theo khối 32 MiB và file mmap thưa nên thực chiếm (`du` trong container) chỉ 95 MiB.
- `indexed_vectors_count` = 0 (`GET /collections/<name>`): 8 segment × 3.125 point, mỗi segment
  dưới `indexing_threshold` mặc định (10.000 KB) nên Qdrant không dựng HNSW. Ở cỡ này mọi query
  là quét toàn bộ, **chính xác**, không mất recall do ANN.
- `payload_schema.projection`: keyword, 25.000 point (kiểm trên server thật; client `:memory:`
  bỏ qua payload index).

## Latency (warm, 30 query × 4 lần đo)

30 query = 14 prompt đơn concept + 16 tổ hợp hai concept. Mỗi query chạy 5 lần, bỏ lần đầu, nên
mỗi khâu có n = 120. `encode`/`qdrant`/`total` gọi hàm trực tiếp trong một tiến trình dựng
state y như API; `http` là `POST /api/v1/search` tới uvicorn chạy riêng trên cùng máy. `top_k` 10.

| Khâu | p50 (ms) | p95 (ms) |
|---|---:|---:|
| encode (text tower, CPU) | 129 | 138 |
| qdrant (`query_points`) | 20,5 | 29,1 |
| total (encode + qdrant) | 154 | 158 |
| http (end-to-end qua API) | 150 | 158 |

Cold start: 5.289 ms (dựng state: nạp model, kết nối Qdrant, nạp `images.parquet`, cộng query
đầu). Tiêu chí §17 warm p95 < 1 s: **PASS** (158 ms).

Encode text chiếm phần lớn thời gian (p50 129/154 ms). Overhead HTTP không đo được: p95 qua API
bằng p95 gọi trực tiếp (158 ms), p50 qua API còn thấp hơn (150 so với 154 ms), tức chênh lệch
nằm trong nhiễu giữa hai tiến trình.

## Chất lượng zero-shot trên 25.000 study

Prompt như Phase 2: `"chest x-ray showing {}"` + `concepts.yaml:<c>.en[0]`. Nhãn resolve theo G1.
Dương = `PRESENT`, âm strict = `ABSENT`; đủ mẫu khi ≥ 10 mỗi phía. 25.000 ảnh, 25.000 study
(một ảnh mỗi study). p@10 = tỉ lệ `PRESENT` trong 10 ảnh điểm cao nhất. Prevalence = n_pos / 25.000.

| Concept | AUROC strict 25k | AUROC strict spike 1k | n_pos | n_neg_strict | p@10 | Prevalence |
|---|---:|---:|---:|---:|---:|---:|
| No Finding | 0,585 | thiếu mẫu (0 âm) | 3.814 | 174 | 0,5 | 15,3% |
| Enlarged Cardiomediastinum | 0,602 | 0,614 | 2.462 | 5.938 | 0,2 | 9,8% |
| Cardiomegaly | 0,681 | 0,669 | 6.618 | 4.635 | 0,7 | 26,5% |
| Lung Opacity | 0,646 | 0,660 | 18.394 | 1.171 | 0,9 | 73,6% |
| Lung Lesion | 0,469 | thiếu mẫu (8 âm) | 2.442 | 471 | 0,3 | 9,8% |
| Edema | 0,681 | 0,690 | 10.021 | 3.937 | 0,6 | 40,1% |
| Consolidation | 0,655 | 0,678 | 2.899 | 5.610 | 0,2 | 11,6% |
| Pneumonia | 0,564 | 0,587 | 1.207 | 897 | 0,1 | 4,8% |
| Atelectasis | 0,559 | 0,478 | 7.165 | 335 | 0,5 | 28,7% |
| Pneumothorax | 0,564 | 0,603 | 3.612 | 11.768 | 0,2 | 14,4% |
| Pleural Effusion | 0,763 | 0,769 | 14.252 | 6.073 | 0,9 | 57,0% |
| Pleural Other | 0,496 | thiếu mẫu (1 âm) | 1.065 | 230 | 0,0 | 4,3% |
| Fracture | 0,520 | 0,642 | 2.302 | 1.255 | 0,3 | 9,2% |
| Support Devices | 0,604 | 0,607 | 20.717 | 829 | 1,0 | 82,9% |

**Macro strict (cả 14 concept đủ mẫu): 0,599** [CI 95% 0,592–0,607; bootstrap 1.000 lần, seed
20260101]. Macro lenient cùng tập: 0,598.

Trên **cùng 11 concept** mà spike chấm, macro strict 25k là **0,622**, so với 0,636 của spike
(CI 0,607–0,663). Số 0,622 là trung bình `auroc_strict` của 11 concept đó trong
`vector_baseline.json`, không có CI riêng.

Nhãn khớp bảng prevalence P2 trong WORKLOG tới 0,1 điểm ở cả 14 concept, cả `PRESENT` lẫn
`ABSENT` (vd Pleural Effusion 57,0% / 24,3%).

## Cách đọc

- Nhãn sinh từ báo cáo (CheXbert/CheXpert labeler), không phải bác sĩ. 200 ca `valid/` nằm trong
  25k nhưng không được dùng để tune (D12): số ở đây chỉ để báo cáo, không chọn gì.
- Âm tính strict = `ABSENT`; `NOT_MENTIONED` không phải âm tính (T7). Lenient (âm = mọi thứ
  không `PRESENT`) chỉ để tham khảo.
- `score` của API là cosine thô, không phải xác suất. Trên corpus này, p5–p95 cosine theo concept
  nằm trong 0,264–0,460 (trường `cosine` của JSON), tức trong dải ~0,2–0,5 mà schema ghi.
- Chỉ một model: đây là baseline cho hybrid, không phải so sánh model. P4 vẫn mở.
- Vector mode không lọc nhãn (G3); các số trên là phép xếp hạng mà mode `vector` thực sự chạy.
- So với spike 1.000 ảnh:
  - Macro 14 concept (0,599) thấp hơn spike (0,636) chủ yếu vì **khác tập concept**. Ba concept
    giờ mới đủ mẫu đều yếu: Lung Lesion 0,469 (dưới ngẫu nhiên), Pleural Other 0,496, No Finding
    0,585. Trên cùng 11 concept, 25k cho 0,622, nằm trong CI của spike.
  - **Fracture** đổi hạng nhiều nhất: 0,642 (hạng 6/11 ở spike, 44 âm strict) → 0,520 (hạng
    11/11, 1.255 âm). Số spike là nhiễu của mẫu nhỏ.
  - **Atelectasis** ngược lại: 0,478 (hạng cuối, 12 âm) → 0,559 (335 âm), vượt Fracture.
  - Pneumothorax giảm 0,603 → 0,564. Spike đã có 462 âm nhưng chỉ 135 dương, nên đây vẫn có thể
    là nhiễu mẫu; chưa kiểm.
  - Pleural Effusion vẫn đứng đầu và ổn định (0,769 → 0,763). Năm concept đầu (Pleural Effusion,
    Edema, Cardiomegaly, Consolidation, Lung Opacity) giữ nguyên nhóm.
- Vector gần như mù với Lung Lesion, Pleural Other và Fracture. Ở những concept này nhánh nhãn và
  report của hybrid (Phase 5) phải gánh phần chính.

## Tái lập

```bash
uv run cxr embed --build-id 20261011_chexpert25k_biomedclip_v1   # CPU mặc định, ≈34 phút; build này chạy GPU, xem dưới
uv run cxr index --build-id 20261011_chexpert25k_biomedclip_v1
uv run cxr validate-indexes                        # 6 x PASS
uv run uvicorn src.api.main:app --port 8000        # nền
uv run cxr bench-vector --api-url http://localhost:8000
uv run cxr score-corpus --build-id 20261011_chexpert25k_biomedclip_v1
uv run pytest -m integration                       # cần Qdrant
```

Đường GPU đã dùng cho build này: `uv pip install --reinstall --index-url
https://download.pytorch.org/whl/cu124 torch==2.4.1+cu124 torchvision==0.19.1+cu124`, rồi
`uv run --no-sync cxr embed --build-id 20261011_chexpert25k_biomedclip_v1 --device cuda
--batch-size 64`. Sau đó **phải** chạy `uv sync --all-extras` để về torch CPU của lockfile, **trước**
`cxr index`: `uv run` thường không làm việc này (xem WORKLOG T14).

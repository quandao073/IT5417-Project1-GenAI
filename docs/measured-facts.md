# Sự thật đã đo

> Số tự đo trên dữ liệu thật, kèm cách đo. Không chép từ paper hay tài liệu. Chuyển nguyên văn từ `WORKLOG.md`
> ngày 2026-10-25 để WORKLOG gọn lại; từ nay số đo mới ghi vào đây, append-only.
>
> File này để tra (viết báo cáo, kiểm một con số), không phải đọc mỗi phiên. Vài khối đo những thứ đã gỡ khỏi
> code (hybrid, mode `image`, qwen 3B); chúng giữ làm lịch sử, xem `WORKLOG.md` để biết cái gì còn chạy.
> Mã D/T/P trong file này là mã của WORKLOG; mã đã bỏ khỏi WORKLOG thì tra bằng `git log -p WORKLOG.md`.

## Artifact heldout đã đóng băng

`cxr eval-parser` và `cxr eval-retrieval` ghi đè file mặc định, nên các bản heldout được trích dẫn nằm riêng trong
`artifacts/metrics/`: `parser_eval_heldout_20261014.json` (qwen 3B, sha256 `999fbc70…`),
`parser_eval_heldout_gpt-5-mini.json` (parser mặc định), `retrieval_eval_heldout_20261021_run2.json` (sha256
`e6186239…`, lần được trích dẫn; lần 1 là `retrieval_eval_heldout_20261021.json`, `b35a6b76…`),
`retrieval_eval_heldout_image_probe_20261023.json` (sha256 `af019789…`).

## Dữ liệu nguồn (đo 2026-09-21 → 2026-09-22)

| Hạng mục | Giá trị | Đo bằng |
|---|---:|---|
| CheXpert-v1.0-small | 10,68 GB · 223.649 JPG | duyệt thư mục |
| — train / valid | 223.415 / 234 ảnh | duyệt thư mục |
| CheXpert Plus CSV | 386 MB · 223.462 record | `csv.DictReader` |
| Join rate `path_to_image` | **1,0000** (strategy `exact`) | `cxr audit` → `artifacts/audit/data_audit.json` |
| Dòng frontal AP/PA | 191.054 | quét CSV |
| → study duy nhất | **187.674** / 64.710 patient | quét CSV |
| `section_impression` có nội dung | 190.913 / 191.054 = **99,93%** | quét CSV |
| `section_findings` có nội dung | 59.441 / 223.462 = **26,6%** | quét CSV |
| `report` có nội dung | 223.462 = 100% | quét CSV |
| Hướng chụp | 191.071 frontal · 161.622 AP · 29.432 PA | quét CSV |
| Ảnh có nhãn bác sĩ | 234 ảnh nhưng chỉ **200 study** / 200 patient | duyệt thư mục |
| RadGraph-XL trên đĩa | 804 MB (không dùng ở V1) | `ls` |
| Disk trống C: | 191 GB | `Get-PSDrive` |

## Bất biến cấu trúc

- **`study_id` duy nhất toàn cục.** `train/` là `patient00001..patient64540`,
  `valid/` là `patient64541..patient64740`, **giao nhau 0 patient**. Nhờ vậy
  `study_id` = `patientNNNNN/studyN` dùng được làm primary key mà không cần kèm
  split. `contracts.py` dựa vào điều này.
- **Plus CSV có newline nhúng.** 10.746.681 dòng vật lý cho 223.462 record —
  trung bình **48 dòng/record**. Xem T2.
- **FTS5 có sẵn** trong `sqlite3` của Python (SQLite 3.50.4). Phase 5 không cần
  build SQLite riêng.

## Chi phí của Phase 1 (đo 2026-09-22)

| Thao tác | Đo được | Ghi chú |
|---|---|---|
| `polars.read_csv` Plus CSV, 6 cột | **223.462 dòng — khớp đúng** ground truth của `csv.DictReader` | Polars xử lý đúng newline nhúng. Dùng polars, không tự parse |
| Đọc kích thước ảnh (PIL) | 7,24 ms/ảnh → **~181 s cho 25k** | Đây là **cold disk I/O**, không phải chi phí decode |
| sha256 toàn file | 0,08 ms/ảnh → ~1,9 s cho 25k | Rẻ vì file đã nằm trong page cache sau lượt đọc trên |
| Tổng byte đọc cho 25k ảnh | ~1,30 GB (trung bình 50,6 KB/ảnh) | |

**Hệ quả thiết kế:** chi phí thật là I/O của lượt duyệt đầu tiên, không phải phép
tính. Phase 1 phải lấy **kích thước và checksum trong cùng một lượt mở file** để
chỉ trả phí I/O một lần — khoảng 3 phút cho 25k ảnh.

Đo thêm trên 500 ảnh ngẫu nhiên: cạnh dài nằm trong **320–483 px**, 500/500
checksum phân biệt. Khẳng định lại `data.yaml`: không ảnh nào vượt 512 px, nên
không có bước resize nào cả.

## Kết quả Phase 1 (corpus 25.000, seed 20260101, đo 2026-09-23)

| | |
|---|---:|
| Plus rows → eligible rows (frontal AP/PA + có impression + file tồn tại) | 223.462 → 190.913 |
| → study sau khi lấy 1 ảnh/study | **187.537** |
| → chọn | **25.000** study / 18.007 patient |
| Tỉ lệ AP/PA trong corpus | 84,5% / 15,5% (pool 84,6%) — phân tầng giữ đúng |
| `valid/` vào corpus | 200 study, ghim đầu manifest |
| Có findings section | 6.419 / 25.000 = 25,7% |
| `labels.parquet` | 1.139.866 dòng |
| Tổng dung lượng | 11 MB |

187.537 ở đây nhỏ hơn con số 187.674 đo trước đó vì phép đo cũ chỉ lọc frontal
AP/PA, chưa yêu cầu impression non-empty.

**Pool ngoài corpus** (đo 2026-10-22 cho câu hỏi finetune VLM, chưa có quyết định): bỏ mọi
`patient_id` của corpus (18.007) và cả `valid/`, còn **90.654 ảnh frontal / 88.853 study / 46.682
bệnh nhân** có impression, 25.603 study có findings. Corpus chiếm phần lớn số dòng vì bệnh nhân của
nó có nhiều study. Cách đo: `uv run python artifacts/metrics/train_pool_probe.py` (`join.read_plus_rows`,
lọc AP/PA + impression, chỉ `train/`; không kiểm file trên đĩa, join rate 1,0000).

## P2 đã giải quyết: prevalence sau resolve (% của 25.000)

| Concept | PRESENT | ABSENT | UNCERTAIN | NOT_MENTIONED |
|---|---:|---:|---:|---:|
| Pneumothorax | 14,4% | **47,1%** | 2,2% | 36,3% |
| Pleural Effusion | 57,0% | **24,3%** | 6,4% | 12,3% |
| Enlarged Cardiomediastinum | 9,8% | **23,8%** | 14,5% | 51,9% |
| Consolidation | 11,6% | **22,4%** | 22,0% | 43,9% |
| Cardiomegaly | 26,5% | **18,5%** | 6,5% | 48,5% |
| Edema | 40,1% | **15,7%** | 10,0% | 34,2% |
| Fracture | 9,2% | **5,0%** | 0,5% | 85,3% |
| Lung Opacity | 73,6% | **4,7%** | 2,3% | 19,4% |
| Pneumonia | 4,8% | **3,6%** | 17,7% | 73,8% |
| Support Devices | 82,9% | **3,3%** | 0,5% | 13,3% |
| Lung Lesion | 9,8% | **1,9%** | 1,1% | 87,2% |
| Atelectasis | 28,7% | **1,3%** | 26,6% | 43,4% |
| Pleural Other | 4,3% | **0,9%** | 2,3% | 92,5% |
| No Finding | 15,3% | **0,7%** | 0,0% | 84,0% |

Phân bố lưỡng cực: bác sĩ chủ động loại trừ vài thứ (tràn khí, tràn dịch, trung
thất) nên `ABSENT` phổ biến; còn xẹp phổi hay tổn thương phổi thì gần như không
ai viết "không có". Median `ABSENT` 5,0%, bốn concept dưới 2%.

## Phase 2 spike (đo 2026-10-03)

**Cách đo:** `uv run cxr spike-models --models <tên>` cho từng model, mỗi model một tiến trình
riêng chạy lần lượt, rồi `uv run cxr spike-models --rescore`. Mẫu manifest rank 200–1199
(1.000 study, không có ca `valid/`), torch CPU, 14 luồng, batch 16. s/ảnh là trung bình warm
(bỏ batch đầu); peak RSS là peak working set của tiến trình. Prompt
`"chest x-ray showing {}"`, AUROC strict trên 11 concept đủ mẫu (≥ 10 dương và ≥ 10 `ABSENT`).
Số lấy từ `artifacts/metrics/model_spike.json` (embed @ `0cb15d4`, `--rescore` @ `07314cd`);
thư viện theo `library_versions` của JSON: torch 2.4.1, torchvision 0.19.1, open_clip_torch
2.24.0, transformers 4.44.2, numpy 2.5.3.

| Model | Macro strict (CI 95%) | Macro lenient (14 concept) | s/ảnh | Dự báo 25k | ms/query | Peak RSS | Weights | Dim |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **BiomedCLIP** | **0,636** (0,607–0,663) | 0,609 | 0,093 | 0,644 h | 158 | 1.960 MiB | 784,6 MB | 512 |
| CheXzero | 0,461 (0,432–0,489) | 0,540 | 0,031 | 0,215 h | 42 | 1.494 MiB | 353,5 MB | 512 |
| XrayCLIP | 0,616 (0,587–0,645) | 0,626 | 0,637 | 4,421 h | 19 | 1.568 MiB | 603,0 MB | 512 |

Hiệu BiomedCLIP − XrayCLIP: 0,0198; CI 95% −0,015–0,053 (bootstrap 1.000 lần, seed 20260101
theo JSON, paired; chỉ báo cáo).
0 ảnh lỗi ở cả ba. Concept thiếu mẫu strict: No Finding (0 `ABSENT`), Lung Lesion (8),
Pleural Other (1). Nhãn của mẫu khớp bảng prevalence ở trên (vd Pleural Effusion 53,8% so với
57,0%, Support Devices 80,7% so với 82,9%). Chi tiết từng concept: `docs/model-decision.md`.

## Phase 3 (đo 2026-10-11)

**Cách đo:** build `20261011_chexpert25k_biomedclip_v1`, mọi lệnh ở commit `9f8aaa3`. Embed:
`uv run --no-sync cxr embed --device cuda --batch-size 64` (GPU RTX 2000 Ada, torch `2.4.1+cu124`
cài tạm); số lấy từ `shards.json`. `uv run cxr index` chạy khi venv **vẫn còn** torch `+cu124`
(index không dùng torch, nhưng `build_manifest.json` ghi torch/torchvision `+cu124`, numpy 2.5.2;
xem T14). Sau `uv sync --all-extras`, phần còn lại chạy trên torch `2.4.1+cpu`, 14 luồng:
`uv run cxr validate-indexes`, `uv run cxr bench-vector --api-url http://localhost:8000`
(uvicorn riêng trên cùng máy), `uv run cxr score-corpus --build-id <build_id>`, integration test;
số lấy từ `artifacts/metrics/vector_baseline.json`. Chi tiết: `docs/vector-baseline.md`.

| Hạng mục | Giá trị | Đo bằng |
|---|---:|---|
| Embed 25.000 ảnh, GPU | **0,0117 s/ảnh** · 293 s · 0 lỗi | `shards.json`: `s_per_image`, `started_at` → `finished_at` (không tính nạp model) |
| Embed CPU (lần chạy bỏ dở, batch 32) | 0,081 s/ảnh → 25k ≈34 phút | mtime 17 shard trong `_cpu_partial_<build_id>/`, 16 khoảng liên tiếp; chưa tính nạp model |
| Vector GPU vs CPU, 17.000 ảnh chung | cosine min 0,999995 · \|Δ\| max 4,6e-4 | so shard 0000–0016 của hai thư mục |
| Shard trên đĩa | 25 × 1.000 · 57,6 MB | tổng kích thước `.npz` |
| `cxr index` | 13 s | `time`, kể cả khởi động `uv run` |
| Volume Qdrant | **95 MiB** thực chiếm (58 segment + 37 WAL) · 682 MiB biểu kiến | `du` trong container; `docker system df -v` báo 714,4 MB là kích thước biểu kiến |
| HNSW | không dựng (`indexed_vectors_count` 0) → quét toàn bộ, chính xác | `GET /collections/<name>`: 8 segment × 3.125 point, dưới `indexing_threshold` 10.000 KB |
| Cold start | 5.289 ms | dựng state như API + query đầu |
| Warm p50 / p95 encode | 129 / 138 ms | n = 120 (30 query × 4, bỏ lần đầu) |
| Warm p50 / p95 qdrant | 20,5 / 29,1 ms | như trên |
| Warm p50 / p95 total | 154 / **158 ms** → §17 PASS | như trên |
| Warm p50 / p95 HTTP | 150 / 158 ms → overhead HTTP không đo được | `POST /api/v1/search` qua uvicorn |
| Macro strict AUROC, 14 concept | **0,599** (CI 95% 0,592–0,607) | `score-corpus`, bootstrap 1.000, seed 20260101; cả 14 concept đủ mẫu |
| Macro strict, 11 concept của spike | 0,622 (spike 0,636, CI 0,607–0,663) | trung bình `auroc_strict` của 11 concept trong JSON |

Nhãn 25k khớp bảng P2 ở trên tới 0,1 điểm cho cả 14 concept, cả `PRESENT` lẫn `ABSENT`.
`vector_baseline.json` không có mục `embed{}`/`index{}` như spec §6.3 viết (Task 8 không ghi); số
embed/index ở trên lấy từ `shards.json`, `build_manifest.json`, `du` và mtime shard.

## Phase 4 (đo 2026-10-14)

**Cách đo:** code ở `cd890d8`, cộng prompt và lexicon của commit Task 10 (prompt render sha256
`c975e4ca…`, `concepts.yaml` sha256 `735997a2…`; `parser_eval.json` ghi `git_commit` `cd890d8`).
Ollama 0.40.3 native, GPU RTX 2000 Ada Laptop 8 GB, model `qwen2.5:3b-instruct-q4_K_M`, `.env` rỗng.
Lệnh: `uv run cxr eval-parser --split heldout --parsers rule llm auto --label-filter` (chạy một lần)
và `uv run cxr eval-parser --split dev --parsers rule llm auto --out artifacts/metrics/parser_eval_dev.json`.
API: `uv run uvicorn src.api.main:app` native, Qdrant Docker, gọi qua `127.0.0.1`. Latency parser
bỏ câu đầu (n = 159). Chi tiết: `docs/parser-evaluation.md`.

| Hạng mục | Giá trị | Đo bằng |
|---|---:|---|
| Ollama 0.40.3 + qwen 3B q4 | 1,9 GB trên đĩa; 2,16 GB VRAM, trọn trên GPU, context 4.096; cold start xem T16 | `/api/tags`, `/api/ps` |
| Prompt cuối | 2.328 token, 20 ví dụ | `prompt_eval_count`; `prompt_examples()` |
| Heldout `auto` (= `llm`) concept / status / view macro-F1 | **0,880 / 0,718 / 0,890 → FAIL** (ngưỡng 0,90) | `parser_eval.json:eval.verdict` |
| Heldout `rule` concept / status / view | 0,793 / **0,862 / 0,945** | như trên |
| Heldout exact match `auto` / `rule` | 0,650 / 0,606 | như trên |
| Heldout `auto` schema-valid · fallback · retry | 1,000 · 0 · 0,0375 (6/160) | như trên |
| Heldout `auto` theo ngôn ngữ, status F1 | EN 0,826 · VI 0,593 | `by_language` |
| Latency p50 / p95 | rule 0,1 / 0,2 ms · llm 756 / 1.366 ms · auto 778 / 1.473 ms | `latency_ms` |
| Dev cuối `auto` / `rule` | 0,912 / 0,827 / 0,833 · 1,000 / 1,000 / 1,000 | `parser_eval_dev.json` |
| Rule exact match cả 200 câu, sau chỉnh | EN 68/100 · VI 69/100 | heldout `by_language` × n + dev 20/20 |
| Label filter, nhóm `negation` (40 câu heldout), median study khớp | `exclude_present/EXCLUDE` **10.223** · `explicit_absence_required/EXCLUDE` **1.013** | `label_filter.summary` |
| — theo từng câu: `explicit_absence_required` / `exclude_present` | median 0,199 (min 0,01, max 0,664); 0 câu khớp 0 ở mọi cặp policy | `label_filter.per_query` |
| API `/health` | `ready`, `llm: "ok"`, `studies` 25.000 | curl |
| API mode `label`, "Find edema with no pleural effusion" | `total_matches` 2.760 (`exclude_present`) · 1.532 (`explicit_absence_required`); `latency_ms` 2 · 1 | `POST /api/v1/search`, `top_k` 3 |
| API mode `vector`, cùng câu, top 10 | **6/10** mang `Pleural Effusion: PRESENT`, 1/10 `UNCERTAIN`; 3/10 thỏa ràng buộc | `labels` của từng kết quả |
| API cold start | `/health` 200 sau 18,3 s từ lúc gọi `uv run uvicorn`; `/query/parse` đầu 3,55 s (`llm_ms` 3.515), lần hai 0,86 s | script poll 0,1 s |

Gọi API bằng `httpx` tới `localhost` thay vì `127.0.0.1` làm mỗi kết nối mới chậm thêm ≈2 s trên
máy này. Hai lần đo cold start liền nhau, cùng script: `/query/parse` thứ hai mất 3,11 s (`llm_ms` 994)
ở `localhost` và 0,86 s (`llm_ms` 823) ở `127.0.0.1`. Số đo latency phải đi qua `127.0.0.1` hoặc dùng
lại connection.

**Quy ước cụm từ của labeler** (D21). Đếm trên `reports.parquet:search_text` (impression + findings)
của 25.000 study. Nhãn resolve theo G1, trong ngoặc là `CHEXPERT_V1` riêng:

| Cụm từ trong report | n study | Nhãn |
|---|---:|---|
| "airspace disease", không có "consolidat" | 162 | Lung Opacity PRESENT 134 (121); Consolidation PRESENT 5 (1) |
| "collapse", không có "atelect"/"pneumothora" | 71 | Atelectasis PRESENT 55 (50). "collapsed lung" chỉ xuất hiện 6 lần |
| "sternotomy wire", không chứa chuỗi con nào trong `tube line catheter pacemaker pacer picc port device lead stent valve clip drain wire` (sau khi bỏ chính cụm "sternotomy wire(s)") | 80 | Support Devices NOT_MENTIONED **28** (73). Khớp chuỗi con nên "portable", "baseline" cũng loại một report: 80 là tập bảo thủ |
| "pleural plaque" hoặc "pleural calcification" | 52 | Pleural Other NOT_MENTIONED **43** (46) |
| "pulmonary congestion" | 11 | Edema UNCERTAIN **7** (7) |

Đếm lại trên cột `report` đầy đủ của Plus CSV (nối qua `source_row_index`) cho "airspace disease",
mọi mention "sternotomy wire", "pleural plaque/calcification" và "pulmonary congestion": n lệch tối
đa 2 study (vd pulmonary congestion 13 so với 11), và chiều của nhãn giữ nguyên.

## Phase 4b — parser với LLM hosted (đo 2026-10-20)

Không được chọn (D22 chọn OpenAI); chỉ giữ kết luận. `gemini-2.5-flash` qua `OpenAICompatClient`, **cùng
prompt `c975e4ca…` và benchmark `8d19c619…`** với qwen 3B, không chỉnh gì cho Gemini: heldout concept / status
/ view **0,994 / 0,980 / 0,947** so với 0,880 / 0,718 / 0,890 của qwen 3B, 0 output sai định dạng. Vậy prompt
không phải nút thắt; lỗi parser của Phase 4 là do model 3B (nó đảo cả câu dương đơn giản như "pneumothorax"
thành exclude). File: `artifacts/metrics/parser_eval_heldout_gemini-2.5-flash.json`. Bẫy lộ ra: T21, T22.

## Phase 4c — chọn model OpenAI cho parser (đo 2026-10-20)

**Luật chọn viết trước khi có số** (`scratchpad/bakeoff.py`, kết quả ở
`artifacts/metrics/llm_bakeoff_dev.json`): đo trên **dev** (40 câu), không đụng heldout; gate cứng 0
output sai định dạng và cả ba macro-F1 ≥ 0,90; chi phí = token đo được × giá niêm yết
(developers.openai.com/api/docs/pricing, đọc 2026-10-20); chọn model rẻ nhất qua gate, trừ khi F1 trung
bình kém model tốt nhất > 0,02 thì chọn model rẻ nhất trong dải 0,02. Snapshot có ngày, không dùng alias.
Cùng prompt `c975e4ca…`, cùng benchmark `8d19c619…`.

| Model (dev, 40 câu) | Concept / Status / View | $ / 1.000 câu | Cache | p50 | Gate |
|---|---|---:|---:|---:|---|
| gpt-5-nano-2025-08-07 | 0,973 / **0,820** / 0,900 | 0,056 | 82% | 1,3 s | FAIL |
| gpt-4.1-nano-2025-04-14 | 0,992 / 0,913 / 0,900 | 0,098 | 91% | 0,8 s | PASS, ngoài dải 0,02 |
| **gpt-5-mini-2025-08-07** | 0,992 / 0,996 / 1,000 | **0,220** | 91% | 1,2 s | **PASS, chọn** |
| gpt-4o-mini-2024-07-18 | 1,000 / 1,000 / 1,000 | 0,228 | 94% | 1,3 s | PASS |
| gpt-5.4-nano-2026-03-17 | 1,000 / 1,000 / 1,000 | 0,267 | 66% | 1,2 s | PASS |
| gpt-4.1-mini-2025-04-14 | 1,000 / 1,000 / 1,000 | 0,423 | 87% | 1,1 s | PASS |
| gpt-5.4-mini-2026-03-17 | 1,000 / 1,000 / 1,000 | 0,503 | 94% | 1,4 s | PASS |

Prompt ~2.590 token mỗi câu, nên giá **input có cache** quyết định chi phí; tỉ lệ cache dao động 66–94%
giữa các lần nên hạng chi phí của những model sát nhau (gpt-5-mini 0,220 so với gpt-4o-mini 0,228) nằm
trong nhiễu. Với tham số trong `models.yaml:llm.hosted_request` mọi model đều 0 token suy luận.

**Heldout, chạy một lần** với model được chọn (`artifacts/metrics/parser_eval_heldout_gpt-5-mini.json`):
concept **0,995**, status **0,988**, view **0,947**, exact **0,981**, schema-valid **1,000**, retry 0,
fallback 0, p50 / p95 **1.305 / 1.814 ms** → **PASS cả verdict**, parser đầu tiên của dự án qua được.
EN 1,000 trên mọi chỉ số; VI 0,990 / 0,976 / 0,890, exact 0,963. Ba lỗi đều VI: vi-044 "vôi hóa màng
phổi" không ra concept nào, vi-052 "có thể thấy rõ gãy xương đòn" thành `UNCERTAIN` (câu bẫy cấu trúc
của D21), vi-079 "tư thế trước-sau" thành PA (Gemini cũng sai đúng câu này). Đầu-cuối qua API: 18/18 câu
dương parse đúng; use case 6 ra Cardiomegaly + Edema, exclude Pleural Effusion, `vi`, mode label 849 ca,
0/10 vi phạm; `pytest -m llm` 2 passed, 1 xpassed.

gpt-4.1-mini và gpt-4o-mini (nhận `temperature: 0`) cũng được đo heldout một lần theo yêu cầu người dùng; cả
hai kém hơn và không tất định hơn, không được chọn. Số và bài học: T25.

**Lưu ý phương pháp:** heldout parser giờ đã được xem cho bốn model (Gemini, gpt-5-mini, gpt-4o-mini,
gpt-4.1-mini). gpt-5-mini là lựa chọn của luật viết trước trên dev và người dùng giữ nó sau khi thấy cả ba
số heldout, nên con số 0,995 / 0,988 / 0,947 nên coi là hơi lạc quan. Phase 7 đo lại parser trên bộ
truy vấn chưa ai nhìn.

## Phase 5 (đo 2026-10-16 → 2026-10-21)

**Cách đo:** probe thiết kế ngày 2026-10-16 trên `main` 9eea090 bằng `uv run python
artifacts/metrics/report_probe.py` và `uv run python artifacts/metrics/fts_probe.py` (chỉ đọc
`data/canonical/{reports,labels}.parquet`, FTS5 in-memory, nhãn resolve theo G1; chạy lại 2026-10-21: số
đếm trùng, latency lệch dưới 2 ms). Phần còn lại trên nhánh `feat/report-search-hybrid`, làm tại chỗ chứ
không qua worktree, vì worktree không thấy `data/`, `artifacts/`, `.env` (gitignored). Report index dựng
và lọc Qdrant đo ngày 2026-10-20; lưới dev ở `241881e`; fusion commit ở `a1361db`; heldout lần 1 ở
`a1361db`, lần 2 ở `47c2943` (khai báo, xem Nhật ký 2026-10-21); HTTP và smoke ở `a1361db`; integration
test ở `c201a9e`. Qdrant Docker, build `20261011_chexpert25k_biomedclip_v1`, `validate-indexes` 7/7 PASS.
Ollama không tham gia; `auto` = OpenAI `gpt-5-mini-2025-08-07`, prompt `c975e4ca…`; `.env` chỉ có key. API
gọi qua `127.0.0.1:8001` vì một uvicorn cũ (code trước Phase 5, không phải của lần đo) giữ cổng 8000.
`exclude_violation@10` chia cho số kết quả **trả về** (bỏ câu không có exclude hoặc không trả gì); lưới chấm
bằng cùng relevance với lần đánh giá. Trước lần 2, report index được dựng lại để `meta` mang
`concepts_sha256`/`git_commit` hiện tại (`corpus_sha256` không đổi). `concepts.yaml` sha256 giờ là
`22095afb…` (thêm `report:` và `blocked_phrases`). Chi tiết: `docs/retrieval-evaluation.md`.

| Hạng mục | Giá trị | Đo bằng |
|---|---:|---|
| Report | 25.000 dòng, 25.000 `study_id` duy nhất, 24.903 `search_text` khác nhau; 6.419 study có findings | `report_probe.py` (polars) |
| Độ dài `search_text` | p50 343 · p95 889 · max 3.250 ký tự | `report_probe.py`, `str.len_chars` |
| SQLite | 3.50.4; FTS5 và tokenizer `porter unicode61` có sẵn | `sqlite3.sqlite_version`; `fts_probe.py` |
| FTS thô trên dạng `en` của lexicon parser: khớp / trong đó ABSENT | pneumothorax 9.775 / **7.275** · consolidation 7.658 / 2.763 (+3.278 UNCERTAIN) · pleural effusion 11.059 / 2.785 · edema 10.285 / 2.276 → report leg không đứng một mình được (D23) | `report_probe.py`: MATCH OR các dạng `en`, join nhãn resolve |
| `en` lệch cách report viết: khớp / PRESENT | Lung Lesion 140 / 2.442 · Lung Opacity 2.410 / 18.394 · Fracture 885 / 2.302 · Pleural Other 548 / 1.065 (D27) | như trên; số PRESENT từ `fts_probe.py` |
| Từ vựng thử rộng hơn, porter stem: recall (precision) trên study PRESENT | Lung Lesion 0,40 (0,69) · Lung Opacity 0,68 (0,93) · Fracture 0,50 (0,73) · Pleural Other 0,27 (0,49) · Atelectasis 0,60 (0,49) · Support Devices 0,76 (0,93) | `fts_probe.py` |
| bm25 top-100 `"pleural effusion"` | 100/100 PRESENT | `report_probe.py`, `order by bm25` |
| bm25 trong tập id cho phép (bảng TEMP), nạp + truy vấn | 500 id 0,4 + 11,2 ms · 2.760 id 3,4 + 13,1 ms · 10.000 id 10,7 + 16,3 ms | `fts_probe.py`, 2026-10-16 |
| Hình thái `search_text` | xuống dòng cứng giữa câu; impression đánh số `1. 2.` → T26 | đọc mẫu; `fts_probe.py` tách thử theo newline ra mảnh `"There "` |
| Report index thật | 25.000 dòng · 32.759.808 B · 1,048 s (dựng lại 2026-10-21: 0,998 s) · `corpus_sha256` `60b72389…` | `time uv run cxr build-report-index`; `ls -l` |
| `meta.concepts_sha256` là hash byte của working copy | lần dựng đầu ghi `2d455b2f…`, đúng là `c2f29fc8…` (lexicon Phase 4) ở dạng CRLF | `git show 07a990b:configs/concepts.yaml \| sed 's/$/\r/' \| sha256sum` |
| Bất biến rule parser sau khi thêm `report:` và `blocked_phrases` | 200/200 `SearchQuery` giống hệt trên cả benchmark | script so `rule_parser.py` + `concepts.yaml` của `main` với bản mới, 2026-10-20 |
| Lọc Qdrant trong tập cho phép, p50 / p95 ms (n = 19) | 10.000 id: `HasIdCondition` **30,7 / 35,6** (chọn) · `MatchAny` + payload index 31,5 / 47,0 · top 2.000 rồi giao 30,9 / 35,5. 25.000 id: 46,0 / 62,0 · 45,4 / 63,7 · 31,4 / 34,7. Top 2.000 chỉ giữ 40/100 hit ở 500 id | `LLM_DISABLED=true uv run python artifacts/metrics/qdrant_filter_probe.py`, một lần chạy |
| Tập cho phép của report leg bằng `json_each` | 500 id 24,6 ms · 10.000 id 37,1 ms, trên chỉ mục **tổng hợp** 25.000 report, không phải corpus thật. Trên chỉ mục thật chưa bấm giờ riêng | script tạm, trung bình 5 lần |

**Heldout, lần 2** (152 câu có gold, 8 câu `other` bỏ; policy mặc định; `uv run cxr eval-retrieval --split
heldout --parsers gold rule auto --modes label vector hybrid --policies negation_both`, chạy 6 phút 47 giây):

| Parser / mode | p@5 | p@10 | nDCG@10 | nDCG nhị phân | MRR | violation@10 | returned@10 | phủ evidence / polarity | p50 / p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| gold label | 1,000 | 1,000 | 0,531 | 1,000 | 1,000 | 0,000 | 1,000 | 0,537 / 0,849 | 18 / 24 |
| gold vector | 0,382 | 0,359 | 0,235 | 0,371 | 0,528 | 0,319 | 0,908 | 0,510 / 0,837 | 179 / 196 |
| **gold hybrid** | 1,000 | 1,000 | **0,607** | 1,000 | 1,000 | 0,000 | 1,000 | 0,922 / 0,862 | 207 / 259 |
| rule label | 0,655 | 0,657 | 0,349 | 0,657 | 0,673 | 0,063 | 0,730 | 0,561 / 0,814 | 16 / 24 |
| rule vector | 0,297 | 0,284 | 0,179 | 0,294 | 0,433 | 0,359 | 0,789 | 0,525 / 0,793 | 175 / 194 |
| rule hybrid | 0,730 | 0,722 | 0,427 | 0,722 | 0,752 | 0,136 | 0,868 | 0,924 / 0,819 | 201 / 252 |
| auto label | 0,980 | 0,980 | 0,520 | 0,980 | 0,980 | 0,000 | 0,993 | 0,536 / 0,848 | 18 / 23 |
| auto vector | 0,378 | 0,356 | 0,234 | 0,368 | 0,526 | 0,327 | 0,908 | 0,511 / 0,827 | 173 / 193 |
| auto hybrid | 0,980 | 0,980 | 0,595 | 0,980 | 0,980 | 0,000 | 1,000 | 0,921 / 0,855 | 204 / 256 |

Verdict **PASS**: single p@10 1,000 (ngưỡng 0,70), constraint satisfaction 1,000 (0,95), gain (0,607 −
0,235) / 0,235 = **1,579** (0,05). Mọi số `gold` trừ latency, kể cả verdict, **trùng từng bit** với lần 1
(so dict trên `metrics`, `by_group`, `by_language`, `by_policy`, `per_query`); lần 1 chậm hơn ở mọi ô latency
(gold label / vector / hybrid p50 25 / 232 / 300 ms), chưa rõ vì sao. `fallback_rate` 0,000 cho `rule` và
`auto` ở cả hai lần. `auto` trượt 3/152 câu (`vi-044`, `vi-052`, `vi-079`: đúng ba lỗi parse của Phase 4c);
giữa hai lần ít nhất 4 parse `auto` khác nhau (cận dưới, đọc từ `per_query`). Gold vector trả dưới 10 kết quả
ở 14/152 câu, đúng 14 câu VI chỉ có exclude (không có vector leg, D26); với chúng hybrid cũng không có report
leg nên trả theo thứ tự label.

| Gold, theo nhóm (n) | nDCG@10 label / vector / hybrid | vector p@10 | vector violation@10 |
|---|---|---:|---:|
| single (70) | 0,557 / 0,309 / **0,664** | 0,440 | n/a |
| multi (60) | 0,580 / 0,199 / **0,635** | 0,290 | 0,307 |
| negation (40) | 0,395 / 0,140 / **0,435** | 0,302 | 0,319 |
| negation (40), `explicit_absence_required` | 0,612 / 0,059 / **0,640** | 0,082 | 0,319 |
| uncertainty (16) | 0,590 / 0,079 / **0,640** | 0,125 | n/a |
| view (20) | 0,570 / 0,449 / **0,657** | 0,620 | 0,100 |
| en (76) · vi (76) | 0,531 / 0,259 / 0,615 · 0,531 / 0,212 / 0,599 | 0,414 · 0,304 | 0,320 · 0,317 |

Gold label và hybrid có p@10 = 1,000 ở mọi dòng. Polarity của evidence yếu ở hedge: nhóm `uncertainty` của
hybrid chỉ khớp 0,292.

| Hạng mục | Giá trị | Đo bằng |
|---|---:|---|
| Lưới dev (gold, hybrid, 38 câu), nDCG@10 | `rrf_k` 10, (vector 0,5, report 1) **0,6565** → chọn · 60, (0,5, 1) 0,6545 · 200, (0,5, 1) 0,6476 · mặc định 60, (1, 1) 0,6441; mọi (1, 0,5) thấp nhất (0,617–0,626); p@10 1,000 ở cả 9 | `uv run cxr eval-retrieval --split dev --parsers gold --modes hybrid --grid --out artifacts/metrics/retrieval_dev.json` ở `241881e` |
| Dev tham chiếu, fusion đã chọn | gold label / vector / hybrid nDCG@10 0,524 / 0,302 / 0,657; ba parser giống hệt | `uv run cxr eval-retrieval --split dev --parsers gold rule auto --out artifacts/metrics/retrieval_dev.json` |
| HTTP p50 / p95 (n = 120) | `vector`, server `LLM_DISABLED=true` **156 / 171 ms** (Phase 3: 150 / 158) · `hybrid`, LLM tắt 174 / 219 ms · `hybrid`, LLM bật 188 / 221 ms (trúng cache parse; lần lặp đầu, lần duy nhất gọi LLM, bị bỏ) | `uv run cxr bench-vector --api-url http://127.0.0.1:8001 --mode {vector,hybrid}`, qua `uv run python artifacts/metrics/http_phase5.py A\|B 8001` |
| Encode + Qdrant trong tiến trình, warm p50 / p95 | 139,5 / 142,3 · 141,0 / 143,6 · 140,4 / 142,8 ms (ba lần bench trên) | `vector_baseline_phase5_*.json` |
| Cold start | `/health` 200 sau **7,6 s** (LLM tắt) / **8,1 s** (LLM bật) từ lúc gọi `uv run uvicorn`; `cold_ms` của bench 6.995–7.406 ms | `http_phase5.py`, poll 0,5 s |
| Một lần gọi LLM thật (parse) | `llm_ms` 1.079–1.756 ms, 6 câu | `smoke_phase5.py` |
| API smoke, use case 3 "Find edema with no pleural effusion." | vector **6/10** `Pleural Effusion: PRESENT` (Phase 4 cũng 6/10), hybrid **0/10**, dưới cả hai policy; `total_matches` 2.760 / 1.532 | `uv run python artifacts/metrics/smoke_phase5.py http://127.0.0.1:8001 artifacts/metrics/smoke_phase5.json` |
| API smoke, sáu use case × ba mode | hybrid: 10/10 kết quả có evidence ở mọi câu, 0 warning, `latency_ms` 188–227; 5/6 hit đầu có `report_rank` 1 và không có `vector_rank` (`rrf` 1/11) → thứ hạng chủ yếu do report dẫn | như trên |
| Integration hybrid, 50 ảnh thật | 11/50 study thỏa "edema, không tràn dịch"; hybrid trả đúng tập đó, evidence là lát cắt đúng | `uv run pytest -m integration` (3 passed) |
| Evidence trên 1.000 report thật, mọi concept PRESENT | **2.375 / 3.803 cặp (62,5%)** có câu evidence, mọi câu là lát cắt đúng. Theo concept: No Finding 23,6% · Enlarged Cardiomediastinum 4,4% · Cardiomegaly 42,9% · Lung Opacity 67,4% · Lung Lesion 45,2% · Edema 62,4% · Consolidation 57,9% · Pneumonia 57,9% · Atelectasis 58,5% · Pneumothorax 66,4% · Pleural Effusion 72,2% · Pleural Other 45,1% · Fracture 52,5% · Support Devices 77,9% | `uv run pytest tests/integration/test_hybrid_live.py -m integration -s` |
| Gold, nDCG@10 hybrid / label / vector trên câu heldout chỉ có exclude (gold `include` rỗng, `exclude` khác rỗng) | 14 câu EN **0,409 / 0,328 / 0,210**, 62 câu EN còn lại 0,661 / 0,577 / 0,269 · 14 câu VI 0,328 / 0,328 / 0,000, 62 câu VI còn lại 0,661 / 0,577 / 0,260. Hybrid EN hơn VI (0,615 / 0,599) chỉ ở 14 cặp câu này | `uv run python artifacts/metrics/exclude_only_probe.py artifacts/metrics/retrieval_eval_heldout_20261021_run2.json` (chỉ đọc; 2026-10-21): trung bình `per_query[id]["ndcg@10"]` dưới `parsers.gold` của JSON đóng băng lần 2 |
| Report leg của 14 câu EN đó, top 10 bm25 trong tập cho phép (policy mặc định) | MATCH quote nguyên văn câu, kể cả từ của concept bị loại (`"show" OR "films" OR "with" OR "evidence" OR "pneumothorax"` cho en-023). **44/140** hit trích một câu phủ định concept bị loại (en-023, en-027 8/10; en-051, en-087 0/10); **37/140** trích cùng một câu hai lần, vd `"1. EXPIRATORY FILM SHOWING NO EVIDENCE OF PNEUMOTHORAX."` vừa là Pneumothorax `negated` vừa là `mention` free-text. Đây là report leg đứng riêng, chưa hợp nhất với vector leg | cùng script: `build_match` + `search_reports` trên `data/indexes/report_search.sqlite` (`mode=ro`), `EvidenceMatcher.build` |
| Ablation nhánh trong hybrid, **dev** (38 câu gold, policy mặc định, fusion đang dùng), nDCG@10 | label 0,524 · vector 0,302 · lọc + chỉ vector **0,611** · lọc + chỉ report **0,658** · hybrid 0,657. Hybrid so với chỉ report: thắng 12 / hòa 19 / thua 7 câu, chênh lớn nhất +0,009 / −0,042; top 10 trùng trung bình 9,5/10. Trong 380 hit top 10 của hybrid: 69 có cả hai hạng, 281 chỉ report, **0 chỉ vector**, 30 không nhánh nào (thứ tự label). Theo cấu trúc: với `rrf_k` 10 và trọng số (0,5, 1), study chỉ có ở vector leg được tối đa 0,5/11 ≈ 0,045 < 1/20 = 0,05 của report hạng 10, nên không bao giờ vào top 10 khi report leg có ≥ 10 ứng viên. Heldout chưa chạy ablation này; relevance đọc cùng văn bản với report leg (thiên lệch như P11) | `uv run python artifacts/metrics/leg_ablation_probe.py <out.txt>` ở `0940d3e` (chỉ đọc; tắt nhánh bằng cách thay `pipeline._vector_search` / `pipeline.search_reports` bằng hàm trả `[]`), 2026-10-22 |

## RadGraph-XL (đo 2026-10-22, chưa dùng ở V1)

**Cách đo:** `uv run python artifacts/metrics/radgraph_align_probe.py <out>` (cách khớp với CSV) và
`uv run python artifacts/metrics/radgraph_content_probe.py <out>` (nội dung trên corpus 25.000), chỉ đọc,
join theo text đã chuẩn hóa (T29). Đếm theo lần nhắc (entity), không theo study, trừ khi ghi rõ.

| Hạng mục | Giá trị |
|---|---:|
| Bản ghi `section_impression.json` / `section_findings.json` | 223.292 / 57.805; khớp text CSV **100%** cả hai; 157.063 / 39.160 khớp đúng một dòng, phần còn lại trùng text giữa nhiều dòng |
| Phủ corpus | impression 25.000 / 25.000 · findings 6.231 / 6.419 study có findings |
| Entity / study | trung bình 24,9 · trung vị 21 · p95 53 |
| Nhãn entity | Observation present 277.453 · Anatomy present 273.098 · Observation **absent 40.654** · Observation **uncertain 29.011** · measurement 2.374 · Anatomy uncertain/absent 131 |
| Quan hệ | `modify` 259.362 · `located_at` 152.124 · `suggestive_of` 22.508; 24.913 / 25.000 study có ≥ 1 `located_at`; 21.760 study có anatomy left/right/bilateral |
| Từ vựng thô (lowercase, chưa chuẩn hóa) | anatomy 3.581 chuỗi (1.861 chỉ gặp 1 lần) · observation 7.104 chuỗi (3.773 gặp 1 lần); entity là từ đơn ghép bằng `modify` ("left" → "lobe" ← "lower"), không có CUI |
| Thử truy vấn giải phẫu: opacity/opacities/consolidation/atelectasis `located_at` anatomy có "left" + lower/base/basilar (tính cả modifier) | 3.864 lần nhắc present · 15 absent |
| Nguồn gốc | `data_split: "inference"`: output của model, không phải nhãn người; chưa kiểm chất lượng bằng mắt trên mẫu nào |

## Image probe (đo 2026-10-23, nhánh `feat/image-concept-probe`)

**Cách đo:** pool `data/probe_pool/` (`pool.json` ghi `a16f1d3`), embed CPU ở `0940d3e` (`shards.json`), build
`20261023_pool88k_biomedclip_v1`; `uv run cxr train-probe` ở `6f774bb` (sinh lại ở `667cafb` chỉ để thêm
`selection.n_val`, `W`/`b` và `corpus_scores.parquet` trùng từng byte); lưới dev, dev tham chiếu, ablation,
heldout và `bench-vector` ở `6f774bb` với `retrieval.yaml` chỉ đổi comment, mỗi lệnh có `--out` riêng của
nhánh (lệnh đầy đủ: `docs/image-probe.md`, *Tái lập*). Corpus build `20261011_chexpert25k_biomedclip_v1`,
`validate-indexes` 8/8, Qdrant Docker, không có API chạy. Heldout chạy một lần (8 phút 2 giây), đóng băng
`retrieval_eval_heldout_image_probe_20261023.json`, sha256 `af019789…`. Bảng theo concept, nhóm và ngôn
ngữ: `docs/image-probe.md`.

| Hạng mục | Giá trị | Nguồn |
|---|---|---|
| Pool ngoài corpus | 88.853 study / 46.682 bệnh nhân (AP 71.692 / PA 17.161), chỉ `train/`, 0 bệnh nhân chung với corpus (kiểm lại bằng polars); embed CPU 0,0815 s/ảnh, 7.242 s | `pool.json`, `shards.json` |
| Chọn λ: macro strict AUROC trên pool-val (13 concept, 9.037 study) | 0,1 **0,6196** · 1 0,6166 · 10 0,6114 · 100 0,6051 → λ 0,1, ở mép lưới chọn trước | `image_probe.json:probe.selection` |
| Corpus: macro strict AUROC 14 concept | zero-shot 0,599 · ensemble (A1) 0,599 · prompt đối (A2) 0,640 · **probe 0,659**; probe − zero-shot +0,060 [0,050; 0,069] paired → S1 PASS. Probe hơn zero-shot 13/14 (Fracture 0,487 < 0,520), hơn A2 7/14; yếu: Fracture 0,487, Lung Lesion 0,533, Pleural Other 0,563 | `image_probe.json:probe.comparison` |
| Lưới dev, 36 cấu hình (gold, hybrid, 38 câu) | mốc (`rrf_k` 10; vector 0,5 / report 1 / image 0) 0,6565 = `chosen`; tốt nhất (10; 0 / 1 / 0,5) 0,6605, +0,0040 < 0,005 → **S3 FAIL**, `image` giữ 0 | `retrieval_dev_image_probe.json:grid` |
| Ablation dev, nDCG@10 | image 0,367 · lọc + chỉ vector 0,611 · lọc + chỉ image 0,632 · lọc + chỉ report 0,658 · hybrid 0,657 | `…:ablation.configs` |
| Heldout gold, nDCG@10 (152 câu) | label 0,531 · vector 0,235 · **image 0,299** · hybrid 0,607 → S4 PASS. Gold và rule ở label / vector / hybrid trùng từng bit với Phase 5 lần 2 | `eval.parsers.gold.modes.<mode>.metrics` |
| `image_vs_vector` (138 câu vector trả lời) | nDCG@10 0,297 / 0,259, thắng / hòa / thua 77 / 12 / 49, vi phạm exclude 0,165 / 0,319 (26 câu) → **S2 PASS**. 14 câu VI chỉ có exclude: image 0,312, thứ tự label 0,328 | `eval.image_vs_vector` |
| Latency p50 / p95 ms | trong tiến trình (gold): image 22 / 33, hybrid 215 / 279; `bench-vector` warm tổng 152,2 / 171,5, cold 7.450 | `latency_ms`; `vector_baseline_image_probe.json` |

## Pipeline một đường, CXR-CLIP zero-shot (đo 2026-10-24, nhánh `feat/cxrclip-image-only`)

**Cách đo:** cây làm việc của bước 2 (trước commit), native trên CPU, build `20261023_chexpert25k_cxrclip_v1`.
Dev: `uv run cxr eval-retrieval --split dev --parsers gold rule auto --out artifacts/metrics/retrieval_dev_onepath_cxrclip.json`
(policy mặc định và `explicit_absence_required` cho nhóm negation; 38 câu có gold). Smoke: 7 câu qua uvicorn thật với
parser gpt-5-mini, log `notes/cxrclip_step2_smoke.log` (vi phạm đếm offline từ `labels.parquet`; API không đọc nhãn).

| Hạng mục | Giá trị | Đo bằng |
|---|---:|---|
| Dev, nDCG@10 nhị phân (số chính) · graded (số phụ) | **0,482** · 0,314 | `eval.parsers.<parser>.metrics` |
| Dev, P@10 · MRR · exclude_violation@10 | 0,479 · 0,596 · 0,400 | như trên |
| Dev: gold, rule và auto | trùng từng số, và đúng là phải trùng: rule và auto parse **trùng gold ở cả 38/38 câu** (include, exclude, views và câu mẫu); phép so không rỗng: 17 câu mẫu khác nhau, ghép lệch hàng thì 38/38 bị bắt | `artifacts/metrics/onepath_parse_vs_gold.py` |
| So với `image_joint` của nhánh V1.1 (cùng encoder, `retrieval_dev_lair.json`) | 36/38 câu trùng từng số; 2 câu No Finding (`en-012`, `vi-012`) P@10 0,3 → 0,5 nhờ "normal chest" thay "no finding" | so `per_query` |
| Latency dev, trong tiến trình | p50 / p95 71 / 92 ms (gold) | `latency_ms` |
| Smoke: câu có phủ định, top 10 (Wilson 95% của 4/10: [0,17; 0,69]) | "không có tràn khí màng phổi": 4/10 mang Pneumothorax PRESENT, **nền corpus 14,4%** (≈ 2,8 lần nền, cận dưới vẫn trên nền); "pulmonary edema without pleural effusion": 4/10 mang Pleural Effusion PRESENT, **nền corpus 57,0%**, trong study có Edema 67,0% (dưới nền, nhưng trong khoảng tin cậy). 10 kết quả là quá ít để kết luận; bench câu mẫu ở bước 3–4 mới đo | `notes/cxrclip_step2_smoke.log`, `artifacts/metrics/onepath_base_rates.py` |
| Evidence trong response thật (cùng câu edema, top 10, `TestClient` trên state thật) | 8/10 kết quả có trích dẫn; polarity affirmed 10 · negated 2 · uncertain 1; mỗi trích dẫn có `concept`, `text`, `start`, `end`, `polarity`. 25.000/25.000 study của corpus có report không rỗng, nên "ảnh không có report" chỉ kiểm được bằng unit test (`test_a_study_without_a_report_comes_back_with_empty_evidence`: rỗng, không lỗi, không warning) | đếm `reports` trong `report_search.sqlite` |

## Dữ liệu cho KG (đo 2026-10-25, trên `main`)

**Cách đo:** `uv run python artifacts/metrics/kg_data_probe.py <out>` (chỉ đọc): 25.000 study của corpus nối với Plus CSV
qua `source_row_index` (nối 25.000/25.000), đếm theo study, khớp chuỗi con không phân biệt hoa thường theo regex trong
script, nên là cận trên (vd "blunt" cũng khớp "blunt trauma"; chưa đọc mẫu). Trên đĩa không có file UMLS, RadLex hay
ontology nào (`find data -maxdepth 4`); `radgraph_xl/` chỉ có `section_findings.json` và `section_impression.json`.

| Hạng mục | Giá trị |
|---|---:|
| `section_clinical_history` / `section_history` có nội dung | 61,7% / 16,5%; có ít nhất một: **78,2%** |
| Tuổi / giới | 99,9% / 100% |
| Triệu chứng ghi trong hai section đó | khó thở **4,4%** · đau ngực 2,4% · sốt 2,4% · ho 1,9% · thiếu oxy 1,4% · ho ra máu 0,1%; chấn thương/té 2,2% |
| Dấu hiệu hình ảnh ghi trong impression + findings | "blunt" (góc sườn hoành tù) **3,3%** · "layering" 1,4% · meniscus 2 study · perihilar/bat-wing 2,6% · vascular congestion/cephalization 1,1% · volume loss 1,2% · air bronchogram 0,4% · Kerley 0,2% · pleural line/deep sulcus 0,2% · cardiothoracic ratio 1 study |
| 10 history ngẫu nhiên (seed 20260101) | phần lớn là bối cảnh (ICU, thủ thuật, chẩn đoán lúc nhập viện); 2/10 ghi một triệu chứng ngực (thiếu oxy, đau ngực) |

**Hệ quả thiết kế:** dữ liệu từng ca hầu như không ghi triệu chứng, nên không dựng được cạnh ảnh → triệu chứng từ
dữ liệu; triệu chứng chỉ nối với ảnh qua bệnh (ảnh → bệnh → triệu chứng trong KG). Report gọi tên bệnh, ít khi tả dấu
hiệu, nên cạnh bệnh → dấu hiệu hình ảnh phải đến từ nguồn tri thức ngoài (P15), không trích được từ corpus.

**Từ vựng RadGraph-XL trên corpus** (`uv run python artifacts/metrics/radgraph_vocab_probe.py <out>`, cùng phép nối text
T29, đếm theo lần nhắc, token viết thường):

| | Lần nhắc | Chuỗi khác nhau | ≥ 20 lần | Số chuỗi phủ 80% / 90% / 95% / 99% |
|---|---:|---:|---:|---|
| Token quan sát | 347.724 | 7.104 | 782 | 135 / **343** / 763 / 3.627 |
| Cụm quan sát (modifier + head, theo `modify`) | 347.724 | 26.078 | 1.423 | 850 / 3.190 / 9.024 / 22.601 |
| Token giải phẫu | 274.997 | 3.581 | 461 | 63 / **142** / 284 / 1.279 |

Nhiều token quan sát đứng đầu không phải finding mà là modifier: "stable" 12.650, "unchanged" 10.576, "small" 8.595,
"mild" 7.064, "tip" 6.803, "low", "normal", "clear". Cụm đầy đủ dàn trải gấp ≈ 9 lần token, nên liên kết vào KG phải ở
mức head token, modifier thành thuộc tính.

**MeSH** (đếm 2026-10-25 qua SPARQL `id.nlm.nih.gov/mesh/sparql`, chỉ descriptor của năm hiện hành: URI không có năm;
tổng 30.594 descriptor, khớp quy mô MeSH; không lọc thì endpoint trộn mọi năm và đếm ra ≈ 12 lần): bệnh đường hô hấp
C08 **254** · bệnh tim C14.280 196 · u vùng ngực C04.588.894 24 · bệnh động mạch chủ C14.907.109 23 · chấn thương ngực
C26.891 7 · **hợp năm nhánh 482**. Triệu chứng: hô hấp C23.888.852 30 · đau C23.888.592.612 50 · thay đổi thân nhiệt
C23.888.119 8.

## UMLS 2026AA Level 0 Subset (đo 2026-10-27, V2-1)

**Cách đo:** MD5 bằng `hashlib.md5` trên `data/umls/raw/umls-2026AA-metathesaurus-level0.zip`; đếm dòng và group-by
bằng polars streaming qua `src/kg/umls_filter.scan_rrf` (tên cột theo MRFILES.RRF), trên MRCONSO trước khi lọc; lọc
bằng `uv run cxr kg-filter-umls` (manifest `data/umls/2026AA/manifest.json`). Bảng đầy đủ (file trong META, CUI theo
SAB, LAT, REL × RELA) ở `notes/v2-inventory.md`, chỉ có trên máy vì `notes/` gitignored. Chỉ số đếm, không có chuỗi
nào lấy từ UMLS (T38).

| Hạng mục | Giá trị |
|---|---|
| Zip | 2.012.898.307 byte; MD5 `7c37d94ed322c244edc060384cd16027`, khớp thiết kế; giải nén 10,43 GiB, 56 file `.RRF` |
| Số dòng (đếm được = MRFILES) | MRCONSO 7.668.674 · MRREL 23.920.986 · MRSTY 2.820.576 · MRSAB 201 |
| SAB trong MRCONSO | 91; MSH 464.015 CUI · NCI 213.759 · FMA 102.597 · HPO 21.569; không có SNOMEDCT_US |
| Ngôn ngữ | 24 giá trị LAT; ENG 6.127.262 dòng; không có `VIE` |
| MRREL của MSH, NCI, FMA, HPO | 432 cặp (SAB, REL, RELA): FMA 131 · MSH 19 · NCI 272 · HPO 10 |
| Phân cấp | `PAR`/`CHD` với RELA `inverse_isa`/`isa`: NCI 234.295 · FMA 104.341 · HPO 23.749; MSH dùng RELA rỗng: 42.746 |
| Bộ phận trong FMA | `regional_part_of` 27.672 · `constitutional_part_of` 21.492 · `part_of` 23 |
| Sau `cxr kg-filter-umls` | MRCONSO 5.681.716 dòng (2.381.838 CUI, 52 SAB; chỉ `ENG`, `SUPPRESS = N`, `SRL` 0) · MRREL 6.111.416 (MSH, NCI, FMA, HPO) · MRSTY 2.820.576 (127 TUI) · MRSAB 201; Parquet 152,1 + 72,7 + 14,9 MiB |

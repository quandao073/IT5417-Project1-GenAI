# WORKLOG

Ngữ cảnh dự án cho coding agent. Đọc file này **trước khi** sửa bất cứ thứ gì.

File này cố tình **không** kể lại thứ chỗ khác đã kể:

| Cần biết | Đọc ở |
|---|---|
| Thay đổi gì, lúc nào | `git log` |
| Phải làm việc thế nào | `CLAUDE.md` |
| Kế hoạch đầy đủ, 8 phase | `docs/ke-hoach-du-an-nho-cxr-semantic-retrieval-v1.md` |
| Cách chạy | `README.md` |
| **Đang ở đâu, đã quyết gì và vì sao, chỗ nào có bẫy** | **file này** |

Đọc tối thiểu: *Trạng thái hiện tại* + *Quyết định đang có hiệu lực* + *Bẫy đã biết*.
Ba mục đó đủ để không phá vỡ thứ gì. Các mục còn lại tra khi cần.

---

## Trạng thái hiện tại

> Cập nhật lại mục này mỗi lần ghi log. Mục này mà cũ thì tệ hơn là không có.

**Cập nhật lần cuối:** 2026-10-15 (sau đợt sửa review cuối Phase 4)

| | |
|---|---|
| Đã xong | Part A · Phase 0a · Phase 1 (canonical corpus 25.000 study) · Phase 2 (model gate: BiomedCLIP) · Phase 3 (vector baseline) · **Phase 4 (parser + label mode)** — heldout parser **FAIL** (xem Phase 4 trong *Sự thật đã đo*) |
| Đang làm | — |
| Tiếp theo | **Phase 5 (report FTS + hybrid)**, hoặc Phase 0b song song |
| Chặn Phase 5 | không có. P4–P9 đang mở nhưng không chặn |
| Chặn Phase 0b | `pnpm` chưa cài trên máy — xem T5 |

Mã thật hiện có: `src/common/{ids,config}.py`,
`src/data/{contracts,labels,join,sample,canonical,audit}.py`,
`src/embeddings/{base,spike,biomedclip,chexzero,xrayclip,embed,index,validate}.py`,
`src/query/{schema,rule_parser,llm_parser,parser}.py`,
`src/retrieval/{image_search,label_filter}.py`, `src/api/{main,dependencies}.py`,
`src/api/routes/*` (có `query.py`), `src/api/schemas/search.py`,
`src/evaluation/{zero_shot,vector_baseline,parser_benchmark,parser_eval}.py`, `src/cli/main.py`.
Ngoài `src/`: `benchmarks/parser_queries_v1.jsonl` (200 câu có gold),
`configs/prompts/query_parser_v1.txt`. `src/explanation/` còn rỗng.

Dữ liệu đã dựng: `data/canonical/*.parquet` (5 bảng, 11 MB) và
`artifacts/audit/corpus_report.json`. Cả hai gitignored — dựng lại bằng
`uv run cxr build-corpus --limit 25000` (~4 phút). Spike Phase 2:
`data/cache/model_spike/` + `artifacts/metrics/model_spike.json`, cũng gitignored.
Phase 3 (build `20261011_chexpert25k_biomedclip_v1`, gitignored hoặc trong Docker):
`data/indexes/embeddings/<build_id>/` (25 shard + `shards.json`, 58 MB),
`data/indexes/build_manifest.json`, volume Docker `cxr-semantic-retrieval_qdrant_data`
(collection `cxr_images_<build_id>`), `artifacts/metrics/vector_baseline.json`. Dựng lại:
`uv run cxr embed` → `uv run cxr index` → `uv run cxr validate-indexes`. Số đo:
`docs/vector-baseline.md`.
Phase 4: `artifacts/metrics/parser_eval.json` (heldout + `label_filter`) và
`artifacts/metrics/parser_eval_dev.json`, gitignored. Dựng lại bằng `uv run cxr eval-parser --label-filter`
(cần Ollama native, `.env` rỗng); kết quả LLM không tái lập từng số (T18). **`cxr eval-parser` ghi đè
`parser_eval.json`**: lần heldout 2026-10-14 được đóng băng ở
`artifacts/metrics/parser_eval_heldout_20261014.json` (sha256 `999fbc70…`). Số dev lấy từ
`uv run cxr eval-parser --split dev --out artifacts/metrics/parser_eval_dev.json`. Script thăm dò của dev
loop: `artifacts/metrics/{views_probe,probe_wide,leak}.py`, prompt thử ở `artifacts/metrics/probe_prompts/`.
Số đo: `docs/parser-evaluation.md`.

Lệnh kiểm tra sau mỗi thay đổi:

```bash
uv run pytest -q                 # 629 tests (4 `model` + 1 `integration` + 3 `llm` bị bỏ chọn mặc định)
uv run pytest -m llm             # 3 test parse thật qua Ollama: 2 passed, 1 xfailed (thứ tự include và language của use case 6, T17, P7)
uv run pytest -m model           # 4 smoke test adapter thật, cần weights đã tải
uv run pytest -m integration     # đầu-cuối 50 ảnh, cần Qdrant Docker (`docker compose up -d qdrant`)
uv run cxr validate-indexes      # 6 x PASS, cần Qdrant Docker
uv run ruff check src tests
uv run cxr --help
docker compose config > /dev/null
```

---

## Sự thật đã đo

> Chỉ ghi số **tự đo trên dữ liệu thật**, kèm cách đo. Không chép từ paper hay tài liệu.
> Append-only: số ở đây không cũ đi, trừ khi dataset đổi.

### Dữ liệu nguồn (đo 2026-09-21 → 2026-09-22)

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

### Bất biến cấu trúc

- **`study_id` duy nhất toàn cục.** `train/` là `patient00001..patient64540`,
  `valid/` là `patient64541..patient64740`, **giao nhau 0 patient**. Nhờ vậy
  `study_id` = `patientNNNNN/studyN` dùng được làm primary key mà không cần kèm
  split. `contracts.py` dựa vào điều này.
- **Plus CSV có newline nhúng.** 10.746.681 dòng vật lý cho 223.462 record —
  trung bình **48 dòng/record**. Xem T2.
- **FTS5 có sẵn** trong `sqlite3` của Python (SQLite 3.50.4). Phase 5 không cần
  build SQLite riêng.

### Chi phí của Phase 1 (đo 2026-09-22)

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

### Kết quả Phase 1 (corpus 25.000, seed 20260101, đo 2026-09-23)

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

### P2 đã giải quyết: prevalence sau resolve (% của 25.000)

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

### Phase 2 spike (đo 2026-10-03)

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

### Phase 3 (đo 2026-10-11)

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

### Phase 4 (đo 2026-10-14)

**Cách đo:** code ở `cd890d8`, cộng prompt và lexicon của commit Task 10 (prompt render sha256
`c975e4ca…`, `concepts.yaml` sha256 `735997a2…`; `parser_eval.json` ghi `git_commit` `cd890d8`).
Ollama 0.40.3 native, GPU RTX 2000 Ada Laptop 8 GB, model `qwen2.5:3b-instruct-q4_K_M`, `.env` rỗng.
Lệnh: `uv run cxr eval-parser --split heldout --parsers rule llm auto --label-filter` (chạy một lần)
và `uv run cxr eval-parser --split dev --parsers rule llm auto --out artifacts/metrics/parser_eval_dev.json`.
API: `uv run uvicorn src.api.main:app` native, Qdrant Docker, gọi qua `127.0.0.1`. Latency parser
bỏ câu đầu (n = 159). Chi tiết: `docs/parser-evaluation.md`.

| Hạng mục | Giá trị | Đo bằng |
|---|---:|---|
| Ollama cài native | 0.40.3; thư mục cài 2,8 GB | `/api/version`; `du -sh %LOCALAPPDATA%\Programs\Ollama` |
| Model 3B q4 | 1.929.912.432 B trên đĩa; 2.159.374.499 B VRAM, trọn trên GPU, context 4.096 | `/api/tags`, `/api/ps` |
| Pull model | 1,9 GB, 72 s | `ollama pull`, 2026-10-13 (spec §1) |
| Gọi `/api/chat` lần đầu sau khi cài | **72 s** (load 27 s + prompt eval 41 s) | 2026-10-13 (spec §1), xem T16 |
| Gọi lần đầu sau khi server khởi động lại, model chưa nạp | 5,3 s (`llm_ms`); lần hai 1,07 s | `cxr parse` hai lần liền; `/api/ps` rỗng trước lần một |
| Prompt cuối | 2.328 token, 20 ví dụ | `prompt_eval_count`; `prompt_examples()` |
| Heldout `auto` (= `llm`) concept / status / view macro-F1 | **0,880 / 0,718 / 0,890 → FAIL** (ngưỡng 0,90) | `parser_eval.json:eval.verdict` |
| Heldout `rule` concept / status / view | 0,793 / **0,862 / 0,945** | như trên |
| Heldout exact match `auto` / `rule` | 0,650 / 0,606 | như trên |
| Heldout `auto` schema-valid · fallback · retry | 1,000 · 0 · 0,0375 (6/160) | như trên |
| Heldout `auto` theo ngôn ngữ, status F1 | EN 0,826 · VI 0,593 | `by_language` |
| Latency p50 / p95 | rule 0,1 / 0,2 ms · llm 756 / 1.366 ms · auto 778 / 1.473 ms | `latency_ms` |
| Dev cuối `auto` / `rule` | 0,912 / 0,827 / 0,833 · 1,000 / 1,000 / 1,000 | `parser_eval_dev.json` |
| Dev `llm` trước → sau 4 vòng chỉnh | 0,761 / 0,606 / 0,900 → 0,909 / 0,841 / 0,900 | `parser_dev_0.json`, `parser_dev_4.json` |
| Hai lần chạy dev, cùng prompt | 5/40 dự đoán `llm` khác nhau (T18) | `parser_dev_4.json` so với `parser_eval_dev.json` |
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

---

## Quyết định đang có hiệu lực

> **Mục quan trọng nhất của file này.** Tài liệu kế hoạch viết trước khi có số đo,
> nên vài chỗ đã bị override. Agent nào đọc kế hoạch rồi làm theo nguyên văn sẽ làm
> sai. Trước khi triển khai một mục của kế hoạch, đối chiếu bảng này.

### Đã chốt

| # | Quyết định | Ghi đè | Vì sao |
|---|---|---|---|
| D1 | **Không chia corpus.** Cả 25.000 study đều searchable. Hold-out ở mức *query*: 150 validation / 250 test, patient-disjoint. Canonical tables **không có** cột `project_split`. | §7.5 (18000/3500/3500) | Không tham số nào học từ corpus nên không có leakage ở mức corpus. Chia chỉ thu nhỏ index và làm nDCG khó so sánh giữa các mode. → `docs/adr/0002` |
| D2 | **Bỏ Tier A/B.** `studies.parquet` **không có** `corpus_tier`, chỉ có `has_report`/`has_impression`/`has_findings`. "Degradation" hiểu là *runtime* (FTS index thiếu/tắt), không phải thành phần corpus. | §7.4, §8.2 | Impression phủ 99,93% và pool gấp 7,5× target → Tier B rỗng theo construction |
| D3 | **Package phẳng tên `src/`**, `api/` nằm trong nó, không có `apps/`. `src` là package thật (`packages = ["src"]`), không phải source root trần. | §15 (`src/cxr_search/` + `apps/`) | Repo đã mang tên dự án; lặp lại là thừa. Giữ *một* namespace vì `cxr` và 2 Docker image đều `pip install`. → `docs/adr/0002` |
| D4 | **Giữ `compose.gpu.yaml`**, không tạo `compose.cpu.yaml`. Base compose là CPU-safe. | §15 | Runtime (api/web/qdrant) không bao giờ cần GPU; chỉ worker lúc embed cần. → `docs/adr/0002` |
| D5 | **LLM mặc định `qwen2.5:3b-instruct-q4_K_M`**, không phải 7b. | `models.yaml` cũ | 7B q4 trên CPU tốn 8–20 s/query, ăn trọn ngân sách 12 s end-to-end. 7B giữ làm nhánh so sánh trong parser ablation |
| D6 | **Không hardcode embedding dim ở bất kỳ đâu.** Phase 2 gate chọn 1 trong 3 candidate; encoder tự báo dim, `build_manifest.json` ghi lại, `cxr validate-indexes` assert khớp. | §16 (`dimension: 768`), `models.yaml` (512) | Ba nguồn mâu thuẫn nhau; để số đo quyết |
| D7 | **Qdrant point id = `uuid5(NAMESPACE_URL, normalized_path)`**, không phải `image_id`. | — | Qdrant chỉ nhận uint64 hoặc UUID; sha256 hex64 không phải cả hai. Cắt xuống 64 bit thì mời câu hỏi birthday-collision vô ích |
| D9 | **`absence_policy` mặc định là `exclude_present`**, không phải `explicit_absence_required`. Cái sau vẫn phơi ra UI và báo cáo song song trong evaluation. | §11.3 | Đo trên corpus thật: median `ABSENT` chỉ 5,0%, bốn concept dưới 2% (Pleural Other 0,9%, Atelectasis 1,3%, Lung Lesion 1,9%, No Finding 0,7%). Bắt buộc `ABSENT` sẽ làm recall sụp với phần lớn concept |
| D10 | **`CHEXBERT_FINDINGS` không phát assertion cho study thiếu findings section.** Nguồn không có đầu vào thì không có assertion, nên không đóng góp dòng nào vào `labels.parquet`. | — | Xem T11 |
| D8 | **Không có disk guard.** §9.2, task "Viết disk guard" của Phase 1 và mục disk guard trong §19 đều **không thực hiện**. | §9.2, §17, §19 | 191 GB trống, V1 thêm <1 GB, `serve_originals` nghĩa là không stage nào ghi ảnh. §9.2 viết cho kịch bản materialize ảnh đã xử lý — kịch bản đó đã bị loại. → `docs/adr/0003` |
| D11 | **Image encoder là BiomedCLIP** (`microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` @ `9f341de`), pin trong `configs/models.yaml`. | `models.yaml` (`selected: null`) | Luật gate cố định trước khi chạy: hard gate rồi macro strict AUROC; hai model đầu cách < 0,02 thì chọn rẻ hơn. BiomedCLIP 0,636 vs XrayCLIP 0,616: hiệu 0,0198, dưới ngưỡng chỉ 0,0002 → hòa, BiomedCLIP nhanh hơn 6,9× nên thắng. MIT (thay vì CC-BY-NC) là lợi ích đi kèm; bước license không được xét. CheXzero trượt (0,461). → `docs/model-decision.md` |
| D12 | **200 ca nhãn bác sĩ (`valid/`, rank 0–199) không bao giờ tham gia chọn model hay tune.** Spike lấy rank 200–1199; `select_sample` raise nếu chạm ca `valid/`. | cách hiểu mặc định của §17 | Đó là ground truth người duy nhất của test. Chọn model là tune; dùng chúng ở đây thì con số test cuối cùng không còn sạch |
| D13 | **Spike chấm bằng 14 prompt đơn concept** (`"chest x-ray showing {}"` + `concepts.yaml:<c>.en[0]`), không phải 30–50 canonical query. | §17 | Query phủ định/đa concept cần parser Phase 4; bịa chúng bây giờ là đo sai thứ. Prompt đơn dương là đúng phép vector mode sẽ chạy |
| D14 | **Embed hai bước qua shard file.** `cxr embed` ghi `data/indexes/embeddings/<build_id>/shard_NNNN.npz` + `shards.json`; `cxr index` upsert vào Qdrant và ghi `build_manifest.json`. `build_id` cố định lúc embed bắt đầu, `index` đọc lại chứ không sinh mới. | §17 ("Index vào Qdrant", không nói cách) | Embed CPU ≈34 phút (đo từ mtime shard của lần chạy CPU dở, chưa tính nạp model) là thứ đắt, còn Qdrant/Docker/payload là thứ hay đổi. Tách ra thì nâng image Qdrant hay `--recreate` không phải embed lại; resume theo shard là lưới an toàn; `cxr score-corpus` chấm chất lượng thẳng từ shard, không cần Qdrant. → `docs/vector-baseline.md` |
| D15 | **API Phase 3 tối thiểu**: `/health`, `/search` (chỉ mode `vector`), `/images/{image_id}`. `query` tiếng Anh dùng nguyên văn làm `semantic_query` (`parser: "passthrough"`); `score` là cosine thô từ Qdrant, không chuẩn hóa. `/query/parse`, `/studies/{id}` hoãn sang Phase 4 và 6. | §13 | Parser là việc của Phase 4; một parser tạm sẽ làm baseline đo sai thứ. Hợp nhất ở Phase 5 dùng RRF theo thứ hạng nên không cần chuẩn hóa score; schema ghi rõ score không phải xác suất |
| D16 | **`config_hash` của vector index chỉ phủ `data.yaml` + `models.yaml:image_encoder`** (`VECTOR_INDEX_CONFIGS`, selector `name:section` hash một section đã parse). `build_manifest.json` ghi `config_scope`; `validate-indexes` tính lại hash theo scope đó, không theo mặc định của code. | Phase 0a (hash cả 4 config `data`, `models`, `retrieval`, `concepts`) | `retrieval.yaml`, `concepts.yaml` và `models.yaml:llm` là tham số lúc truy vấn; Phase 4–5 tune chúng liên tục mà vector trong Qdrant không đổi, nên hash 4 file sẽ làm check 1 FAIL ở mỗi lần sửa. Scope nằm trong manifest nên manifest cũ (scope 4 file) vẫn kiểm được; Phase 5 FTS index sẽ ghi scope riêng |
| D17 | **`SearchQuery` gọn:** `semantic_query`, `include` (concept + `PRESENT`/`UNCERTAIN`), `exclude`, `views`, `language`. Parse **không** sinh `explicit_absence`, `uncertainty_policy`, `top_k`, `intent`; `absence_policy`/`uncertainty_policy`/`top_k` là tham số của request `/search`. | §10.2 | Đó là lựa chọn của người dùng/UI, không phải nội dung câu. Bắt LLM 3B đoán chúng chỉ thêm chỗ sai mà không thêm thông tin, và mặc định `absence_policy` đã được số đo quyết (D9) |
| D18 | **Mode `label` không xếp hạng.** Tập khớp trả theo hoán vị cố định `np.random.default_rng(retrieval.yaml:results.label_order_seed).permutation(n)`, kèm `total_matches`; `score` null. | §11.1, §12.2 (không nói thứ tự) | Nhãn không có tín hiệu xếp hạng. `rank_in_manifest` sẽ đặt 200 ca `valid/` (rank 0–199) lên đầu mọi truy vấn. Hoán vị theo seed cho top-k là mẫu không thiên lệch và tái lập được |
| D19 | **"Parsed-query chips" của UI hoãn sang Phase 6.** API trả `parsed_query` + `warnings` để UI dùng sau. | §17 Phase 4 | Chưa có app web (`web/` rỗng, `pnpm` chưa cài — T5) |
| D20 | **LLM qua Ollama `/api/chat` với `format` = JSON schema** (enum 14 concept), Pydantic validate, retry 1 lần rồi về rule parser. Prompt là file có version `configs/prompts/query_parser_v1.txt`; sha256 của prompt đã render nằm trong khóa cache và trong `parser_eval.json`. Chỉ chỉnh prompt/lexicon trên `dev`. `semantic_query` do một quy tắc chung dựng (`QueryParser._finalize`) cho cả hai parser. | §10.3 (prompt nội tuyến, không nói cách ràng buộc output) | Enum chặn concept lạ ngay lúc decoding; vẫn validate vì server cũ bỏ qua `format`. Hash prompt để cache không trả parse của prompt cũ và mọi số đo truy được về prompt. Quy tắc `semantic_query` chung nên ablation LLM vs rule đo ràng buộc, không đo cách diễn đạt |
| D21 | **Benchmark theo quy ước cụm từ của CheXpert labeler.** `airspace disease` thuộc Lung Opacity (không phải Consolidation), `collapsed lung` thuộc Atelectasis (không phải Pneumothorax), cả trong `concepts.yaml` lẫn gold. Ba câu heldout `en-043`, `vi-027`, `vi-052` là bẫy cấu trúc (phủ định đứng xa, "có thể" không phải hedge), viết lại **sau khi** thấy output của rule parser trên heldout; được khai báo. | — | Nhãn resolve là ground truth của mode `label`; gold lệch labeler thì parse "đúng" vẫn lọc sai tập. Số đo: *Sự thật đã đo*, Phase 4, bảng quy ước cụm từ. Ba câu bẫy là những câu duy nhất kiểm cấu trúc thay vì từ vựng, nên số `negation`/`uncertainty` chủ yếu đo độ phủ từ vựng → `docs/parser-evaluation.md` |

**D4, bổ sung 2026-10-11:** `make embed` giờ chạy **trực tiếp trên CPU** (`uv run cxr embed`, cùng pin
torch 2.4.1 với Phase 2 nên vector tái lập được), không qua Docker worker GPU như §20. Lệnh Docker
GPU giữ làm comment trong `Makefile`. Riêng build `20261011_chexpert25k_biomedclip_v1` được embed
**trên GPU, chạy trực tiếp** (torch `2.4.1+cu124` cài tạm, 293 s thay cho ≈34 phút CPU); mặc định
vẫn là CPU. **Mỗi build chỉ embed trên một thiết bị**: phần CPU dở dang được để riêng ở
`_cpu_partial_<build_id>/`, không resume trộn. Đo sau đó trên 17.000 ảnh chung: cosine giữa vector
GPU và CPU của cùng ảnh nhỏ nhất 0,999995, sai lệch từng phần tử lớn nhất 4,6e-4, tức không trùng
bit (*Sự thật đã đo*, Phase 3); ảnh hưởng lên thứ hạng chưa đo.

**D15, ghi chú:** API nạp cả hai nhánh BiomedCLIP (weights 784,6 MB) dù chỉ cần text tower. Chấp nhận
ở V1 (cold start 5,3 s, warm p95 158 ms). Đây là điểm tối ưu để sau, chưa phải quyết định.

**D15, cập nhật 2026-10-15:** `parser: "passthrough"` không còn. Từ Phase 4, `/search` mode `vector`
encode `semantic_query` do `QueryParser` dựng (D20), có thêm mode `label` (D18) và `/query/parse`.
`/studies/{id}` vẫn hoãn sang Phase 6.

**Tinh chỉnh spec Phase 4 (không phải lệch):**
- D20: `models.yaml:llm.prompt_file` là `prompts/query_parser_v1.txt`, tương đối với `configs/`
  (`llm_parser.load_prompt`), không phải `configs/prompts/...` như spec §9 viết.
- D21: spec §7 cho mỗi câu một list `groups`. Benchmark ghi một `group` chính do tác giả gán; `groups`
  được suy ra từ gold lúc nạp (`parser_benchmark.derive_groups`), và câu có `group` không khớp gold bị từ chối.
- D21: `dev` là câu 001–020 của mỗi ngôn ngữ, không phải các câu chia hết cho 5 như spec §7 viết: với
  chu kỳ 20 câu của `PATTERN`, bội số của 5 chỉ rơi vào nhóm uncertainty, multi và negation.
- Rule parser: cue gần concept hơn thắng, hòa thì uncertainty thắng (review Task 2), thay cho
  "uncertainty thắng mỗi khi cả hai cue cùng chạm tới" của spec §4.3.
- `llm_first_try_valid_rate` của spec §8.1 bị bỏ: suy ra được từ `retry_rate` và `fallback_rate`.
- JSON của `cxr eval-parser`: `model`, `ollama_version`, `prompt_sha256` nằm dưới `eval.llm`; breakdown
  theo ngôn ngữ và nhóm nằm trong từng parser (`eval.parsers.<mode>`).
- Test CLI của spec §10 lúc đầu bị thu hẹp thành test mặc định của argparse; đợt sửa 2026-10-15
  khôi phục chúng (`tests/unit/test_cli_phase4.py`).
- Spec §4.5 bước 6 được sửa: kết quả fallback **không** vào cache, vì một lần LLM lỗi thoáng qua
  không được thành vĩnh viễn cho câu đó; cache hit trả `llm_ms` null.

### Lấp lỗ hổng của kế hoạch

Kế hoạch không quyết những điểm này, nhưng chúng âm thầm quyết định hệ thống chạy đúng hay sai.

| # | Quy tắc | Vì sao bắt buộc |
|---|---|---|
| G1 | **`label_resolution`** (`configs/retrieval.yaml`): ưu tiên `CHEXPERT_V1 > CHEXBERT_IMPRESSION > CHEXBERT_FINDINGS > CHEXBERT_REPORT`, lấy giá trị non-`NOT_MENTIONED` đầu tiên. Ma trận resolve dựng trong RAM lúc API startup, **không** tạo bảng canonical thứ sáu. | §8.3 bảo giữ cả 4 nguồn kèm provenance (đúng cho *storage*), nhưng §12.2 `filter_studies_by_labels` cần **đúng một** status cho mỗi `(study, concept)` |
| G2 | **`report_search.match_from: parsed_concepts`**: chuỗi FTS5 MATCH và evidence extractor chỉ dùng `concepts.yaml:*.en`, **không bao giờ** dùng raw string của người dùng. `semantic_query` luôn là tiếng Anh, kể cả khi input tiếng Việt. | Report là tiếng Anh. Nếu MATCH dựng từ raw string thì report leg đóng góp **bằng 0** cho mọi query tiếng Việt → use case 6 hỏng và hybrid trông tệ hơn thực tế. Tiện thể triệt tiêu FTS5 injection theo construction |
| G3 | **`vector` mode**: có lọc view AP/PA (metadata facet), **không** áp ràng buộc nhãn. | Nó là baseline mà hybrid phải vượt. Tính công bằng của con số nDCG chủ đạo phụ thuộc vào phân biệt này. → `docs/evaluation-design.md` |

---

## Bẫy đã biết

> Những chỗ đã tốn thời gian hoặc sẽ tốn. Đọc trước khi động vào vùng tương ứng.

| # | Bẫy | Cách xử lý |
|---|---|---|
| T1 | **Hai CSV viết path khác nhau.** `train.csv`/`valid.csv` cột `Path` **có** prefix `CheXpert-v1.0-small/`; Plus `path_to_image` **không có**. `artifacts/audit/data_audit.json` báo `exact`=1.0 chỉ vì `audit.py` duyệt filesystem, **không** đọc CSV. | Luôn đi qua `src/common/ids.py:normalize_image_path()`. Đã có 23 test + kiểm chứng trên CSV thật |
| T2 | **Plus CSV có newline nhúng** trong `report`/`section_*`: 48 dòng vật lý / record. | **Bắt buộc dùng CSV parser thật.** Đọc theo dòng sẽ sai ~48×, và sai âm thầm |
| T3 | **`labels/*_fixed.json` là JSONL**, không phải JSON, dù đuôi `.json`. Path bên trong cũng **không** có prefix dataset root. | Đọc từng dòng một object; chuẩn hóa path qua `ids.py` |
| T5 | **`pnpm` chưa cài** trên máy dev. Chỉ chặn phần web. | `corepack enable` |
| T6 | **`findings` chỉ có ở 26,6% dòng.** Không bao giờ được đặt làm điều kiện bắt buộc. | `search_text` = impression làm chính, nối findings khi có |
| T7 | **`NOT_MENTIONED` không bao giờ là `ABSENT`.** Đây là semantics mà toàn bộ luận điểm của đồ án dựa vào. | `contracts.py` chặn ở mức schema; Phase 4 phải có test parametrized 4 status × 4 policy |
| T8 | **Console Windows là cp1252**, in tiếng Việt từ script Python sẽ `UnicodeEncodeError`. | In ASCII trong script kiểm tra, hoặc ghi ra file UTF-8 |
| T11 | **CheXbert chạy trên findings section RỖNG luôn xuất `No Finding = 1`.** Đo được **18.598/18.598 = 100,0%** ở study không có findings, so với 6,3% ở study có. 74,4% study không có section này. Vì `CHEXPERT_V1` và `CHEXBERT_IMPRESSION` thường im lặng về "No Finding", artifact thắng trong `resolve` và gán "bình thường" cho 78,5% corpus. Contract test vẫn xanh — chỉ số liệu mới lộ ra. | `canonical._read_all_labels` loại dòng `CHEXBERT_FINDINGS` khi `findings` null. Có test khóa cả hai chiều. **Nguồn nhãn mới nào cũng phải kiểm kiểu này trước khi tin** |
| T9 | **`section_impression` thường mở đầu bằng ký tự xuống dòng** và có đánh số `1. 2. 3.` bên trong. | `.strip()` trước khi kiểm tra rỗng và trước khi ghép `search_text`. Con số 99,93% ở trên đã tính sau khi strip |
| T10 | **`grep -r` ở gốc repo sẽ quét cả `data/` (12 GB, 223k file)** và treo. | Dùng `git grep` hoặc thêm `--exclude-dir=data` |
| T12 | **Model ID và loader trong kế hoạch sai.** `models.yaml` từng ghi `StanfordAIMI/CheXzero` và `StanfordAIMI/XrayCLIP` — cả hai không tồn tại. CheXzero chỉ có trên Google Drive (state dict OpenAI CLIP). XrayCLIP là transformers `CLIPModel`, không phải open_clip. open_clip 2.24 với `hf-hub:` **bỏ qua revision** âm thầm, và không có `open_clip.__version__`. | ID thật + pin trong `configs/models.yaml`; CheXzero pin bằng sha256 file. BiomedCLIP nạp từ `snapshot_download(revision=...)` rồi load file local. Version: `open_clip.version.__version__`. Chỉ báo cáo của `cxr spike-models --rescore` là quyết định: chạy `--models <tên>` lẻ ghi đè `model_spike.json` bằng báo cáo một model |
| T13 | **Qdrant và shard file.** `compose.yaml` từng pin server `qdrant/qdrant:v1.9.2` trong khi `qdrant-client` là 1.19.1 (lệch 10 minor). Docker Desktop phải chạy trước `cxr index` và API. `QdrantClient.search` đã deprecated. Client `:memory:` bỏ qua payload index (chỉ cảnh báo). `np.savez` tự nối `.npz` vào mọi tên khác đuôi, nên `shard.npz.part` thành `shard.npz.part.npz`. Chạy native mà `.env` có `QDRANT_URL=http://qdrant:6333` (hostname trong Docker) thì không kết nối được. `build_id` mặc định lấy ngày hôm nay. | Server nâng lên `v1.19.2`. Dùng `client.query_points`. Ghi shard qua file handle rồi rename. Unit test spy lời gọi `create_payload_index`; `tests/integration/test_vector_index.py` kiểm `payload_schema` trên server thật. Native: `QDRANT_URL=http://localhost:6333` hoặc `--qdrant-url`. `make setup` chép `.env.example` thành `.env` với `QDRANT_URL=http://qdrant:6333` và đường dẫn `/app/...`, nên mọi lệnh native (`cxr index`, `validate-indexes`, `bench-vector`, uvicorn) cần hoặc không có `.env`, hoặc ghi đè `QDRANT_URL=http://localhost:6333`, `IMAGE_ROOT=data/raw/CheXpert-v1.0-small`, `CANONICAL_DIR=data/canonical`, `INDEXES_DIR=data/indexes` (lần chạy thật chỉ chạy được vì `.env` lúc đó rỗng) Resume embed sang ngày khác: `--build-id <build_id cũ>`. Phase 4: `cxr parse` và `cxr eval-parser` native cũng cần `.env` rỗng, vì `.env.example` đặt `LLM_BASE_URL=http://host.docker.internal:11434` |
| T14 | **`uv run` không gỡ torch CUDA cài bằng `uv pip`.** torch/numpy chỉ nằm trong extra, còn `uv run` sync inexact phần base, nên venv lặng lẽ ở `2.4.1+cu124`; `uv sync --all-extras` thì nhận ra và cài lại. Đã xảy ra: `cxr index` chạy sau khi embed GPU ghi torch `+cu124`, numpy 2.5.2 vào `build_manifest.json`. | Sau khi dùng GPU: `uv sync --all-extras`, rồi kiểm `torch.__version__` có `+cpu`. `uv sync --all-extras --dry-run` cho biết venv có lệch lockfile không |
| T15 | **Image API chưa từng được build hay chạy.** Với cấu hình trước bản sửa này, HF cache (`HF_HOME=/app/.hf_cache`) không ghi được bởi uid 10001 nên `snapshot_download` báo PermissionError và API kẹt 503 vĩnh viễn; kể cả khi ghi được, container đầu tiên tải ≈785 MB trọng số trong cửa sổ healthcheck ≈3 phút. Image api nặng hơn ≈1 GB vì giờ mang torch CPU (spec §4.4) | Đã thêm `mkdir`+`chown` `/app/.hf_cache` vào `docker/api.Dockerfile`, mount volume `hf_cache` vào api, ghim `qdrant-client>=1.19,<1.20` trong extra `api`/`worker` (image pip-install từ pyproject, không từ lock). **Chưa kiểm**: Phase 6 phải chạy `docker compose build api` + `make up` thật |
| T16 | **Ollama: cold start và server không tự lên.** Lần gọi `/api/chat` đầu tiên sau khi cài mất **72 s** (load 27 s + prompt eval 41 s, khởi tạo CUDA; đo 2026-10-13, spec §1), trong khi `llm.timeout_seconds` là 20 → câu đầu rơi về rule parser. Ngày 2026-10-14, server vừa khởi động lại, model đã trên đĩa nhưng chưa nạp (`/api/ps` rỗng): lần đầu chỉ 5,3 s. Vì sao chênh với 72 s thì chưa đo. `keep_alive` mặc định của Ollama là 5 phút. App Ollama (tray) không tự chạy lại sau khi khởi động lại phiên trên máy này. | API warm-up ở thread nền (`build_query_parser(warm_up=True)`); mọi request đặt `keep_alive: 30m`. `uv run pytest -m llm` ngay sau khi Ollama lên có thể fail vì timeout: chạy một `cxr parse` trước. Trước `cxr eval-parser`: `curl -s localhost:11434/api/version`; không trả lời thì chạy `ollama serve` hoặc mở app |
| T17 | **3B với prompt thô thì hỏng; với few-shot thì chép mẫu.** Probe thiết kế (spec §1): điền `views` cả AP lẫn PA, gắn `UNCERTAIN` cho finding thường, câu VI sai gần hết. Prompt 5 ví dụ ban đầu: dev status F1 0,606. Khi thêm ví dụ, model gắn concept với list nó xuất hiện trong ví dụ ("không có xẹp phổi" → exclude làm "Which studies have atelectasis?" cũng thành exclude), chọn nhãn theo chữ cái đầu (pacemaker → Pneumothorax), và hai ví dụ có view làm 5/12 câu tự viết không nêu view bị đoán ra view (một lần chạy thăm dò, output không lưu; script ở `artifacts/metrics/views_probe.py`; theo T18, chạy lại sẽ không ra đúng số này). Prompt cuối làm hỏng câu use case 6, trong khi prompt gốc qua được: include ra Edema trước Cardiomegaly và `language` ra `en`, nên `test_use_case_six_through_the_llm` FAIL 4/4 lần. Tập concept vẫn đúng; chỉ `semantic_query` bị đảo thứ tự. | Few-shot trong `query_parser_v1.txt` là phần chịu lực, không phải trang trí. Sửa prompt thì đo lại trên `dev` **và** trên bộ câu tự viết trước khi tin. Không thêm ví dụ có view; cân để mỗi concept không chỉ xuất hiện ở một list. Không nới assertion của live test: từ 2026-10-15 test tách đôi, phần ràng buộc (tập concept, status, exclude, views) phải pass, phần thứ tự include, `language` và `semantic_query` mang `xfail(strict=False)`. Có sửa prompt hay không là P7 |
| T18 | **LLM không tái lập giữa các lần chạy.** Cùng prompt (`c975e4ca…`), cùng 40 câu dev, temperature 0, seed 20260101, Ollama 0.40.3 trên GPU: vòng k=4 và lần chạy dev cuối khác nhau ở **5/40** dự đoán (en-020, vi-002, vi-008, vi-016, vi-020); view F1 0,900 → 0,833, status 0,841 → 0,827. Trong một tiến trình, `llm` và `auto` có lúc cũng lệch (k=1: view 0,800 so với 0,667). | Coi mỗi số LLM từ một lần chạy là có nhiễu cỡ 1–2 câu / 40. Phase 7 chạy LLM parser ≥ 3 lần và báo cáo dao động. So hai prompt bằng nhiều câu hơn một lần chạy dev |
| T19 | **`cxr bench-vector --api-url` (`bench_http`) giờ đo cả parse lẫn search.** Từ Phase 4, `/search` chạy `QueryParser` trước khi tìm: lần gọi đầu của mỗi câu gồm cả LLM (≈0,8 s), các lần lặp sau trúng cache parse. HTTP p50/p95 150/158 ms của Phase 3 vì vậy không còn so được với lần đo mới. | Phase 5: đo lại con số vector-only với `LLM_DISABLED=true` |
| T20 | **Tiếng Việt không tách từ; dạng trần "phù" (Edema) khớp trong "phù hợp"** ("consistent with"). "tìm ca phù hợp viêm phổi" parse thành Edema + Pneumonia PRESENT (reviewer cuối thăm dò bằng `RuleParser.parse`, 2026-10-15). Spec §4.3 cho rằng khớp ở mức token tránh được chuyện này, nhưng "phù hợp" là hai token. Không câu benchmark nào chứa "phù hợp". | Phase 5: chặn hoặc bỏ dạng trần "phù". Đã ghi vào giới hạn trong `docs/parser-evaluation.md` |

T4 (`make setup` thiếu Pillow) đã xử lý 2026-09-22 — `setup` giờ dùng
`uv sync --all-extras`. Số được giữ nguyên, không đánh lại, để tham chiếu cũ không lệch.

---

## Đang chờ quyết định

> Đã triển khai theo một hướng, nhưng chưa được người dùng xác nhận. Đừng coi là đã chốt.

| # | Câu hỏi | Đang làm theo | Vì sao còn mở |
|---|---|---|---|
| P4 | Có xem lại **giao thức chấm của gate** Phase 2 không (một prompt dương `"chest x-ray showing {}"`, spec Q3, khớp phép vector search lúc chạy)? | Giữ nguyên; D11 (BiomedCLIP) có hiệu lực tới khi người dùng quyết | Một lần chấm lại tạm thời, ngoài gate, trên embedding đã cache của CheXzero bằng cặp prompt `"{c}"`/`"no {c}"` cho điểm cao hơn. Lần đó khác cả giao thức lẫn cách viết concept, không có trong `model_spike.json`, script không giữ lại. Gợi ý: tiền xử lý CheXzero nhiều khả năng không hỏng, và giao thức một prompt dương có thể bất lợi cho nó. → `docs/model-decision.md` |
| P5 | API nên **đóng gói trọng số vào image** hay dựa vào volume `hf_cache`? | Dựa vào volume `hf_cache` (lần khởi động đầu tải ≈785 MB, sau đó dùng lại) | Đóng gói làm image to hơn nhưng khởi động đầu không cần mạng và không vướng healthcheck; chọn ở Phase 6 khi build image thật. → T15 |
| P6 | Có pull **`qwen2.5:7b-instruct-q4_K_M`** (4,7 GB) làm nhánh chất lượng cho parser ablation không? | Không pull; mặc định vẫn 3B (D5) | Heldout 3B **FAIL** (concept / status / view 0,880 / 0,718 / 0,890), và rule parser hơn 3B ở status (0,862) lẫn view (0,945). 7B là bước kế tiếp đã khai báo trước (spec §12) nếu luận văn cần nhánh so sánh hoặc muốn thử vượt ngưỡng. Latency và VRAM của 7B trên máy này chưa đo. → `docs/parser-evaluation.md` |
| P7 | Có sửa prompt cho câu use case 6 không? Prompt `c975e4ca` trả include Edema trước Cardiomegaly và `language` `en`, nên live test của câu này đang `xfail` | Giữ prompt hiện tại; phần thứ tự/`language` của live test mang `xfail(strict=False)`, phần ràng buộc phải pass | Sửa thì phải đo lại dev và khai báo một lần chạy heldout thứ hai (số heldout hiện tại gắn với `c975e4ca`). Giữ thì `semantic_query` của câu nhiều concept có thể đảo thứ tự; tập ràng buộc không đổi, nên mode `label` không bị ảnh hưởng. Người dùng quyết. → T17 |
| P8 | Người dùng duyệt `benchmarks/parser_queries_v1.jsonl` có đổi gold heldout không? | Benchmark giữ nguyên như lúc đo (sha256 `8d19c619…`); người dùng duyệt bây giờ | Spec §7/§12 yêu cầu người dùng duyệt file trước khi đo; kế hoạch bỏ cổng đó, và lần chạy heldout duy nhất (2026-10-14) diễn ra trước mọi lần duyệt. Đổi bất kỳ gold heldout nào thì phải khai báo một lần chạy heldout thứ hai (cùng quy tắc với P7). → `docs/parser-evaluation.md` |
| P9 | Câu chỉ có exclude ("no pleural effusion"): vector leg encode gì? | Hai parser đang khác nhau: rule parser encode nguyên câu tiếng Anh (warning `no positive finding; vector leg encodes the raw text`), LLM trả `findings` rỗng nên vector leg bị bỏ qua | 28/40 câu negation heldout chỉ có exclude, nên lựa chọn này quyết định baseline vector của nhóm phủ định. Phải chốt một hành vi trước khi hợp nhất ở Phase 5 và đánh giá ở Phase 7 |

P1 (disk guard) và P3 (`docker/qdrant`) chốt 2026-09-22 → D8. P2 (`absence_policy`) chốt
2026-09-23 bằng số liệu corpus → D9.

---

## Nhật ký

> Mới nhất lên đầu. Mỗi entry ≤ 10 dòng và trả lời **vì sao**, không kể lại **cái gì**
> (`git log` đã kể rồi).

### 2026-10-15 — Sửa sau review cuối Phase 4

Review toàn nhánh tìm ra những chỗ sai mà không test nào đỏ. `build_label_matrix` chép theo vị trí
sau một `join` mà polars không hứa giữ thứ tự: nay đặt theo `study_id`. Kết quả fallback vào cache
nên một lần Ollama lỗi thoáng qua thành vĩnh viễn cho câu đó: nay không cache. Ví dụ của chính prompt
đưa "rule out X" vào exclude còn rule parser include X: thêm cue `rule out` sau khi đo; không câu
benchmark nào chứa "rule out" nên số heldout của rule không đổi, `concepts.yaml` sha256 giờ là
`c2f29fc8…`, prompt render vẫn `c975e4ca…` (cue không render vào prompt). Ngưỡng schema-valid 0,99
vô nghĩa trên `auto` (fallback làm nó 1,0 theo construction); Phase 7 kiểm `llm` hoặc
`1 − fallback_rate`. Cổng người dùng duyệt benchmark trước khi đo đã bị bỏ qua: chờ người dùng (P8).
Hoãn sang Phase 5: P9 (câu chỉ có exclude), T19 (bench HTTP), T20 ("phù" trần).

### 2026-10-15 — Phase 4: query parser và label mode

Schema gọn (D17) vì policy là lựa chọn của người dùng; bắt 3B đoán chúng chỉ thêm lỗi. Mode label
không xếp hạng (D18): nhãn không có tín hiệu xếp hạng, và `rank_in_manifest` đẩy 200 ca `valid/` lên
đầu. 4 vòng dev sửa prompt (5 → 20 ví dụ) và lexicon (`big heart`, `normal chest`): status `llm`
0,606 (`parser_dev_0`) → 0,841 (`parser_dev_4`), heldout 0,718. So với `parser_eval_dev.json`
concept −0,03, view +0,06; view nằm trong nhiễu T18 (so với `parser_dev_4` chỉ −0,01). Kết quả:
**FAIL cả ba macro-F1**, ngưỡng giữ nguyên, bước kế tiếp là P6. Bất ngờ: rule parser hơn 3B ở status
và view; 3B chép mẫu ví dụ (T17) và không tái lập (T18); cold start 5,3 s chứ không phải 72 s (T16);
lỗ "sternotomy" của labeler dưới G1 chỉ 28/80 (`CHEXPERT_V1` riêng 73/80). Live test use case 6
`xfail` với prompt cuối (T17, P7). Câu smoke: mode vector vi phạm phủ định 6/10, mode label 0 theo
construction. `compose.yaml` nâng `ollama/ollama` lên 0.40.3, profile `local-llm` chưa chạy thử.

### 2026-10-11 — Phạm vi `config_hash` (D16)

Phase 0a hash cả 4 config vì lúc đó chưa có artifact nào để biết cái gì thực sự quyết định nội
dung index. Giờ đã rõ: vector chỉ phụ thuộc corpus và image encoder. Giữ hash 4 file thì mỗi lần
tune `retrieval.yaml` ở Phase 4–5 sẽ làm `validate-indexes` báo lệch dù Qdrant không đổi, và một
kiểm tra luôn đỏ là kiểm tra đã bị tắt. Scope được ghi vào manifest để kiểm tra tự mô tả chính nó.
Index dựng lại bằng `--recreate` (shard giữ nguyên) cho manifest mang scope mới; validate 6/6.

### 2026-10-11 — Phase 3: vector baseline

Tách embed khỏi index (D14) trả công ngay lần đầu: đổi thiết bị embed từ CPU sang GPU giữa chừng mà
Qdrant, index và API không phải đụng tới. Phần CPU dở được để riêng cho cả build cùng một thiết
bị; đo lại sau đó thấy đúng là vector hai thiết bị lệch tới 4,6e-4 (cosine vẫn ≥ 0,99999).
Macro 25k (0,599, 14 concept) thấp hơn spike (0,636, 11 concept) chủ yếu vì khác tập: ba concept
mới đủ mẫu đều yếu. Cùng 11 concept là 0,622, trong CI spike. Fracture 0,642 → 0,520 cho thấy số
từng concept của spike nhiễu, macro mới đáng tin.
Bất ngờ: `uv run` không trả venv về torch CPU (T14). Qdrant chưa dựng HNSW ở 25k, nên vector
search hiện là chính xác. Docker báo volume 714 MB, thực chiếm 95 MiB.

### 2026-10-03 — Phase 2: VLM spike, chọn BiomedCLIP

Luật gate viết và test trước khi có số, để kết quả không kéo được luật theo. Bước quyết định là
tie < 0,02: hiệu BiomedCLIP − XrayCLIP 0,0198 lọt ngưỡng chỉ 0,0002, nên chi phí quyết — XrayCLIP
chạy 512 px, chậm 6,9×. License không được xét; MIT chỉ là lợi ích đi kèm. Kết luận không phụ
thuộc tie-break vì BiomedCLIP cũng nhỉnh hơn về điểm. CI bootstrap (chứa 0) chỉ để báo cáo.

Macro lenient trung bình 14 concept, strict trung bình 11 concept: khác tập, không so trực tiếp,
không tham gia gate. Lenient xếp XrayCLIP trên BiomedCLIP. Cùng 11 concept, lenient là 0,613 /
0,528 / 0,627 (BiomedCLIP / CheXzero / XrayCLIP), tính từ `auroc_lenient` trong JSON.

Bất ngờ: CheXzero 0,461, **dưới** ngẫu nhiên ở 7/11 concept (không phẳng quanh 0,5), trên cùng
11 concept lenient (0,528) cao hơn strict. Một lần chấm lại tạm thời bằng cặp prompt dương/âm cho nó điểm cao hơn,
nhưng khác giao thức và cách viết concept nên không so được với gate. Gate giữ nguyên; câu hỏi có
xem lại giao thức chấm hay không để người dùng quyết → P4.

### 2026-09-23 — Phase 1: canonical corpus

`labels.py`, `join.py`, `sample.py`, `canonical.py` và `cxr build-corpus`. TDD
toàn bộ, 208 test.

**Phát hiện đáng giá nhất không đến từ test mà từ việc nhìn số liệu.** Corpus lần
đầu build sạch, mọi contract pass — nhưng `No Finding PRESENT = 78,5%` trong khi
CheXpert thật khoảng 9%. Đào ra T11. Nếu bỏ qua, phần đánh giá phủ định — đóng
góp chính của đồ án — sẽ dựa trên nhãn bịa mà không test nào đỏ.

Sửa một sai sót tôi đã lan ra 5 file: "234 study có nhãn bác sĩ" thực ra là 234
**ảnh** trên 200 study. Nó từng được ghi vào WORKLOG như sự thật đã đo mà không
kèm cách đo — đúng loại lỗi mục đó sinh ra để ngăn.

Phân tầng chạy trên toàn pool 187.537 study trước khi chọn, nếu không corpus sẽ
lệch về thứ tình cờ sort lên đầu. Thành viên thứ *j* của tầng cỡ *n* đặt tại
`(j+0.5)/n` nên mọi prefix giữ phân phối nhãn — đó là thứ cho phép `--limit` đổi
cỡ corpus mà không phải sample lại.

Một test bắt lỗi thiết kế thật: tôi chia quy tắc "impression phải có nội dung"
cho hai hàm; sửa implementation để `filter_eligible` tự chứa quy tắc.

### 2026-09-22 — Dọn trước Phase 1

`make setup` đổi sang `uv sync --all-extras`: các lệnh corpus chạy **native** chứ
không trong Docker (chỉ `embed` trong Docker), nên venv dev chính là worker và
phải có Pillow. Dùng `--all-extras` thay vì liệt kê từng extra để không lệch khi
thêm extra mới.

Gom output audit về `artifacts/audit/data_audit.json` trước khi Phase 1 bắt đầu
ghi `corpus_report.json` cạnh nó — tránh hai quy ước đường dẫn tồn tại song song
rồi có người "sửa cho thống nhất" sau.

Chốt P1: bỏ hẳn disk guard (→ `docs/adr/0003`). Chốt P3: xóa `docker/qdrant/`,
scaffolding chết giống `docker/neo4j/`.

Đo trước 3 thứ Phase 1 phụ thuộc, xem *Chi phí của Phase 1*. Quan trọng nhất:
polars parse đúng CSV có newline nhúng — nếu sai thì Phase 1 sẽ sinh rác âm thầm.

### 2026-09-22 — Phase 0a: ba module nền

Chưa commit. `src/common/ids.py` (57 dòng, 23 test), `src/data/contracts.py`
(145 dòng, 31 test), `src/common/config.py` (87 dòng, 17 test). TDD: test viết
trước, chạy thấy đỏ, rồi mới implement.

Làm ba module này trước vì Phase 1 không chạy được nếu thiếu chúng, và vì chúng
chốt T1 một lần cho cả pipeline. Kiểm chứng thêm trên CSV thật: hai cách viết path
cho ra cùng `image_id` và cùng `point_id`, path chuẩn hóa resolve đúng file.

`config_hash` hash **nội dung đã parse**, không hash file — sửa comment không làm
index đang đúng bị coi là stale. Chỉ phủ 4 config ảnh hưởng build
(`data`, `models`, `retrieval`, `concepts`); `logging`/`evaluation` không thể đổi
thứ nằm trong Qdrant.

Thêm 3 test chống trôi lệch code ↔ config: `ids.ROOT_PREFIX` ↔ `data.yaml`,
`contracts.LABEL_STATUSES` ↔ `data.yaml`, mọi field `Settings` ↔ `.env.example`.

### 2026-09-22 — Thu gọn về phạm vi V1 (`a98669d`)

Bỏ Knowledge Graph / Neo4j / RadGraph / ontology y khoa khỏi configs, compose,
deps, docs. Không một dòng Python nào phải xóa — toàn bộ phần ngoài phạm vi nằm ở
config và scaffolding.

Nhân tiện sửa 5 defect làm stack không chạy được: console script trỏ sai
(`main:app` trong khi chỉ có `main()`) khiến **mọi** lệnh worker/api hỏng; `api`
mount named volume còn `worker` mount bind nên API không bao giờ thấy index;
healthcheck gọi `/health` thay vì `/api/v1/health` khiến `api` vĩnh viễn unhealthy
và `web` không lên được; `worker.Dockerfile` hardcode cu124 nên đường CPU bắt buộc
không build được; `HOST_IMAGE_DIR` trỏ thư mục không bao giờ tồn tại.

`configs/ontology.yaml` (160 dòng UMLS/RadLex/anatomy) → `configs/concepts.yaml`
(14 nhãn × surface form EN+VI). Chuyển pip → uv. Kế hoạch V1 vào `docs/`.

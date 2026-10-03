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

**Cập nhật lần cuối:** 2026-10-03 (sau Phase 2)

| | |
|---|---|
| Đã xong | Part A · Phase 0a · Phase 1 (canonical corpus 25.000 study) · **Phase 2 (model gate: BiomedCLIP)** |
| Đang làm | — |
| Tiếp theo | **Phase 3 (embed 25k bằng BiomedCLIP)**, hoặc Phase 0b song song |
| Chặn Phase 3 | không có; model đã pin trong `configs/models.yaml` (D11). P4 đang mở nhưng không chặn |
| Chặn Phase 0b | `pnpm` chưa cài trên máy — xem T5 |

Mã thật hiện có: `src/common/{ids,config}.py`,
`src/data/{contracts,labels,join,sample,canonical,audit}.py`,
`src/embeddings/{base,spike,biomedclip,chexzero,xrayclip}.py`,
`src/evaluation/zero_shot.py`, `src/cli/main.py`.
`src/{api,query,retrieval,explanation}/` còn rỗng.

Dữ liệu đã dựng: `data/canonical/*.parquet` (5 bảng, 11 MB) và
`artifacts/audit/corpus_report.json`. Cả hai gitignored — dựng lại bằng
`uv run cxr build-corpus --limit 25000` (~4 phút). Spike Phase 2:
`data/cache/model_spike/` + `artifacts/metrics/model_spike.json`, cũng gitignored.

Lệnh kiểm tra sau mỗi thay đổi:

```bash
uv run pytest -q                 # 269 tests (3 test `model` bị bỏ chọn mặc định)
uv run pytest -m model           # 3 smoke test adapter thật, cần weights đã tải
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

T4 (`make setup` thiếu Pillow) đã xử lý 2026-09-22 — `setup` giờ dùng
`uv sync --all-extras`. Số được giữ nguyên, không đánh lại, để tham chiếu cũ không lệch.

---

## Đang chờ quyết định

> Đã triển khai theo một hướng, nhưng chưa được người dùng xác nhận. Đừng coi là đã chốt.

| # | Câu hỏi | Đang làm theo | Vì sao còn mở |
|---|---|---|---|
| P4 | Có xem lại **giao thức chấm của gate** Phase 2 không (một prompt dương `"chest x-ray showing {}"`, spec Q3, khớp phép vector search lúc chạy)? | Giữ nguyên; D11 (BiomedCLIP) có hiệu lực tới khi người dùng quyết | Một lần chấm lại tạm thời, ngoài gate, trên embedding đã cache của CheXzero bằng cặp prompt `"{c}"`/`"no {c}"` cho điểm cao hơn. Lần đó khác cả giao thức lẫn cách viết concept, không có trong `model_spike.json`, script không giữ lại. Gợi ý: tiền xử lý CheXzero nhiều khả năng không hỏng, và giao thức một prompt dương có thể bất lợi cho nó. → `docs/model-decision.md` |

P1 (disk guard) và P3 (`docker/qdrant`) chốt 2026-09-22 → D8. P2 (`absence_policy`) chốt
2026-09-23 bằng số liệu corpus → D9.

---

## Nhật ký

> Mới nhất lên đầu. Mỗi entry ≤ 10 dòng và trả lời **vì sao**, không kể lại **cái gì**
> (`git log` đã kể rồi).

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

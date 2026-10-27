# WORKLOG

Ngữ cảnh dự án cho coding agent: đang ở đâu, quyết định nào còn hiệu lực, chỗ nào có bẫy. Đọc hết trước khi sửa gì.

| Cần biết | Đọc ở |
|---|---|
| Phạm vi đề tài | `docs/scope-and-non-goals.md` |
| Thiết kế KG (V2), kế hoạch và bảng trạng thái | `docs/v2-kg-design.md`, `docs/v2-kg-plan.md` |
| Số đo kèm cách đo | `docs/measured-facts.md` |
| Cách chạy | `README.md` |
| Thay đổi gì, lúc nào; WORKLOG cũ (trước 2026-10-25) | `git log`, `git log -p WORKLOG.md` |

Giữ file này gọn (CLAUDE.md §0): tối đa 150 dòng; sửa tại chỗ thay vì nối thêm; quyết định, bẫy, câu hỏi hết hiệu lực
thì xóa (git giữ lịch sử); không có nhật ký; mã D/T/P không đánh lại.

---

## Trạng thái hiện tại

**Cập nhật:** 2026-10-27, trên nhánh `feat/v2-knowledge-graph` (tách từ `main` `1b50c19`).

- **Hệ thống đang chạy:** câu hỏi EN/VI → parser (gpt-5-mini, rule dự phòng) → câu mẫu `render(include, exclude)` →
  text encoder CXR-CLIP → cosine trên Qdrant, chỉ lọc view → top-k → trích câu report sau xếp hạng (D38–D40). Dev
  nDCG@10 nhị phân 0,482. Đây là baseline B1 của V2.
- **Scope:** KG theo thiết kế V2 `docs/v2-kg-design.md` (D41, D45–D50). Fine-tune tạm dừng (D42); checkpoint FT6 và
  lệnh `cxr build-image-scores` chỉ có trên nhánh `feat/label-aware-image-retrieval`, chưa có trên `main`.
- **Đang làm:** V2-2 (`notes/tip-v2-2.md`): phần đo xong, cổng PASS (Observation phủ 90% lượt nhắc bằng 265 chuỗi,
  ngưỡng 500); kết quả ở `data/kg/vocab/`, số đo ở `docs/measured-facts.md`. Chờ người dùng duyệt
  `data/kg/vocab/chexpert_candidates.csv` để ghi `configs/kg/chexpert_cuis.yaml`. V2-1 xong: UMLS đã lọc vào
  `data/umls/2026AA/`. TIP V2-5, V2-5b đã duyệt. TIP và ghi chú đo nằm ở `notes/` (gitignored, chỉ có trên máy). Bảng
  trạng thái và các task sau: `docs/v2-kg-plan.md`.
- **Chặn:** web UI chờ `pnpm` (T5).

Dữ liệu, chỉ mục, artifact đều gitignored; lệnh dựng lại ở README. README không nói:

- Mỗi build có `data/indexes/builds/<build_id>/manifest.json`; API phục vụ `BUILD_ID`, không đặt thì build duy nhất
  của encoder `selected`. Hai build: `20261023_chexpert25k_cxrclip_v1` (mặc định) và `20261011_chexpert25k_biomedclip_v1`.
- `cxr eval-parser` ghi đè `parser_eval.json`, `cxr eval-retrieval` ghi đè `retrieval_eval.json`; các bản heldout được
  trích dẫn đã đóng băng trong `artifacts/metrics/` (tên và sha256 ở `docs/measured-facts.md`).
- Script thăm dò chỉ đọc nằm ở `artifacts/metrics/*.py` (gitignored); mỗi số đo ghi script của nó.

Kiểm tra sau mỗi thay đổi:

```bash
uv run pytest -q                 # 897 passed; model/integration/llm bị bỏ chọn mặc định
uv run pytest -m model           # bắt buộc khi đổi configs/models.yaml hoặc src/embeddings/ (T37)
uv run pytest -m integration     # cần `docker compose up -d qdrant`
uv run pytest -m llm             # parse thật qua LLM, đọc .env
uv run cxr validate-indexes      # 7 x PASS; `--encoder biomedclip` cũng vậy
uv run ruff check src tests
```

---

## Quyết định đang có hiệu lực

Kế hoạch `docs/ke-hoach-du-an-nho-cxr-semantic-retrieval-v1.md` viết trước khi đo dữ liệu và trước khi đổi scope; chỗ
nào lệch thì bảng này và `docs/scope-and-non-goals.md` đúng.

| # | Quyết định | Vì sao |
|---|---|---|
| D41 | Đề tài có KG tri thức y khoa (giải phẫu, bệnh lý, dấu hiệu hình ảnh) làm trung gian giữa câu hỏi và ảnh; LLM phân tích câu hỏi, ánh xạ vào KG và diễn giải kết quả (RAG). Nguồn: UMLS, RadGraph. Triệu chứng lâm sàng nằm ngoài phạm vi (`docs/scope-and-non-goals.md`) | Người dùng xác nhận 2026-10-25, bám sát mô tả đề tài (thay D28); bỏ triệu chứng ngày 2026-10-27 |
| D42 | Fine-tune CXR-CLIP tạm dừng, xây KG trước | Người dùng 2026-10-25. Điểm yếu phủ định của encoder zero-shot còn nguyên |
| D45 | **Report không vào đường truy vấn.** Report chỉ dùng offline: tính trọng số KG và đánh giá. Hệ thống nhắm tới ảnh chưa có report | Thiết kế V2, quyết định nền tảng 1; thay D43. Câu trích report để hiển thị (D25) là câu hỏi mở của V2 |
| D46 | **KG chỉ chứa tri thức:** nút khái niệm (`Disease`, `Finding`, `Anatomy`, `Device`; ID là CUI) và tên gọi. Không có tầng dữ kiện theo study (`Study`, `Section`, `Fact`) | Thiết kế V2, quyết định nền tảng 2; thay D44 |
| D47 | **Ảnh nối vào KG qua Finding rồi tới Disease.** Điểm CXR-CLIP đã hiệu chỉnh là cạnh `PREDICTED` (ảnh → Finding); Finding → Disease bằng `SUGGESTS`. KG chọn khái niệm và trọng số, CXR-CLIP chấm từng ảnh (ma trận điểm tính sẵn); LLM không quyết định thứ hạng | Thiết kế V2, quyết định nền tảng 3. Mọi trọng số KG lấy từ pool, nên nhãn bạc RadGraph trên corpus dùng được để chấm |
| D48 | UMLS 2026AA, Metathesaurus **Level 0 Subset**; SNOMED CT tắt mặc định (`configs/kg.yaml: umls.include_snomed: false`), chỉ bật sau khi khai báo MLDS và tải Full Subset | Nguồn mức 0 (MSH, NCI, FMA, HPO…) không có hạn chế ngoài license chuẩn; SNOMED CT có điều kiện riêng ngoài nước thành viên |
| D49 | Dữ liệu UMLS nằm trong `data/umls/` (gitignored qua `data/*`), không bao giờ commit, không đưa vào Docker image; container chỉ mount lúc chạy | License UMLS cấm phát tán dữ liệu hoặc subset; repo public (T38) |
| D50 | File gốc ở `data/umls/raw/` (zip và `2026AA/META/`, chỉ đọc); Parquet đã lọc và `manifest.json` ở `data/umls/<version>/`, ghi bởi `cxr kg-filter-umls` | Tách bản gốc của NLM khỏi bản đã lọc; manifest ghi MD5 zip, config, sha256 và số dòng trước, sau khi lọc |
| D52 | Chốt của V2-2, ghi trước khi đo: từ vựng chỉ đo trên pool (88.853 study). Ghép modifier hai bước: (a) bỏ theo `configs/kg.yaml: vocab.drop_modifiers` (Observation bỏ mức độ, diễn tiến, vị trí, bên; Anatomy chỉ bỏ bên); (b) modifier còn lại chỉ giữ khi chuỗi ghép khớp chính xác một tên trong MRCONSO (thử cả chuỗi, rồi từng modifier), không thì chỉ giữ từ chính. Đưa về số ít chỉ khi dạng số ít có trong pool hoặc MRCONSO, không đổi đuôi `as`, `ss`, `us`, `is`. Cổng: Observation theo quy tắc trên phủ 90% lượt nhắc bằng ≤ 500 chuỗi | TIP V2-2 người dùng duyệt 2026-10-27 (`notes/tip-v2-2.md`) |
| D51 | Trong đánh giá V2, **B1 dùng CXR-CLIP FT6**, cùng encoder với ma trận điểm của B2; B1 zero-shot giữ làm dòng phụ. Build corpus FT6 được đăng ký ở V2-5b | Người dùng 2026-10-27: hiệu B2 − B1 phải là đóng góp riêng của KG, không lẫn đóng góp của fine-tune |
| D39 | Một đường tìm kiếm; **thứ hạng chỉ từ ảnh, API không đọc nhãn**. Câu mẫu do code dựng (`templates.render`: "no X", No Finding là "normal chest", thứ tự CHEXPERT_LABELS, từ chối "A and no A"); lọc view là lọc metadata; câu report trích sau xếp hạng, chỉ để hiển thị. Không khái niệm → câu tự do của parser; chỉ phủ định → "no X" kèm warning; UNCERTAIN như PRESENT; trên 2 ràng buộc kèm warning | Hybrid thỏa ràng buộc nhờ lọc trên chính nhãn làm đáp án; một đường áp dụng được cho ảnh chưa có report |
| D38 | Encoder mặc định CXR-CLIP SwinT bản M (sha256 ghim trong `models.yaml`, CC BY-NC 4.0); BiomedCLIP vẫn là candidate | Hòa zero-shot trên dev, CXR-CLIP nhanh hơn 2,4 lần (luật hòa của gate Phase 2) |
| D40 | Sổ đăng ký build: manifest riêng mỗi build ghi encoder; `config_hash` chỉ phủ `data.yaml` + candidate của encoder + `normalize`; mọi lệnh encode text nạp encoder qua `index.build_encoder` (từ chối revision khác build) | Hai encoder so được trên cùng pipeline mà không làm hỏng build của nhau |
| D22 | Parser: OpenAI `gpt-5-mini-2025-08-07`, JSON schema theo 14 khái niệm; rule parser khi LLM lỗi hoặc thiếu key. Ollama qwen 3B còn chạy được (`LLM_PROVIDER=ollama`) nhưng trượt heldout | Bake-off trên dev theo luật viết trước; heldout PASS. Truy vấn đi ra OpenAI, không chứa dữ liệu bệnh nhân |
| D20 | Prompt là file có version; sha256 của prompt đã render nằm trong khóa cache và trong JSON đo; chỉ chỉnh prompt và lexicon trên dev | Mọi số đo truy được về prompt |
| D12 | 200 ca nhãn bác sĩ (`valid/`, rank 0–199) không bao giờ dùng để chọn model hay tune | Ground truth người duy nhất |
| D24 | Đánh giá trên benchmark 200 câu EN/VI: dev 40 để tune, heldout 160 chạy một lần (chạy lại phải khai báo); số chính nDCG@10 nhị phân | — |
| D21 | Benchmark theo quy ước cụm từ của CheXpert labeler ("airspace disease" → Lung Opacity, "collapsed lung" → Atelectasis) | Nhãn resolve là ground truth |
| D9 | `absence_policy` mặc định `exclude_present`; khi chấm báo song song `explicit_absence_required` | Median ABSENT chỉ 5%; bắt buộc ABSENT làm recall sụp |
| G1 | Resolve nhãn: `CHEXPERT_V1 > CHEXBERT_IMPRESSION > CHEXBERT_FINDINGS > CHEXBERT_REPORT`, lấy giá trị khác NOT_MENTIONED đầu tiên; dựng trong RAM | Cần đúng một trạng thái cho mỗi (study, khái niệm) |
| D10 | `CHEXBERT_FINDINGS` không phát nhãn cho study thiếu findings section | T11 |
| D25, D27 | Evidence = câu nguyên văn (offset vào `search_text`), polarity bằng rule parser, một câu mỗi khái niệm. Từ vựng report (`concepts.yaml:report`) tách khỏi lexicon parser | Kiểm được bằng substring; lexicon parser không được đổi sau khi đo heldout |
| D17 | `SearchQuery` gồm `semantic_query`, `include`, `exclude`, `views`, `language`; `top_k` là tham số của request | Lựa chọn của người dùng, không phải nội dung câu |
| D1 | Không chia corpus: cả 25.000 study đều tìm được; hold-out ở mức câu hỏi | Không tham số nào học từ corpus |
| D14 | Embed hai bước qua shard file (`cxr embed` → `cxr index`), resume theo shard; mỗi build embed trên một thiết bị, mặc định CPU | Embed đắt, Qdrant hay đổi |
| D2–D8 | Không có Tier A/B; package phẳng `src/`; giữ `compose.gpu.yaml`; không hardcode số chiều embedding; Qdrant point id = `uuid5(NAMESPACE_URL, normalized_path)`; không có disk guard | `docs/adr/0002`, `docs/adr/0003` |

---

## Bẫy đã biết

| # | Bẫy | Cách xử lý |
|---|---|---|
| T1 | `train.csv`/`valid.csv` ghi path có prefix `CheXpert-v1.0-small/`, `path_to_image` của Plus thì không | Luôn qua `src/common/ids.py:normalize_image_path()` |
| T2 | Plus CSV có newline nhúng (≈ 48 dòng vật lý mỗi record) | Dùng CSV parser thật (polars); đọc theo dòng sai âm thầm |
| T3 | `labels/*_fixed.json` là JSONL, path bên trong không có prefix | Đọc từng dòng; chuẩn hóa qua `ids.py` |
| T5 | `pnpm` chưa cài | `corepack enable`; chỉ chặn web |
| T6, T9 | `findings` chỉ có ở 26,6% dòng; impression hay mở đầu bằng newline và đánh số `1. 2.` | Không bắt buộc findings; `.strip()` trước khi kiểm rỗng |
| T7 | `NOT_MENTIONED` không bao giờ là `ABSENT` | `contracts.py` chặn ở schema; KG cũng vậy (không nhắc thì không có cạnh) |
| T8 | Console Windows là cp1252; in tiếng Việt từ script thì lỗi | In ASCII hoặc ghi file UTF-8 |
| T10 | `grep -r` ở gốc repo quét cả `data/` (12 GB) và treo | `git grep` hoặc `--exclude-dir=data` |
| T11 | CheXbert chạy trên findings rỗng luôn ra `No Finding = 1` | `canonical._read_all_labels` loại các dòng đó; nguồn nhãn mới nào cũng phải kiểm kiểu này |
| T13 | Qdrant khi chạy native: dùng `client.query_points`; server ghim `v1.19.2`; `np.savez` tự nối `.npz`; Docker Desktop phải chạy trước `cxr index` và API; `.env` của `make setup` chỉ đúng trong container | README liệt kê biến cần ghi đè khi chạy native |
| T14 | `uv run` không gỡ torch CUDA cài bằng `uv pip` | Sau khi dùng GPU: `uv sync --all-extras`, kiểm `torch.__version__` có `+cpu` |
| T15 | Image API Docker chưa từng được build hay chạy | Phải chạy `docker compose build api` và `make up` thật trước khi tin |
| T18, T23, T25 | LLM không tái lập giữa các lần chạy; mỗi họ model OpenAI nhận tham số khác nhau (GPT-5 dùng `max_completion_tokens`, `gpt-5-mini` không nhận `temperature: 0`); 40 câu dev không tách được các model mạnh | Coi mỗi số LLM có nhiễu 1–2 câu trên 40; chạy ≥ 3 lần khi báo cáo; tham số theo model ở `models.yaml:llm.hosted_request` |
| T20, T24 | Tiếng Việt không tách từ: "phù" khớp trong "phù hợp" (đã chặn bằng `blocked_phrases`); "tư thế trước-sau", "vôi hóa màng phổi" chưa có trong lexicon | Thêm dạng vào `concepts.yaml` chỉ trên dev |
| T26 | `search_text` xuống dòng giữa câu và đánh số `1. 2.` | `evidence.sentences` chỉ tách theo dấu câu; offset vào `search_text` nguyên văn |
| T27, T28 | Một connection SQLite read-only dùng chung giữa các thread; dựng lại report index khi file đang mở thì `PermissionError` trên Windows | Mọi truy cập qua `state.report_lock`; dừng API trước khi `cxr build-report-index` |
| T29 | RadGraph-XL không có khóa ảnh và thứ tự bị xáo; nối theo vị trí ra 0/2.000 | Nối theo text chuẩn hóa `re.sub(r"[^a-z0-9]", "", text.lower())` của đúng section: khớp 100% (`artifacts/metrics/radgraph_align_probe.py`) |
| T36 | `data/` dùng chung giữa các nhánh; `data/indexes/build_manifest*.json` là của nhánh `feat/label-aware-image-retrieval`. Ngày 2026-10-27 đã xóa phần dữ liệu riêng của nhánh đó: `training/.venv`, `data/colab_bundle`, `data/train_cache`, `data/b16_rehearsal`, `data/exp_cxrclip` (giữ `runs/` và các embedding) | Lưu sha256 manifest trước khi index, kiểm lại sau. Quay lại nhánh V1.1 thì phải dựng lại cache train và bundle Colab |
| T37 | Model test bị bỏ chọn mặc định, nên `pytest -q` xanh vẫn lọt lỗi adapter | Chạy `-m model` khi đổi `models.yaml` hoặc `src/embeddings/` |
| T38 | Repo GitHub public; UMLS cấp license cá nhân, cấm phát tán tập con, phải báo cáo sử dụng hằng năm (tháng 1) | Mọi thứ trích từ UMLS nằm dưới `data/` (gitignored); repo chỉ chứa script và CUI |
| T39 | MRREL không đồng nhất giữa các nguồn: phân cấp MSH là `PAR`/`CHD` với RELA **rỗng**, còn NCI, FMA, HPO dùng `isa`/`inverse_isa`; bộ phận trong FMA nằm ở `regional_part_of` và `constitutional_part_of`, RELA `part_of` chỉ có 23 dòng. REL/RELA mô tả quan hệ của CUI2 đối với CUI1 | Lọc `IS_A`, `PART_OF` theo cả REL lẫn RELA của từng nguồn; kiểm chiều trên một cặp đã biết trước khi tin (`docs/measured-facts.md`, *UMLS 2026AA*) |

---

## Đang chờ quyết định

| # | Câu hỏi | Đang làm theo |
|---|---|---|
| P5 | Đóng gói trọng số vào image API hay dựa vào volume `hf_cache` | Volume; chọn khi build image thật |
| P8 | Người dùng duyệt gold heldout của `benchmarks/parser_queries_v1.jsonl` | Giữ như lúc đo (sha256 `8d19c619…`); đổi gold heldout thì khai báo một lần chạy mới |

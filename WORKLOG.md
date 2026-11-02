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

**Cập nhật:** 2026-11-02, trên nhánh `feat/v2-knowledge-graph` (tách từ `main` `1b50c19`).

- **Hệ thống đang chạy:** câu hỏi EN/VI → parser (`claude-haiku-5-5`, rule dự phòng; D61) → câu mẫu `render(include, exclude)` →
  text encoder CXR-CLIP FT6 → cosine trên Qdrant, chỉ lọc view → top-k → trích câu report sau xếp hạng (D38–D40,
  D54). Dev nDCG@10 nhị phân 0,539 (parser gold); zero-shot 0,482 là dòng phụ. Đây là baseline B1 của V2.
- **Scope:** KG theo thiết kế V2 `docs/v2-kg-design.md` (D41, D45–D50). Fine-tune tạm dừng (D42); checkpoint FT6
  (candidate `cxrclip_ft6_lr1e-04`) và `cxr build-image-scores` đã mang về nhánh này ở V2-5; code fine-tune vẫn chỉ ở
  `feat/label-aware-image-retrieval`.
- **Đang làm:** V2-1 tới V2-5b xong. V2-3 (linker, D56): 95% đầu từ vựng pool nối CUI, cổng đạt theo chuẩn bạc
  (precision 0,985, độ phủ 0,949). V2-4 xong (D57–D60): bảng KG ở `data/kg/graph/` (`cxr kg-build-graph`), cổng nối
  về `scorable` nhận ở 0,899; 6 cạnh `SUGGESTS` mới Claude bên chat đã giữ. Parser đã chuyển sang Anthropic (D61), heldout PASS.
  Ma trận điểm mở rộng 23 cột (D62). Tiếp theo: V2-6. Ghi chú: V2-4 ở `notes/v2-vocab.md`, `notes/v2-graph.md`; V2-6, V2-9 ở
  `notes/v2-scores.md`; V2-6, V2-7 ở `notes/v2-eval.md`. TIP và ghi chú đo nằm ở `notes/` (gitignored, chỉ có trên
  máy). Bảng trạng thái và các task sau: `docs/v2-kg-plan.md`.
- **Chặn:** web UI chờ `pnpm` (T5).

Dữ liệu, chỉ mục, artifact đều gitignored; lệnh dựng lại ở README. README không nói:

- Mỗi build có `data/indexes/builds/<build_id>/manifest.json`; API phục vụ `BUILD_ID`, không đặt thì build duy nhất
  của encoder `selected`. Ba build: `20261024_chexpert25k_cxrclip_ft6_lr1e-04_v1` (mặc định),
  `20261023_chexpert25k_cxrclip_v1` (zero-shot) và `20261011_chexpert25k_biomedclip_v1`.
- `cxr eval-parser` ghi đè `parser_eval.json`, `cxr eval-retrieval` ghi đè `retrieval_eval.json`; các bản heldout được
  trích dẫn đã đóng băng trong `artifacts/metrics/` (tên và sha256 ở `docs/measured-facts.md`).
- Script thăm dò chỉ đọc nằm ở `artifacts/metrics/*.py` (gitignored); mỗi số đo ghi script của nó.
- Ma trận điểm V2 nằm ở `data/indexes/image_scores/<build_id>/` (`cxr build-image-scores`). Ba file nằm thẳng trong
  `data/indexes/image_scores/` là ma trận zero-shot cũ của nhánh V1.1 (câu mẫu cũ), không dùng cho V2.

Kiểm tra sau mỗi thay đổi:

```bash
uv run pytest -q                 # 959 passed; model/integration/llm bị bỏ chọn mặc định
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
| D53 | 14 nhãn CheXpert → `configs/kg/chexpert_cuis.yaml`, chỉ mã, `check_mapping` kiểm: 10 nhãn là CUI, No Finding là `PRJ:no_finding`. Nhãn mà tên không có trong UMLS và chỉ khớp khái niệm hẹp hơn dùng `PRJ:<tên>`, kèm `is_a` (một CUI cha) và `children` (các CUI hẹp hơn khớp chính xác dạng `en`, nối `IS_A` → PRJ; không lấy CUI thứ hai của cùng dạng): Enlarged Cardiomediastinum, Pleural Other, Support Devices. Cardiomegaly là con của `PRJ:enlarged_cardiomediastinum` (phân cấp CheXpert) | Người dùng chốt 2026-10-27 trên ứng viên V2-2 (xếp theo khớp đúng tên nhãn trước); mã và lý do ở `notes/v2-vocab.md` |
| D56 | Chốt của V2-3 (linker), ghi trước khi đo: liên kết phần đầu phủ 95% lượt nhắc theo D52 (Observation 645, Anatomy 346 chuỗi). Luật theo thứ tự: nhóm normal; override tay `configs/kg/linker_overrides.yaml` (CUI hoặc `skip`, có danh sách từ định tính); 9 từ mô tả đứng một mình; khớp chính xác MRCONSO; tính từ giải phẫu → danh từ; `out_of_scope` (khớp tên MRCONSO nhưng mọi CUI có TUI ngoài bảng gán nhãn; không thêm T073, T082); còn lại qua bước chọn. Bảng gán nhãn thêm **T024 Tissue** vào Anatomy. Mọi ứng viên (exact, SapBERT, cặp) lọc theo miền `linker.domain`, **phương án C (gốc âm)**: bỏ CUI chỉ có tên từ nguồn hệ mã, danh mục sản phẩm, phân loại thống kê (MTH không tính); bỏ CUI dưới gốc âm (Anatomy: đầu, chi dưới, chậu; Observation: bệnh mắt, răng miệng, thần kinh, da, thai, tiết niệu sinh dục, tâm thần, tai) mà không dưới gốc dương nào (lồng ngực, cổ, chi trên, bụng; hô hấp, tim mạch, lồng ngực, gãy xương, cột sống, u); "dưới" đi theo FMA và `CHD` mọi nguồn. Bước chọn, cho mọi chuỗi không do normal, override, `skip`, từ mô tả (quyết qua cặp) hay `out_of_scope` quyết: ứng viên = CUI exact (tính từ thì của danh từ) gộp với CUI của 10 tên SapBERT (commit `090663c3`) gần nhất trong miền, **sàn τ = 0,5** (chốt 2026-11-01; không còn ngưỡng nhận). Hai model Anthropic chạy độc lập, cùng prompt `configs/prompts/kg_link_select_v1.txt` (sha256 vào manifest) và cùng input: `claude-opus-5-5` và `claude-sonnet-5-5` (effort medium), cache `data/kg/link/llm_cache_<opus|sonnet>.jsonl`; không có fallback, model từ chối thì `none` lý do "refused". **Hai model cùng họ nên độc lập yếu hơn hai provider** (không còn credit OpenAI); đồng thuận của chúng không thay được chấm mẫu. Input chỉ gồm chuỗi, loại, lượt nhắc, tối đa 5 chuỗi `located_at` kèm số đếm, ứng viên (CUI, tên, TUI, nguồn, cosine); không gửi text report. Output Pydantic `decision` (cui, skip, out_of_scope, none), `cui`, `reason`; CUI ngoài danh sách thì `rejected`. Đồng thuận thì nhận, lệch thì `none`; `data/kg/link/adjudication.csv` (Claude bên chat ghi) thắng mọi luật. `review.csv` có quyết định và lý do của hai LLM, `agree`, `flags` (nguồn chỉ CHV/MTH, tên có từ ngoài miền, TUI lệch loại, chuỗi `skip` có CUI, `chexpert_mismatch`, bên chưa quy), dòng lệch lên đầu. Chuẩn vàng một phần: mention phrases của `stanfordmlgroup/chexpert-labeler` (MIT, commit `44ddeb36`). CUI mang bên quy về cha không phân bên. Song song, liên kết cặp `located_at` (cặp phủ 90%, tối đa 10, hai thứ tự). Cổng kép: precision ≥ 0,85 trên 200 liên kết LLM đồng thuận chưa phân xử (seed 20260101), bác sĩ chấm hoặc giám khảo `claude-fable-5-1` với prompt riêng (chuẩn bạc, sau cờ `--judge`); mẫu kèm CUI argmax để so "SapBERT argmax" với "SapBERT + LLM"; độ phủ theo lượt nhắc phần đầu 95% ≥ 0,90, chỉ `none` và `pending` là thiếu. Chạy thử tối đa 10 chuỗi mỗi provider rồi dừng trước khi gọi toàn bộ | TIP V2-3 duyệt 2026-10-29; đổi cùng ngày vì chấm sơ bộ cho SapBERT argmax đúng khoảng 1/3. Ngày 2026-11-01 chốt C (gốc dương như spec loại oan lung, rib, consolidation…; `notes/v2-link-roots.md`), bụng thay bụng trên, T024 vì pleura, soft tissue, subcutaneous tissue, costal cartilage, fat là giải phẫu thấy được trên phim ngực và mang T024. Người dùng không phán y khoa: LLM đề xuất, Claude bên chat phân xử |
| D57 | Chốt cho V2-4: (a) khái niệm chung nối `IS_A` (`source: chexpert`) tới nhãn theo mention phrases của CheXpert labeler (D21): nodule, mass, mass lesion, nodular opacity, nodular density, tumor, neoplasm → Lung Lesion; infiltrate, increased density, reticular pattern, scar → Lung Opacity; fibrosis → `PRJ:pleural_other`; infection, infectious process → Pneumonia. (b) chf và heart failure **không** `IS_A` Edema: là Disease, nối `SUGGESTS` sang Edema, Cardiomegaly, Pleural Effusion với trọng số tính trên pool | Người dùng 2026-11-01, sau đối chiếu cụm từ CheXpert ở V2-3 (`notes/v2-vocab.md`) |
| D58 | Chốt của V2-4, ghi trước khi đo (`notes/tip-v2-4.md`): lõi = 14 nhãn (cả PRJ, `is_a`, `children`) + CUI cuối của V2-3 + CUI liên kết cặp. Mở rộng trong miền C: giải phẫu theo FMA `regional_part_of`, `constitutional_part_of`, `isa`, tối đa 3 bước mỗi chiều; finding, bệnh theo `IS_A` mọi nguồn, xuống tối đa 2 bước, lên tối đa 6 bước và dừng ở gốc chặn (gốc dương của miền, Finding, Disease, Sign or Symptom, Abnormality, Neoplasm by site); không thêm khái niệm mang bên. Thiết bị: CUI T074 của lõi khớp mention phrases `support_devices` của CheXpert labeler thì `IS_A` → `PRJ:support_devices`, còn lại `IS_A` → C0025080. Cạnh CheXpert theo thiết kế và D57. `SUGGESTS` (Finding → Disease, kể cả Edema, Cardiomegaly, Pleural Effusion → chf/heart failure; V2-6 đi ngược cạnh) và `TYPICALLY_AT` tính trên cả pool: nguồn present, đích của `suggestive_of` bất kể polarity, lượt nhắc có anatomy dùng liên kết cặp nếu có; `p = n(f → x) / n(f)` theo số report; giữ khi `n ≥ 20`. V2-4 không bỏ cạnh tới khái niệm dưới cổng `quality` (D55 áp lúc truy vấn). LLM: `claude-opus-5-5` duyệt sơ bộ 50 cạnh `SUGGESTS` mạnh nhất và đề xuất alias tiếng Việt (14 nhãn + 200 khái niệm `n_pool` lớn nhất), Claude bên chat phân xử; dưới 5 USD thì chạy. Cổng: ≥ 90% khái niệm lõi nối được về một khái niệm `scorable` | Người dùng duyệt TIP V2-4 ngày 2026-11-02, sửa D (thiết bị) và xác nhận chiều G |
| D60 | Cổng nối về `scorable` của V2-4 **nhận ở 0,899 theo lượt nhắc** (ngưỡng 0,90; theo khái niệm 290/595; sau `IS_A` → Edema của "vascular prominence" C5194648 và "vascular congestion" C0242073, cặp congestion + vascular override về C0242073). Trước lần đo cuối chỉ sửa cho nhất quán với luật đã có, không nhằm qua cổng: "normal limit" vào nhóm normal; abnormality, disease, acute disease → `skip`; 5 cặp lệch phân xử. Không cụm nào trong năm đề xuất nối theo D57 có trong mention phrases của CheXpert labeler (blunting + costophrenic, cardiomediastinal, pulmonary vasculature, aicd/defibrillator, aspiration), nên không nối. Phần còn thiếu chủ yếu là dấu hiệu ngoài 14 cột của model ảnh (lượt nhắc pool): low lung volumes 13.679, aspiration 4.648, degenerative 4.239, soft tissue 3.559, normal heart size 3.085, elevated hemidiaphragm 2.665, device 2.086, aicd 1.669, deformity 1.295, emphysema 1.277, enlargement 1.270, tortuous aorta 1.253, interstitial marking 1.238, osteopenia 1.196, subcutaneous emphysema 1.064, calcification 998. Hướng xử lý là V2-5c (thêm cột cho model ảnh), không phải cạnh tay | Người dùng 2026-11-02 |
| D59 | Cổng nối về `scorable` của V2-4, **đổi định nghĩa sau lần đo đầu** (0,366 theo khái niệm, 0,789 theo lượt nhắc): số chính theo lượt nhắc (`n_pool`), ngưỡng 0,90; vẫn báo theo khái niệm. "Nối được" định nghĩa đệ quy: khái niệm `scorable`; hoặc `IS_A`* tới một khái niệm nối được; hoặc Disease có `SUGGESTS` (đã giữ khi phân xử) từ một khái niệm nối được; hoặc Anatomy mà chính nó hay một phần của nó (`PART_OF`*) là đích `TYPICALLY_AT` của một khái niệm nối được. Cạnh `SUGGESTS` bị bỏ khi phân xử được đánh dấu, không dùng lúc truy vấn | Người dùng 2026-11-02: định nghĩa đầu chỉ nhận một bước từ đúng 14 nhãn, nên finding con và giải phẫu nơi thiết bị nằm không bao giờ tính; trọng số theo lượt nhắc phản ánh từ vựng thật của report |
| D55 | Mỗi khái niệm scorable mang `quality` = AUROC lenient trên pool-val (kèm khoảng tin cậy bootstrap 95%, tính ở V2-6, chỉ để báo cáo); ngưỡng **0,65**, so với ước lượng điểm. Dưới ngưỡng: vẫn chấm khi hỏi trực tiếp hoặc leo `IS_A`, kèm cảnh báo bắt buộc; không làm đích của `SUGGESTS`, `TYPICALLY_AT`, mở rộng "rộng hơn". Theo V2-5, sáu khái niệm dưới ngưỡng: Pleural Other, Enlarged Cardiomediastinum, Lung Lesion, Fracture, Pneumonia, Atelectasis. Giữ đủ 14 cột, không hạ trọng số tay; ablation bật/tắt cổng ở V2-8 | Người dùng chốt 2026-10-29, trước khi đo dev (thay P15); `notes/v2-scores.md` |
| D54 | Luật V2-5b, ghi trước khi đo: API đổi encoder mặc định sang FT6 (`image_encoder.selected: cxrclip_ft6_lr1e-04`) khi nDCG@10 nhị phân trên dev của FT6 ≥ của zero-shot, cả hai đo bằng `eval-retrieval --parsers gold`, cùng code; thấp hơn thì dừng và báo. Điều kiện tái lập: zero-shot phải ra đúng 0,482, lệch thì dừng. FT6 thua thì B1 và B2 cùng dùng zero-shot (D51 đổi theo) và ma trận V2-5 dựng lại bằng zero-shot | TIP V2-5b người dùng duyệt 2026-10-27, bước 4 sửa 2026-10-29: parser gold cho cả hai build, vì parse LLM không tái lập (T18) |
| D51 | Trong đánh giá V2, **B1 dùng CXR-CLIP FT6**, cùng encoder với ma trận điểm của B2; B1 zero-shot giữ làm dòng phụ. Build corpus FT6 `20261024_chexpert25k_cxrclip_ft6_lr1e-04_v1` đã đăng ký (V2-5b) | Người dùng 2026-10-27: hiệu B2 − B1 phải là đóng góp riêng của KG, không lẫn đóng góp của fine-tune |
| D39 | Một đường tìm kiếm; **thứ hạng chỉ từ ảnh, API không đọc nhãn**. Câu mẫu do code dựng (`templates.render`: "no X", No Finding là "normal chest", thứ tự CHEXPERT_LABELS, từ chối "A and no A"); lọc view là lọc metadata; câu report trích sau xếp hạng, chỉ để hiển thị. Không khái niệm → câu tự do của parser; chỉ phủ định → "no X" kèm warning; UNCERTAIN như PRESENT; trên 2 ràng buộc kèm warning | Hybrid thỏa ràng buộc nhờ lọc trên chính nhãn làm đáp án; một đường áp dụng được cho ảnh chưa có report |
| D38 | Encoder mặc định CXR-CLIP SwinT bản fine-tune FT6 (`cxrclip_ft6_lr1e-04`, từ 2026-10-29; sha256 ghim trong `models.yaml`, CC BY-NC 4.0). Bản M zero-shot và BiomedCLIP vẫn là candidate, build vẫn đăng ký. Trọng số FT6 chỉ có trên máy (`runs/`, sinh bằng code fine-tune của nhánh V1.1) | CXR-CLIP hòa BiomedCLIP zero-shot trên dev và nhanh hơn 2,4 lần (luật hòa Phase 2); FT6 hơn zero-shot trên dev, 0,539 so với 0,482 (D54) |
| D40 | Sổ đăng ký build: manifest riêng mỗi build ghi encoder; `config_hash` chỉ phủ `data.yaml` + candidate của encoder + `normalize`; mọi lệnh encode text nạp encoder qua `index.build_encoder` (từ chối revision khác build) | Hai encoder so được trên cùng pipeline mà không làm hỏng build của nhau |
| D62 | Ma trận điểm mở rộng ngoài 14 nhãn: khái niệm Finding/Disease lõi chưa `scorable` (tối đa 30 theo lượt nhắc, loại câu bình thường, từ chung chung, giải phẫu, thiết bị) được một cột nếu FT6 chấm được: caption theo quy tắc của 14 cột (3 chuỗi pool hay gặp nhất, embedding trung bình), hiệu chỉnh như 14 cột trên pool-val, nhãn bạc RadGraph (present / uncertain loại / absent hoặc không nhắc). Giữ khi ≥ 50 PRESENT và AUROC lenient ≥ 0,65 (D55); cột yếu loại hẳn, không giữ kèm cảnh báo. Kết quả: giữ 23/30, ma trận `..._ext1` (14 cột gốc giữ nguyên), 23 khái niệm thành `scorable`; độ nối D59 lên 0,945 | Người dùng 2026-11-02: phần thiếu của cổng V2-4 là dấu hiệu ngoài 14 cột; cột mới chưa có kiểm chứng nào khác nên chỉ giữ cột đạt cổng |
| D61 | Parser: Anthropic `claude-haiku-5-5` qua SDK (thay D22 `gpt-5-mini`), output là tool strict theo cùng JSON schema 14 khái niệm, `tool_choice` ép buộc, `effort: low`, cache prompt system; không gửi temperature (model Claude hiện tại trả 400 với sampling khác mặc định). Provider và model lấy từ `models.yaml:llm`; OpenAI và Ollama giữ sau `LLM_PROVIDER`. Rule parser khi LLM lỗi hoặc thiếu key. Dev 40 câu: 1,000 / 1,000 / 1,000 (gpt-5-mini 0,992 / 0,996 / 1,000), ≈ 0,14 USD mỗi 1.000 câu, p50 1,9 s. Heldout chạy một lần: 0,981 / 0,994 / 0,945, PASS (`docs/measured-facts.md`). LLM judge của V2-7 phải khác model của pipeline (pipeline dùng Haiku/Sonnet → judge dùng Opus hoặc Fable) | Người dùng 2026-11-02: hết credit OpenAI, mọi LLM của pipeline là Anthropic; đo trên đúng bộ dev và metric của Phase 4c (`docs/measured-facts.md`) |
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
| T40 | Cosine câu–ảnh của FT6 lệch âm (median −0,13, max 0,08 cho một câu trên 25.000 ảnh; zero-shot median 0,10). Thứ hạng đúng, giá trị tuyệt đối thì không | Không đặt ngưỡng hay giả định dấu trên cosine thô; điểm cần ý nghĩa tuyệt đối thì dùng logit đã hiệu chỉnh của ma trận V2-5 |
| T39 | MRREL không đồng nhất giữa các nguồn: phân cấp MSH là `PAR`/`CHD` với RELA **rỗng**, còn NCI, FMA, HPO dùng `isa`/`inverse_isa`; bộ phận trong FMA nằm ở `regional_part_of` và `constitutional_part_of`, RELA `part_of` chỉ có 23 dòng. REL/RELA mô tả quan hệ của CUI2 đối với CUI1 | Lọc `IS_A`, `PART_OF` theo cả REL lẫn RELA của từng nguồn; kiểm chiều trên một cặp đã biết trước khi tin (`docs/measured-facts.md`, *UMLS 2026AA*) |

---

## Đang chờ quyết định

| # | Câu hỏi | Đang làm theo |
|---|---|---|
| P5 | Đóng gói trọng số vào image API hay dựa vào volume `hf_cache` | Volume; chọn khi build image thật |
| P8 | Người dùng duyệt gold heldout của `benchmarks/parser_queries_v1.jsonl` | Giữ như lúc đo (sha256 `8d19c619…`); đổi gold heldout thì khai báo một lần chạy mới |

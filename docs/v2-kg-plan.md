# Kế hoạch V2: KG tri thức (UMLS) cho truy vấn X-quang

Lộ trình thực hiện `docs/v2-kg-design.md`. Mỗi task là một TIP riêng, làm xong thì dừng để người dùng review; số đo
ghi vào `docs/measured-facts.md`, quyết định vào `WORKLOG.md`. Ngưỡng của mỗi cổng chốt **trước** khi đo.

## Bảng trạng thái

`⏳` chưa tới · `🔄` đang làm · `✅` xong · `⛔` chặn

| Task | Việc | Phụ thuộc | Trạng thái |
|---|---|---|---|
| V2-1 | Kiểm kê đầu vào, lọc UMLS thành Parquet | — | ✅ |
| V2-2 | Probe từ vựng trên pool; ánh xạ 14 nhãn sang CUI | V2-1 | ✅ |
| V2-5 | Ma trận điểm CXR-CLIP FT6, hiệu chỉnh trên pool-val | V2-2 (sau phần đo) | ✅ |
| V2-5b | Đăng ký build corpus FT6 cho B1; zero-shot thành dòng phụ (D51) | V2-5 | ✅ |
| V2-3 | Linker chuỗi → CUI | V2-2 | ✅ |
| V2-4 | Mở rộng có kiểm soát; trọng số `SUGGESTS`, `TYPICALLY_AT` trên pool | V2-3 | ✅ |
| V2-5c | Mở rộng ma trận điểm ngoài 14 nhãn (khái niệm lõi có lượt nhắc lớn) | V2-4 | ✅ |
| V2-6 | Neo4j, bốn tool, LLM chọn khái niệm và diễn giải; B2 chạy end-to-end | V2-4, V2-5 | ✅ 2026-11-05 (D65, D66) |
| V2-7 | Benchmark và pooling | V2-6, V2-5b | ✅ 172 câu (dev 72, heldout 100), gold đã duyệt, pool heldout 4.602 cặp (D70, D72–D74) |
| V2-8 | Đánh giá chính thức, heldout một lần | V2-7 | ✅ heldout một lần (D75): B2 + trộn − B1 +0,138 [+0,078; +0,198]; `docs/kg-evaluation.md` |
| V2-9 | Agent (ablation) và UI | V2-8; UI chờ `pnpm` (T5) | ⏳ |
| V2-10 | Thêm ảnh không dựng lại KG: embed rồi nối logit vào ma trận bằng probe đã có (N6) | V2-6 | ⏳ |
| V2-11 | CI: pytest + ruff mỗi lần push (N12) | — | ⏳ |
| V2-12 | Số đo vận hành: log JSON từng request, báo cáo độ trễ, chi phí LLM, tỉ lệ fallback (N10) | V2-6 | ⏳ |
| V2-13 | Runbook và chạy thử `docker compose` trên máy sạch (N12, T15) | V2-6 | ⏳ |
| V2-14 | Quy trình kiểm lại hiệu chỉnh cho ảnh từ nguồn khác (N7) | V2-6 | ⏳ |

Ngoài V2 (chưa xếp lịch):

| Task | Việc | Phụ thuộc | Trạng thái |
|---|---|---|---|
| X-1 | Cột vòng hai: scoliosis, dextroscoliosis, pneumoperitoneum, hyperinflation (`graph.round_two_columns`, 4 Disease lõi không chấm được) — cần thêm dữ liệu hiệu chỉnh | dữ liệu hiệu chỉnh mới | ⏳ |

38 Disease lõi chưa có đường và có dưới 50 lượt nhắc trên pool (danh sách ở `data/kg/graph/diseases_without_route.json`;
sau khi thêm cạnh của lượt rà đường còn 37): không chấm được, không phân xử thêm trong V2.

**TIẾP THEO:** duyệt gold heldout (P9), rồi chạy heldout một lần (`docs/kg-evaluation.md`, *Chạy heldout*); song song: alias tiếng Việt cho bệnh, cạnh graph của xơ phổi và chấn thương (error analysis). B2 chạy end-to-end (`cxr kg-query`, `POST /api/v1/search/kg`; D66); ma trận 37 cột (D62); parser dùng `claude-haiku-5-5` (D61). TIP nằm ở `notes/tip-*.md` (gitignored).

## Dữ liệu

```
data/umls/raw/                                  bản gốc NLM, chỉ đọc (D50)
  umls-2026AA-metathesaurus-level0.zip            UMLS_ZIP
  2026AA/META/*.RRF                               UMLS_META_DIR
data/umls/2026AA/                               cxr kg-filter-umls: mrconso, mrsty, mrrel, mrsab .parquet + manifest.json
data/kg/{vocab,link,graph}/                     V2-2 → V2-4: từ vựng, bảng liên kết, nút, cạnh, manifest
```

Cả hai thư mục nằm dưới `data/` (gitignored). Không commit, không đưa vào Docker image (D49). Thứ được commit: code,
`configs/kg.yaml` và bảng ánh xạ của dự án sang CUI (chỉ mã, không tên).

## Hiện trạng trên `main` (2026-11-08)

| Thành phần | Có gì | Thiếu gì |
|---|---|---|
| Pool, patient split | `cxr build-probe-pool` → `data/probe_pool/` (88.853 study); `src/embeddings/probe.py:patient_split` (sha256 của `seed:patient`, chia theo bệnh nhân) | — |
| RadGraph-XL | Khớp text CSV 100% ở cả hai section; phủ 25.000/25.000 ca corpus và cả pool (V2-2) | — |
| CXR-CLIP | FT6 `cxrclip_ft6_lr1e-04` là encoder mặc định (D38, D54); zero-shot và BiomedCLIP vẫn là candidate; ba build corpus đã đăng ký | Trọng số FT6 chỉ có trên máy (`runs/`) |
| Embedding | Pool 88.853 ảnh (zero-shot); pool-val 9.037 ảnh (zero-shot và FT6); corpus FT6 25.000 ảnh | — |
| Ma trận điểm | `data/indexes/image_scores/<build_id>_ext1/`: 37 cột (14 nhãn + 23 cột mở rộng, D62), hiệu chỉnh trên pool-val; `cxr kg-quality` → `quality.json` (6 cột yếu, D55) | Cột vòng hai (X-1) |
| KG | `cxr kg-build-graph` → `data/kg/graph/`, `kg_build_id` **kg_1637d5c0ee06** (băm năm bảng KG; bảng ghi theo thứ tự cố định nên dựng lại hay sửa chú thích config không đổi id): 9.632 nút (lõi 595), IS_A 10.234, PART_OF 3.826, SUGGESTS 69, TYPICALLY_AT 579; độ nối D59 theo lượt nhắc 0,948; cạnh tay ở `graph.manual_edges` | 48 Disease lõi có lượt nhắc vẫn chưa có đường (`docs/v2-6-smoke.md`) |
| Neo4j | service `neo4j` (profile `kg`), `cxr kg-load-neo4j`, `validate-indexes --kg` (D65) | — |
| Truy vấn B2 | `cxr kg-query`, `POST /api/v1/search/kg` (D66–D69); parser, chooser, diễn giải đều dùng `claude-haiku-5-5`; smoke 24 câu ở `docs/v2-6-smoke.md` | Diễn giải v4 kém trung thực khi so cặp (P22); N2 dao động quanh 1,5 s; CI (V2-11); UI chờ `pnpm` (T5) |

## Các task

### V2-1 · Kiểm kê đầu vào và lọc UMLS (xong)

TIP của người dùng ngày 2026-10-27. Ra: `data/umls/2026AA/*.parquet` + `manifest.json`, số đo trong `docs/measured-facts.md`,
`configs/kg.yaml`, `src/kg/umls_filter.py`, `cxr kg-filter-umls`, `tests/kg/`. Quyết định D45–D50, bẫy T39.
MD5 khớp; đủ MSH, NCI, FMA, HPO; không có tiếng Việt; MRCONSO còn 5.681.716 dòng, MRREL 6.111.416 dòng.

### V2-2 · Probe từ vựng trên pool, ánh xạ 14 nhãn (xong)

- Đo độ phủ RadGraph-XL trên toàn pool (join text theo T29); không tách, vì trọng số KG dùng cả pool-train lẫn
  pool-val (thiết kế, *Chia dữ liệu*).
- Đếm chuỗi observation (ghép modifier theo `modify`) và chuỗi anatomy trên pool; vẽ đường cong độ phủ; lấy phần đầu
  phủ 90–95% số lần nhắc.
- Ánh xạ 14 nhãn CheXpert sang CUI, hoặc `PRJ:<tên>` kèm `IS_A` tới CUI gần nhất và các CUI con (D53); người dùng
  duyệt. File chỉ chứa mã.
- **Cổng:** cả 14 nhãn có CUI hoặc ID dự án (cổng bước 1 của thiết kế); phần đầu phủ 90% đủ nhỏ để duyệt tay.
- **Đo 2026-10-27:** impression khớp 99,997%, findings 97,6%; Observation theo D52 phủ 90% lượt nhắc bằng 265 chuỗi
  (cổng PASS); 13 nhãn có ứng viên CUI khớp chính xác. Người dùng chọn `PRJ` cho 3 nhãn chỉ khớp khái niệm hẹp hơn
  (D53). Ra: `configs/kg/chexpert_cuis.yaml` (10 CUI, 3 `PRJ` kèm `is_a` và `children`, `PRJ:no_finding`).

### V2-3 · Linker chuỗi → CUI

- Khớp chính xác sau chuẩn hóa với `mrconso.parquet`; không khớp thì SapBERT tìm láng giềng gần nhất, chỉ trong các
  semantic type cho phép (bảng gán nhãn của thiết kế). Tham khảo Nützel và cộng sự (arXiv 2508.19915).
- Người dùng duyệt tay phần đầu bảng.
- **Cổng:** precision của linker trên 200 chuỗi lấy mẫu theo seed đạt ngưỡng.

### V2-4 · Mở rộng có kiểm soát và trọng số

- Giải phẫu: cây vùng ngực của FMA theo `PART_OF`, `IS_A`. Finding và bệnh: bao đóng `IS_A` lên, xuống của lõi, dừng
  ở danh sách gốc chặn. Thiết bị: các nhánh có trong từ vựng V2-2. Khái niệm gắn bên quy về cha không phân bên.
- Cạnh phân cấp CheXpert (`source: 'chexpert'`); alias tiếng Việt từ `concepts.yaml` và LLM đề xuất, người duyệt.
- `SUGGESTS` và `TYPICALLY_AT` tính trên pool (`n`, `p`, bỏ cặp dưới ngưỡng `n`).
- **Cổng:** đếm nút, cạnh theo loại; tỉ lệ khái niệm nối được về khái niệm `scorable`; 50 cạnh `SUGGESTS` mạnh nhất
  được người có kiến thức y khoa xác nhận. Bước này quyết có cần SNOMED CT không (D48).

### V2-5c · Mở rộng ma trận điểm ngoài 14 nhãn (xong)

- Mỗi khái niệm lõi Finding hay Disease không `scorable` và có lượt nhắc lớn trên pool (D60: low lung volumes, emphysema,
  calcification, osteopenia, degenerative…): một câu mô tả, cosine với ảnh, hiệu chỉnh trên pool-val với nhãn bạc từ
  RadGraph (present hoặc absent trên report pool-val).
- Giữ cột có AUROC lenient ≥ 0,65 (D55); khái niệm có cột thì thành `scorable`.
- Thứ tự: V2-5c → V2-6 (parser đã chuyển sang Anthropic, D61).
- **Đo 2026-11-02** (D62): giữ 23/30 cột (AUROC lenient 0,675–0,936); ma trận `..._ext1`; độ nối về `scorable` 0,945.
- **Hoàn thiện 2026-11-03** (D63): 3 cột borderline; thêm `IS_A` → Lung Opacity cho interstitial marking và lung
  opacification; quy tắc gộp cho nhãn cha yếu (Fracture noisy-OR, Atelectasis cột cha); 8 alias mới đã phân xử (giữ 6).

### V2-5 · Ma trận điểm (xong)

- Mang candidate FT6 và `cxr build-image-scores` từ nhánh `feat/label-aware-image-retrieval` về, chỉ các file cần.
- Ma trận giữ 14 cột theo tên nhãn (H6), logit hiệu chỉnh trên pool-val; nối cột với CUI làm trong KG. Không dùng
  pool-train để hiệu chỉnh.
- Ba kiểu câu sinh logit (P13) và lựa chọn joint hay decomposed để task sau.
- **Cổng:** ma trận đủ 25.000 study và 14 cột; `build_id` của ma trận khớp encoder.
- **Đo 2026-10-29:** cổng đạt; hiệu chỉnh trên 9.037 ảnh pool-val, không nhãn nào `uncalibrated` hay `saturated`.
  AUROC strict trên pool-val từ 0,474 (Fracture) tới 0,860 (Pleural Effusion); ba nhãn dưới 0,5 (`docs/measured-facts.md`).
  `load_encoder` tìm adapter theo `framework` của candidate (người dùng duyệt thêm `src/embeddings/base.py`).

### V2-5b · B1 dùng FT6 (D51, xong)

Đăng ký build corpus FT6 (`cxr index`), đo B1-FT6 và B1-zero-shot trên dev; B1-FT6 là dòng chính, zero-shot là dòng
phụ trong bảng đánh giá.

- **Đo 2026-10-29** (parser gold, cùng code): FT6 0,539 so với zero-shot 0,482; theo câu 24 hơn, 10 kém, 4 hòa, sign test
  p ≈ 0,024. Zero-shot tái lập đúng 0,482. Theo D54, `selected` đổi sang FT6.

### V2-6 · Neo4j, bốn tool, LLM (xong 2026-11-05)

Kết quả: D65 (Neo4j, nạp, chấm điểm trong RAM), D66 (đường truy vấn, rào chắn, API). `explain_path` là đường đi do
`expand_concepts` ghi sẵn (`Intent.path`) cộng điểm từng khái niệm của mỗi ảnh (`ScoreMatrix.probabilities`), không có
truy vấn Cypher thứ năm. Smoke 24 câu và số đo N1–N4 ở `docs/measured-facts.md`; việc còn lại của 6e (luật chọn cho "giảm
thể tích phổi trái") đạt ở D67 (chú giải đi cùng ứng viên).

Sau smoke có đợt sửa định tuyến và độ phủ KG (D67): nhãn cha không nuốt khái niệm con, vị trí chỉ cảnh báo khi đi cùng dấu hiệu, cạnh
thủ công (Pleural Diseases, interstitial lung disease), `sides`, nạp sẵn ma trận. Việc còn lại trước V2-7: Disease chưa có đường
tới cột (76/124, chờ phân xử), N2 chưa đạt, góc sườn hoành bão hòa, bật SapBERT cho ablation. Đợt hai (D68): SapBERT là bước
cuối của lookup, vùng giải phẫu dùng noisy-OR có trọng số (hết bão hòa), alias trong facts, cache chooser trên đĩa; còn 67
Disease chờ phân xử, 4 là ứng viên cột vòng hai; N2 p95 1,51 s.

- **6a.** Service `neo4j` (Community, profile `kg`); nạp `Concept`, `Alias`, `SemanticType`, `Image`, cạnh tri thức và
  `PREDICTED` bằng `UNWIND` (chậm thì `neo4j-admin database import full`); user chỉ đọc; build manifest;
  `validate-indexes` thêm bốn check của thiết kế.
- **6b.** Parser thêm trường `terms`; `lookup_concept`, `expand_concepts`, `score_images`, `explain_path` (Cypher
  template có tham số, không do LLM viết).
- **6c.** LLM chọn CUI trong ứng viên; LLM viết diễn giải từ đường đi và điểm; rào chắn: kiểm căn cứ, chặn ngôn ngữ
  chẩn đoán, cảnh báo bắt buộc, ghi vết.
- **6d. Yêu cầu phi chức năng gộp vào V2-6** (`docs/v2-kg-design.md`, *Yêu cầu phi chức năng*): ngân sách độ trễ
  và đo từng bước (N1); diễn giải streaming (N2); timeout 3 s mỗi lượt LLM, circuit breaker, fallback tất định (N3);
  ma trận điểm nạp vào RAM, chấm điểm không đọc Neo4j (N4); `PREDICTED` chỉ top-N mỗi khái niệm (N5); API vẫn trả kết
  quả khi mất LLM hoặc Neo4j, `/health` báo Neo4j (N8); phản hồi ghi `kg_build_id`, phiên bản ma trận, model, sha256
  prompt (N9).
- **6e. Chọn khái niệm theo ngữ cảnh** (D64): LLM chọn CUI nhận cả câu hỏi gốc, không chỉ cụm từ. Smoke bắt buộc:
  "giảm thể tích phổi trái" → Atelectasis; "giảm thể tích phổi hai bên" → low lung volumes (C5563137). Hiện rule
  parser và Haiku đều gán Atelectasis cho cả hai câu (lexicon `concepts.yaml` bắt cụm), nên parser phải đưa cụm mơ hồ
  vào `terms` để `lookup_concept` thấy được, hoặc không gán nhãn khi cụm thuộc nhiều khái niệm.
- **Cổng:** khoảng 20 câu EN/VI chạy end-to-end bằng B2; p95 < 5 s chưa tính diễn giải; test lỗi LLM cho fallback
  < 1 s.

### V2-10 – V2-14 · Vận hành (sau V2-6)

- **V2-10.** Lệnh thêm ảnh: embed shard mới, nối logit bằng probe đã hiệu chỉnh, không đổi các dòng cũ của ma trận,
  không dựng lại KG; 1.000 ảnh < 5 phút CPU.
- **V2-11.** CI (GitHub Actions): `uv sync`, `pytest -q`, `ruff check`; không cần dữ liệu thật; < 5 phút.
- **V2-12.** Nạp `configs/logging.yaml` (đang chưa dùng); mỗi request một dòng log JSON có độ trễ từng bước, token và
  chi phí LLM, cờ fallback, các id truy vết; lệnh báo cáo p50/p95, tỉ lệ fallback, chi phí mỗi 1.000 câu.
- **V2-13.** Runbook: dựng từ đầu, nâng cấp encoder hoặc KG, khôi phục khi Qdrant hoặc Neo4j hỏng; build và chạy
  thật image API (T15).
- **V2-14.** Ảnh từ nguồn khác: ≥ 200 ảnh có nhãn, AUROC lenient theo cột, gắn cờ cột < 0,65 cho nguồn đó.

### V2-7 · Benchmark và pooling

- **Dev xong 2026-11-10** (D70, `docs/kg-evaluation.md`): 72 câu, gold Opus + duyệt Sonnet; pool 1.000 cặp, judge Opus,
  kappa với Sonnet 0,702; nDCG@10 B0 0,396 · B1 0,512 · B2 0,541 · Tham chiếu 0,663; B2 − B1 +0,029 [−0,065; +0,114].

60–80 câu EN/VI, chia dev và heldout. Các nhóm: đồng nghĩa và tiếng Việt, khái niệm hẹp hơn, rộng hơn, bệnh không thấy
trực tiếp, phủ định khái niệm rộng, vị trí giải phẫu, cộng nhóm đối chứng. Pooling: hợp top-10 của B0–B3, LLM chấm trên
chữ report, người chấm một mẫu để báo Cohen's kappa.

### V2-8 · Đánh giá chính thức

Bốn lớp của thiết kế (chất lượng KG, hiểu câu hỏi, end-to-end, diễn giải). Heldout chạy một lần, có khai báo (D24);
error analysis theo nhóm câu; các ablation của thiết kế.

Diễn giải theo tri thức KG (WORKLOG D71, nhánh `feat/kg-knowledge-explanation`): gloss 298 khái niệm, phần "kiến thức
liên quan" dẫn [K#], cờ `explain.knowledge` cho ablation lớp 4 của V2-8. Chờ: phân xử `definitions_review.csv` (P20).

### V2-9 · Agent và UI

Agent tự gọi tool (ablation, tối đa 5 lượt). UI hiển thị đường đi, diễn giải và cảnh báo; chờ `pnpm` (T5).

## Nhánh

Theo CLAUDE.md §6: mỗi task một nhánh tách từ `main`, đặt tên theo công việc, PR nhỏ, merge commit. Commit khi người
dùng bảo.

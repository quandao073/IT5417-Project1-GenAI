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
| V2-5b | Đăng ký build corpus FT6 cho B1; zero-shot thành dòng phụ (D51) | V2-5 | ⏳ TIP đã duyệt |
| V2-3 | Linker chuỗi → CUI | V2-2 | ⏳ |
| V2-4 | Mở rộng có kiểm soát; trọng số `SUGGESTS`, `TYPICALLY_AT` trên pool | V2-3 | ⏳ |
| V2-6 | Neo4j, bốn tool, LLM chọn khái niệm và diễn giải; B2 chạy end-to-end | V2-4, V2-5 | ⏳ |
| V2-7 | Benchmark và pooling | V2-6, V2-5b | ⏳ |
| V2-8 | Đánh giá chính thức, heldout một lần | V2-7 | ⏳ |
| V2-9 | Agent (ablation) và UI | V2-8; UI chờ `pnpm` (T5) | ⏳ |

**TIẾP THEO:** V2-5b (`notes/tip-v2-5b.md`). TIP nằm ở `notes/tip-*.md` (gitignored).

## Dữ liệu

```
data/umls/raw/                                  bản gốc NLM, chỉ đọc (D50)
  umls-2026AA-metathesaurus-level0.zip            UMLS_ZIP
  2026AA/META/*.RRF                               UMLS_META_DIR
data/umls/2026AA/                               cxr kg-filter-umls: mrconso, mrsty, mrrel, mrsab .parquet + manifest.json
data/kg/<kg_build_id>/                          V2-2 → V2-4: từ vựng, bảng liên kết, nút, cạnh, manifest (dự kiến)
```

Cả hai thư mục nằm dưới `data/` (gitignored). Không commit, không đưa vào Docker image (D49). Thứ được commit: code,
`configs/kg.yaml` và bảng ánh xạ của dự án sang CUI (chỉ mã, không tên).

## Hiện trạng trên `main` (2026-10-27)

| Thành phần | Có gì | Thiếu gì |
|---|---|---|
| Pool, patient split | `cxr build-probe-pool` → `data/probe_pool/` (88.853 study); `src/embeddings/probe.py:patient_split` (sha256 của `seed:patient`, chia theo bệnh nhân) | — |
| RadGraph-XL | Khớp text CSV 100% ở cả hai section; phủ 25.000/25.000 ca corpus; độ phủ trên pool đo ở V2-2 | — |
| CXR-CLIP | Zero-shot `data/cache/models/cxrclip/swint_m.pt`, build corpus `20261023_chexpert25k_cxrclip_v1` đã đăng ký | Candidate FT6 `cxrclip_ft6_lr1e-04` (`runs/FT6_lr1e-04/model.pt`) mang về ở V2-5 |
| Embedding | Pool 88.853 ảnh (zero-shot); pool-val 9.037 ảnh (zero-shot và FT6); corpus FT6 25.000 ảnh | Build corpus FT6 chưa đăng ký trên `main` |
| Ma trận điểm | V2-5: `cxr build-image-scores` → `data/indexes/image_scores/<build_id>/`, ma trận FT6 đã dựng | — |

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

### V2-5 · Ma trận điểm (xong)

- Mang candidate FT6 và `cxr build-image-scores` từ nhánh `feat/label-aware-image-retrieval` về, chỉ các file cần.
- Ma trận giữ 14 cột theo tên nhãn (H6), logit hiệu chỉnh trên pool-val; nối cột với CUI làm trong KG. Không dùng
  pool-train để hiệu chỉnh.
- Ba kiểu câu sinh logit (P13) và lựa chọn joint hay decomposed để task sau.
- **Cổng:** ma trận đủ 25.000 study và 14 cột; `build_id` của ma trận khớp encoder.
- **Đo 2026-10-29:** cổng đạt; hiệu chỉnh trên 9.037 ảnh pool-val, không nhãn nào `uncalibrated` hay `saturated`.
  AUROC strict trên pool-val từ 0,474 (Fracture) tới 0,860 (Pleural Effusion); ba nhãn dưới 0,5 (`docs/measured-facts.md`).
  `load_encoder` tìm adapter theo `framework` của candidate (người dùng duyệt thêm `src/embeddings/base.py`).

### V2-5b · B1 dùng FT6 (D51)

Đăng ký build corpus FT6 (`cxr index`), đo B1-FT6 và B1-zero-shot trên dev; B1-FT6 là dòng chính, zero-shot là dòng
phụ trong bảng đánh giá.

### V2-6 · Neo4j, bốn tool, LLM

- **6a.** Service `neo4j` (Community, profile `kg`); nạp `Concept`, `Alias`, `SemanticType`, `Image`, cạnh tri thức và
  `PREDICTED` bằng `UNWIND` (chậm thì `neo4j-admin database import full`); user chỉ đọc; build manifest;
  `validate-indexes` thêm bốn check của thiết kế.
- **6b.** Parser thêm trường `terms`; `lookup_concept`, `expand_concepts`, `score_images`, `explain_path` (Cypher
  template có tham số, không do LLM viết).
- **6c.** LLM chọn CUI trong ứng viên; LLM viết diễn giải từ đường đi và điểm; rào chắn: kiểm căn cứ, chặn ngôn ngữ
  chẩn đoán, cảnh báo bắt buộc, ghi vết.
- **Cổng:** khoảng 20 câu EN/VI chạy end-to-end bằng B2.

### V2-7 · Benchmark và pooling

60–80 câu EN/VI, chia dev và heldout. Các nhóm: đồng nghĩa và tiếng Việt, khái niệm hẹp hơn, rộng hơn, bệnh không thấy
trực tiếp, phủ định khái niệm rộng, vị trí giải phẫu, cộng nhóm đối chứng. Pooling: hợp top-10 của B0–B3, LLM chấm trên
chữ report, người chấm một mẫu để báo Cohen's kappa.

### V2-8 · Đánh giá chính thức

Bốn lớp của thiết kế (chất lượng KG, hiểu câu hỏi, end-to-end, diễn giải). Heldout chạy một lần, có khai báo (D24);
error analysis theo nhóm câu; các ablation của thiết kế.

### V2-9 · Agent và UI

Agent tự gọi tool (ablation, tối đa 5 lượt). UI hiển thị đường đi, diễn giải và cảnh báo; chờ `pnpm` (T5).

## Nhánh

Mọi task V2 làm tuần tự trên `feat/v2-knowledge-graph` (tách từ `main` `1b50c19`), không dùng worktree. Commit khi
người dùng bảo.

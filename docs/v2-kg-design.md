# Thiết kế V2: Truy vấn X-quang bằng KG tri thức (UMLS), LLM và CXR-CLIP

Cập nhật Oct 27, 2026 (bản đầu 24/10) · @Quân Đào

## Mục tiêu và phạm vi

V2 thêm một Knowledge Graph tri thức y khoa làm trung gian giữa câu hỏi và điểm ảnh của CXR-CLIP. KG dựng từ UMLS, cộng thêm trọng số thống kê lấy từ RadGraph-XL trên report của pool. Lúc truy vấn, hệ thống chỉ dùng ảnh: không đọc report, không đọc nhãn (D39). Pipeline một đường hiện tại giữ nguyên, làm mode riêng và làm baseline B1.

- **Câu hỏi nghiên cứu:** KG có cải thiện truy vấn chứa khái niệm mà model ảnh không học trực tiếp không? Đó là từ đồng nghĩa, khái niệm hẹp hơn, khái niệm rộng hơn, vị trí giải phẫu và bệnh không thấy trực tiếp trên phim. So với pipeline hiện tại (B1).
- **Câu hỏi phụ:**
  - KG có làm tụt câu hỏi nằm gọn trong tập khái niệm đã học không?
  - LLM chọn khái niệm trong ứng viên của KG có hơn tra cứu tất định không?
  - Diễn giải do LLM viết có trung thực với KG không?
  - LLM agent tự gọi tool có hơn pipeline cố định không? Đây là ablation.
- **Người dùng:** nghiên cứu viên và sinh viên khám phá dataset. Không dùng cho chẩn đoán.

| Loại câu hỏi | Ví dụ | B1: parser, template, CLIP | V2: thêm KG |
| --- | --- | --- | --- |
| Đồng nghĩa, tiếng Việt | "dịch màng phổi", "bóng tim to" | Phụ thuộc parser và `concepts.yaml` | Alias từ UMLS cộng alias tiếng Việt; LLM chọn trong ứng viên |
| Khái niệm hẹp hơn | "PICC", "viêm phổi thùy" | Không có, hoặc parser tự quy về nhãn | Đi lên `IS_A` tới khái niệm chấm được, kèm cảnh báo |
| Khái niệm rộng hơn | "mờ phổi" theo nghĩa mọi tổn thương gây mờ | Chỉ cột Lung Opacity | Gộp chính nó với các khái niệm con chấm được |
| Vị trí giải phẫu | "bất thường màng phổi" | Không có | `PART_OF` rồi `TYPICALLY_AT` sang các finding chấm được |
| Bệnh không thấy trực tiếp | "nghi suy tim" | Không có | `SUGGESTS` sang các finding, có trọng số |
| Phủ định khái niệm rộng | "không có mờ phổi" | Phủ định một cột | Phủ định mọi khái niệm con |
| Bên, mức độ | "tràn dịch phải lượng ít" | Không có | Ngoài phạm vi tới khi model ảnh học được (V1.2); bỏ bên và mức độ, kèm cảnh báo |

**Đối chiếu với đề bài.** Mỗi yêu cầu của đề tài ứng với một phần của thiết kế:

| Đề bài yêu cầu | Thiết kế V2 |
| --- | --- |
| Tri thức về giải phẫu phổi, bệnh lý và dấu hiệu hình ảnh được mô hình hóa thành KG | Nút `Anatomy`, `Disease`, `Finding`, `Device` từ UMLS; cạnh `IS_A`, `PART_OF`, `SUGGESTS`, `TYPICALLY_AT` |
| KG làm trung gian giữa truy vấn ngôn ngữ và đặc trưng hình ảnh | Câu hỏi → khái niệm trong KG → cột điểm của CXR-CLIP; cạnh `PREDICTED` nối ảnh với Finding |
| LLM phân tích truy vấn | Parser |
| LLM ánh xạ sang khái niệm trong KG | LLM chọn một CUI trong các ứng viên mà `lookup_concept` trả |
| LLM hỗ trợ diễn giải kết quả | LLM viết diễn giải từ đường đi trong KG và điểm (RAG trên KG) |
| Xây KG y khoa từ UMLS, RadGraph | UMLS làm xương sống; RadGraph cho từ vựng lõi và trọng số `SUGGESTS`, `TYPICALLY_AT` |

## Quyết định nền tảng

So với bản ngày 24/10, ba quyết định đổi trục của tài liệu:

1. **Report không vào đường truy vấn.** Mục tiêu là ảnh chưa có report. Report chỉ dùng offline: để tính trọng số cho KG và để đánh giá.
2. **KG chỉ chứa tri thức.** Nút là khái niệm (Disease, Finding, Anatomy, Device) và tên gọi của chúng. Không còn tầng dữ kiện theo study (`Study`, `Section`, `Fact`).
3. **Ảnh nối vào KG qua Finding, rồi tới Disease.** Điểm CXR-CLIP đã hiệu chỉnh là cạnh `PREDICTED` từ ảnh tới Finding. Finding nối tới Disease bằng `SUGGESTS`.

Hệ quả:

- Bỏ cách "KG lọc study theo report, CLIP xếp" và bỏ tool `find_studies`. Giờ KG chọn khái niệm và trọng số; CXR-CLIP chấm từng ảnh, điểm tính sẵn thành ma trận.
- Pipeline cố định là bản chính. LLM làm ba việc: phân tích câu hỏi, chọn khái niệm trong ứng viên của KG, diễn giải kết quả. LLM không quyết định thứ hạng. Agent tự gọi tool là một ablation.
- Vòng khép kín ở khâu đánh giá biến mất ngay từ thiết kế, vì hệ thống không đọc report lúc truy vấn. Nhãn bạc từ RadGraph trên corpus dùng được để chấm, với điều kiện mọi trọng số của KG lấy từ pool, không lấy từ corpus.
- UMLS thay cho bộ từ vựng tự dựng. UMLS có sẵn CUI, tên đồng nghĩa, semantic type và phân cấp, nên bước chuẩn hóa thành bài toán liên kết chuỗi với CUI.
- KG giới hạn trong miền X-quang ngực, không nạp toàn bộ UMLS. Lý do không phải dung lượng: đề bài không cần, và tra cứu sẽ nhiễu ("effusion" khớp cả tràn dịch khớp gối).

## Kiến trúc tổng thể

Hệ thống có hai giai đoạn. Offline: dựng KG và ma trận điểm, mỗi build một lần. Online: trả lời câu hỏi bằng một pipeline cố định.

&#91;embedded content: Dựng offline · ba nguồn, hai kho, nối qua CUI\]

UMLS và RadGraph dựng KG; ảnh qua CXR-CLIP thành ma trận điểm. Hai kho nối nhau bằng CUI: mỗi cột của ma trận ứng với một khái niệm `scorable` trong KG. Cạnh `PREDICTED` được nạp vào Neo4j để đồ thị thể hiện liên kết giữa ảnh và khái niệm; chấm điểm vẫn đọc ma trận.

&#91;embedded content: Truy vấn online · pipeline cố định, năm bước\]

Parser phân tích câu hỏi. `lookup_concept` trả các CUI ứng viên và LLM chọn một. `expand_concepts` đổi CUI thành các khái niệm chấm được kèm trọng số. `score_images` cộng logit trên ma trận rồi lọc view. `explain_path` lấy đường đi trong KG, và LLM viết diễn giải từ đó. Không bước nào đọc report hay nhãn.

## Nguồn dữ liệu và giấy phép

| Nguồn | Phần dùng | Đóng góp | Lúc truy vấn |
| --- | --- | --- | --- |
| UMLS 2026AA | Level 0 Subset; SNOMED CT chỉ khi cần | CUI, tên đồng nghĩa EN, semantic type, `IS_A`, `PART_OF` | Đọc KG |
| RadGraph-XL (CheXpert Plus) | Report của pool | Chọn khái niệm cần có; trọng số `SUGGESTS`, `TYPICALLY_AT` | Không |
| RadGraph-XL | Report của corpus | Không vào KG; chỉ làm nhãn bạc khi đánh giá | Không |
| CXR-CLIP bản fine-tune | Ảnh corpus; hiệu chỉnh trên pool-val | Ma trận logit, mỗi cột một CUI (cạnh `PREDICTED`) | Đọc ma trận |
| Tự soạn | `concepts.yaml`, LLM đề xuất, người duyệt | Alias tiếng Việt; ánh xạ 14 nhãn CheXpert sang CUI | Đọc KG |

**UMLS**

- Bản 2026AA, phát hành ngày 4/5/2026. Tải **Metathesaurus Level 0 Subset**: nén 1,9 GB, giải nén 10,5 GB, MD5 `7c37d94ed322c244edc060384cd16027`. Đặt ngoài repo vì giấy phép.
- Subset dựng sẵn có cả nội dung đã lỗi thời và bị ẩn, nên phải lọc `SUPPRESS = 'N'`. Subset không kèm Semantic Network; phân cấp semantic type dựng từ cột `STN` của MRSTY.
- Level 0 Subset chỉ gồm các nguồn mức 0, trong đó có MeSH, NCI Thesaurus, FMA, HPO. Các nguồn này không có hạn chế nào ngoài điều khoản chuẩn của license, và đủ cho mọi bước của thiết kế.
- SNOMED CT không có trong Level 0 Subset. Nó có phân cấp finding và giải phẫu tốt nhất, nhưng có điều kiện riêng:
  - ngoài nước thành viên của SNOMED International có thể phải trả phí, và phải khai báo qua MLDS trước khi dùng;
  - Việt Nam có tên trong bảng phí gia nhập 2025 của SNOMED International, nên nhiều khả năng chưa là thành viên.

  Chỉ tính tới khi bước 4 cho thấy nguồn mức 0 không đủ: khai báo MLDS, tải thêm Full Subset (nén 5,4 GB, giải nén 38 GB) và bật cờ `umls.include_snomed`.
- UMLS gần như chắc chắn không có nguồn tiếng Việt. Kiểm bằng cột `LAT` của MRCONSO ở bước 1.
- RadLex không nằm trong UMLS. Để sau.

**Giấy phép khi repo công khai**

License UMLS cấm phân phối dữ liệu UMLS hoặc subset của nó, trừ khi là phần gắn liền của ứng dụng.

- Commit: script dựng KG, và bảng ánh xạ khái niệm của dự án sang CUI (chỉ mã).
- Không commit: chuỗi tên lấy từ UMLS, file Parquet, CSV, volume Neo4j.
- Docker image không đóng gói sẵn KG. Ai chạy lại thì tự dựng bằng license của mình.
- README liệt kê các nguồn UMLS đã dùng.

**RadGraph-XL**

- CheXpert Plus có RadGraph riêng cho Findings và Impression trên toàn bộ dataset, nên pool cũng có. Đo độ phủ trên pool trước khi dùng.
- Join theo text đã chuẩn hóa của đúng section (T29), không bao giờ join theo vị trí.
- RadGraph là output của model và chưa ai kiểm. Chỉ dùng làm trọng số và nhãn bạc, không coi là sự thật.

**Chia dữ liệu**

| Phần | Dùng cho |
| --- | --- |
| pool-train | Fine-tune CXR-CLIP (đã làm); thống kê trọng số KG |
| pool-val | Thống kê trọng số KG; hiệu chỉnh logit; chọn tham số và cách gộp điểm |
| Corpus trừ 200 valid | Kho ảnh; đánh giá chính (eval\_large) |
| 200 study valid | Đánh giá sạch (eval\_clean), chỉ báo cáo |

Hiệu chỉnh và chọn tham số không dùng pool-train, vì CXR-CLIP đã học trên phần này.

## Schema Knowledge Graph

KG chỉ có tầng tri thức. Không có nút theo study; ảnh tham gia qua ma trận điểm.

Bốn quyết định thiết kế:

1. **ID là CUI.** CUI ổn định giữa các bản UMLS; build manifest ghi bản UMLS và danh sách nguồn (SAB). Nhãn CheXpert nào không có CUI tương đương (có thể là Support Devices, No Finding) thì nhận ID dự án `PRJ:<tên>` và nối `IS_A` tới CUI gần nhất.
2. **Một loại nút `Concept`, nhãn phụ theo semantic type.** Gán nhãn bằng quy tắc ở bảng dưới; 14 nhãn CheXpert duyệt tay.
3. **Giải phẫu không phân bên.** FMA và SNOMED có khái niệm gắn bên, ví dụ "left lower lobe". Khi liên kết, quy về khái niệm cha không phân bên; bên là thuộc tính của câu hỏi, không phải của nút.
4. **Phân cấp CheXpert thêm tay.** CheXpert coi Lung Opacity bao Edema, Consolidation, Lung Lesion, Atelectasis (Consolidation bao Pneumonia), và Enlarged Cardiomediastinum bao Cardiomegaly. UMLS có thể không có đúng các cạnh này, nên thêm `IS_A` với `source: 'chexpert'`.

| Nút | Thuộc tính | Nguồn |
| --- | --- | --- |
| `Concept`, kèm một trong `Finding`, `Disease`, `Anatomy`, `Device` | `cui`, `name`, `sty`, `chexpert_label`, `scorable`, `n_pool` | UMLS, tự soạn, RadGraph |
| `Alias` | `text`, `norm`, `lang`, `source` | MRCONSO, `concepts.yaml`, LLM rồi người duyệt |
| `SemanticType` | `tui`, `name`, `stn` | MRSTY |
| `Image` | `image_id`, `study_id`, `view` | Corpus |

- `scorable`: model ảnh có cột điểm cho khái niệm này.
- `n_pool`: số report của pool nhắc tới khái niệm.

| Quan hệ | Từ → đến | Thuộc tính | Nguồn |
| --- | --- | --- | --- |
| `IS_A` | `Concept` → `Concept` | `sab` hoặc `source` | MRREL, thêm tay |
| `IS_A` | `SemanticType` → `SemanticType` |  | Cột `STN` của MRSTY |
| `PART_OF` | `Anatomy` → `Anatomy` | `sab` | MRREL (FMA) |
| `HAS_ALIAS` | `Concept` → `Alias` |  | MRCONSO, tự soạn |
| `HAS_STY` | `Concept` → `SemanticType` |  | MRSTY |
| `SUGGESTS` | `Finding` → `Disease` | `n`, `p` | `suggestive_of` trên pool |
| `TYPICALLY_AT` | `Finding` → `Anatomy` | `n`, `p` | `located_at` trên pool |
| `PREDICTED` | `Image` → `Finding` | `logit`, `build_id` | Ma trận điểm; nạp vào Neo4j để demo và giải thích, chấm điểm vẫn đọc ma trận |

**Gán nhãn theo semantic type** (quy tắc mặc định):

| Nhãn | Semantic type |
| --- | --- |
| `Disease` | T047 Disease or Syndrome, T191 Neoplastic Process |
| `Finding` | T033 Finding, T046 Pathologic Function, T184 Sign or Symptom, T190 Anatomical Abnormality, T037 Injury or Poisoning |
| `Device` | T074 Medical Device |
| `Anatomy` | T023 Body Part, Organ, or Organ Component; T029 Body Location or Region; T030 Body Space or Junction |

**Template Cypher.** Code điền tham số, LLM không viết. Các giới hạn độ sâu là đề xuất, chốt sau khi đo.

```cypher
// Các khái niệm chấm được nằm dưới một khái niệm, gồm chính nó
MATCH (c:Concept {cui: $cui})<-[:IS_A*0..4]-(d:Concept {scorable: true})
RETURN DISTINCT d.cui AS cui;

// Tổ tiên chấm được gần nhất
MATCH p = (c:Concept {cui: $cui})-[:IS_A*1..6]->(a:Concept {scorable: true})
RETURN a.cui AS cui, length(p) AS hops
ORDER BY hops LIMIT 1;

// Các finding chấm được thường gặp ở một vùng giải phẫu, gồm vùng con
MATCH (a:Anatomy {cui: $cui})<-[:PART_OF*0..5]-(:Anatomy)<-[t:TYPICALLY_AT]-(f:Finding {scorable: true})
WHERE t.n >= $min_n
RETURN f.cui AS cui, max(t.p) AS p
ORDER BY p DESC;

// Các finding chấm được gợi ý một bệnh
MATCH (f:Finding {scorable: true})-[s:SUGGESTS]->(:Disease {cui: $cui})
WHERE s.n >= $min_n
RETURN f.cui AS cui, s.p AS p
ORDER BY p DESC;
```

Ràng buộc và index: duy nhất trên `Concept.cui` và `Image.image_id`; fulltext index trên `Alias.text`; index trên `Alias.norm`. Quy mô đo ở bước 4 của lộ trình; KG giới hạn trong miền X-quang ngực nên Neo4j Community là đủ.

## Dựng KG offline

Mỗi build chạy một lần, bằng lệnh `cxr build-kg`; code đặt trong `src/kg/`. Mọi artifact ghi sha256 theo quy ước hiện có của repo.

1. **Lọc UMLS thành Parquet.**
   - Đọc RRF bằng Polars với `separator="|"`, `quote_char=None`, không header. Mỗi dòng có thêm một cột rỗng ở cuối.
   - MRCONSO: giữ `LAT = 'ENG'`, `SUPPRESS = 'N'` và cột `SRL`; build mặc định chỉ lấy `SRL = 0`. MRSTY: giữ cột `STN`. MRREL: giữ các SAB khai trong `configs/kg.yaml`.
   - Ghi kích thước trước và sau khi lọc.
2. **Từ vựng mục tiêu từ RadGraph trên pool.**
   - Join report của pool với RadGraph theo text đã chuẩn hóa (T29).
   - Đếm chuỗi observation (ghép modifier, ví dụ "pleural" với "effusion") và chuỗi anatomy.
   - Đo đường cong độ phủ; lấy phần đầu bảng phủ 90–95% số lần nhắc.
3. **Liên kết chuỗi với CUI.**
   - Khớp chính xác sau chuẩn hóa với MRCONSO.
   - Không khớp thì dùng SapBERT tìm láng giềng gần nhất, chỉ trong các semantic type cho phép.
   - Duyệt tay phần đầu bảng và 14 nhãn CheXpert.
   - Tham khảo pipeline RadGraph-XL + SapBERT → CUI của Nützel và cộng sự (arXiv 2508.19915, có code).
4. **Mở rộng có kiểm soát.**
   - Phạm vi theo miền X-quang ngực, không nạp toàn bộ UMLS:
     - giải phẫu: toàn bộ cây vùng ngực của FMA, theo `PART_OF` và `IS_A`;
     - finding và bệnh: bao đóng `IS_A` lên và xuống của các khái niệm lõi (14 nhãn cộng từ vựng ở bước 2); đi lên thì dừng ở danh sách gốc chặn ("Finding", "Disease"…);
     - thiết bị: các nhánh thiết bị có trong từ vựng ở bước 2 (ống, catheter, dây dẫn, máy tạo nhịp…).
   - Khái niệm gắn bên quy về cha không phân bên.
   - Thêm cạnh phân cấp CheXpert và alias tiếng Việt.
5. **Trọng số thống kê trên pool.**
   - `SUGGESTS`: cặp `suggestive_of` giữa hai observation, quy về cặp CUI (Finding → Disease). `p = n(f → D) / n(f)`, với `n(f)` là số report nhắc f ở trạng thái present. Bỏ cặp có `n` dưới ngưỡng.
   - `TYPICALLY_AT`: từ `located_at`, tính cùng cách.
6. **Nạp Neo4j và kiểm.**
   - Service `neo4j` (Community, bản hiện tại) trong Docker Compose, dưới profile `kg`. Nạp khái niệm và cạnh tri thức bằng `UNWIND` theo batch qua driver.
   - Nạp nút `Image` và cạnh `PREDICTED` (logit đã hiệu chỉnh, `build_id`) từ ma trận điểm. Đây là phần lớn nhất. Nếu `UNWIND` chậm thì dựng toàn bộ KG từ CSV bằng `neo4j-admin database import full`; lệnh này chỉ nạp vào database rỗng.
   - Runtime dùng user chỉ đọc.
   - Build manifest ghi `kg_build_id`, bản UMLS, danh sách SAB, cờ SNOMED, sha256 của file RadGraph, `pool_sha256` và phiên bản alias.
   - `validate-indexes` thêm bốn check:
     - mỗi `Concept` `scorable` có đúng một cột trong ma trận điểm;
     - mỗi nhãn CheXpert ánh xạ đúng một nút;
     - `build_id` của ma trận khớp encoder đang dùng;
     - số cạnh `PREDICTED` bằng số ảnh nhân số khái niệm `scorable`.

**Kiểm chất lượng:**

- Precision của linker trên 200 chuỗi lấy mẫu theo seed.
- 50 cạnh `SUGGESTS` mạnh nhất được người có kiến thức y khoa xác nhận là hợp lý.
- Thống kê số nút, số cạnh theo loại, và số khái niệm không nối được về khái niệm `scorable` nào.

## Truy vấn online

Pipeline cố định; mỗi bước là một hàm Python có test. LLM làm ba việc, đúng như đề bài: phân tích câu hỏi, chọn khái niệm trong ứng viên của KG, và diễn giải kết quả. LLM không quyết định thứ hạng.

1. **Parser** (gpt-5-mini, rule fallback). Giữ `include`, `exclude` trên 14 nhãn như hiện tại, thêm trường `terms` cho cụm khái niệm ngoài 14 nhãn, mỗi cụm có cực tính (có hoặc không). Câu không có `terms` thì đi đúng đường của B1.
2. **`lookup_concept`:** cụm → tối đa 5 CUI ứng viên, lấy theo thứ tự alias khớp chính xác, fulltext, SapBERT. Mỗi ứng viên kèm tên, semantic type và khái niệm cha. LLM chọn một CUI hoặc trả "không có" qua structured output; code kiểm CUI đó nằm trong danh sách ứng viên. Không ứng viên nào đạt ngưỡng thì báo "không nhận ra" và bỏ cụm. Bản tất định (lấy ứng viên đầu khi đạt ngưỡng) là fallback khi LLM lỗi, đồng thời là một dòng ablation.
3. **`expand_concepts`:** CUI → các khái niệm `scorable` kèm trọng số, theo bảng dưới.
4. **`score_images`:** tính trên ma trận logit đã hiệu chỉnh (cách decomposed của `image_concepts.py`, thêm trọng số và phép gộp).
   - Một ý chấm được trực tiếp: `log σ(z_c)`.
   - Một ý rộng: `log(1 − Π(1 − σ(z_c)))` trên các khái niệm con (noisy-OR).
   - Một ý bệnh: `Σ w_c · log σ(z_c)`, với `Σ w_c = 1`.
   - Một ý phủ định: `log(1 − p)`, trong đó p là xác suất của ý đó.
   - Điểm câu = tổng các ý; lọc view bằng metadata; lấy top-k.
5. **`explain_path` và diễn giải:** code trả đường đi trong KG và điểm từng khái niệm cho mỗi ý, ví dụ `suy tim ← SUGGESTS (p) ← tim to`. LLM viết 2–3 câu cho mỗi kết quả, chỉ từ dữ liệu này (RAG trên KG). Code kiểm mọi khái niệm được nhắc tới đều có trong đường đi và chặn từ ngữ chẩn đoán; không qua kiểm thì hiển thị đường đi thô. UI hiển thị diễn giải cùng các cảnh báo.

| Trường hợp | Ví dụ | Quy tắc | Cảnh báo |
| --- | --- | --- | --- |
| Chấm được trực tiếp | tràn dịch màng phổi | Dùng thẳng, trọng số 1 | Không |
| Hẹp hơn | PICC, viêm phổi thùy | Tổ tiên `scorable` gần nhất theo `IS_A` | "Chỉ khớp ở mức \<tổ tiên>" |
| Rộng hơn | mờ phổi | Chính nó cùng các con `scorable`, gộp noisy-OR | Không |
| Vị trí giải phẫu | bất thường màng phổi | Vùng đó và các vùng con theo `PART_OF`, rồi các finding `scorable` qua `TYPICALLY_AT`, gộp noisy-OR | "Khớp theo các finding thường gặp ở \<vùng>; không xét bên" |
| Bệnh không thấy trực tiếp | suy tim | Các finding `scorable` qua `SUGGESTS`, trọng số chuẩn hóa từ `p` | "Suy từ dấu hiệu, không phải chẩn đoán" |
| Không có đường | khái niệm không nối về khái niệm `scorable` nào | Bỏ khỏi điểm | "Không chấm được \<cụm>" |

Quy tắc cho phủ định:

- "Không X" với X rộng: phủ định mọi khái niệm con mà `expand_concepts` trả cho X.
- "Không X" với X hẹp (ví dụ "không có PICC"): không phủ định, chỉ cảnh báo. Hạ về tổ tiên sẽ loại mọi thiết bị, mạnh hơn hẳn điều người dùng hỏi.

**Chọn cách chấm và câu sinh logit.**

- Cách joint (một câu template H1, cosine trên Qdrant) chỉ dùng được khi mọi ý đều chấm được trực tiếp. Chọn joint hay decomposed trên pool-val.
- Logit hiệu chỉnh trên pool-val.
- Ba kiểu câu sinh logit được đo trên pool-val trước khi chốt (P13): template H1, câu kiểu report, cặp dương/âm.

**Ví dụ: "nghi suy tim, không tràn khí, phim PA"**

1. Parser: `terms = [suy tim (có)]`, `exclude = [Pneumothorax]`, `views = [PA]`.
2. `lookup_concept("suy tim")` trả các ứng viên, trong đó có Heart failure (Disease) nhờ alias tiếng Việt; LLM chọn Heart failure.
3. `expand_concepts` ra Cardiomegaly, Edema, Pleural Effusion, trọng số lấy từ `SUGGESTS` đo trên pool.
4. `score_images`: tổng có trọng số của ba logit, cộng `log(1 − p)` của Pneumothorax, lọc PA, lấy top-10.
5. LLM viết diễn giải từ đường đi và điểm, ví dụ: điểm cao cho tim to và phù phổi, hai dấu hiệu mà KG ghi là gợi ý suy tim; điểm tràn khí thấp. Kèm cảnh báo "suy từ dấu hiệu, không phải chẩn đoán".

**Agent (ablation).** Khác bản chính ở chỗ LLM tự quyết gọi tool nào, theo thứ tự nào, và được tra lại bằng cụm diễn đạt khác; tối đa 5 lượt. Danh sách cuối vẫn do code ghép từ các id mà tool đã trả.

## Rào chắn

Nguyên tắc chung: kết quả luôn lấy từ dữ liệu mà tool đã trả, không bao giờ từ lời của LLM.

- **LLM không viết Cypher.** Mọi truy vấn Neo4j là template có tham số, chạy bằng user chỉ đọc và có timeout.
- **LLM chỉ chọn trong ứng viên.** CUI nằm ngoài danh sách mà `lookup_concept` trả bị loại; khi đó lùi về bản tất định.
- **Diễn giải phải có căn cứ.** Mọi khái niệm và con số trong diễn giải phải có trong đường đi và điểm mà tool trả. Câu vi phạm bị bỏ.
- **Không đọc report, không đọc nhãn lúc truy vấn** (D39). Câu trích report để hiển thị (D25) còn để ở câu hỏi mở.
- **Kiểm tham số.** CUI phải có trong KG; cực tính và view kiểm bằng Pydantic như parser hiện tại.
- **Cảnh báo bắt buộc** khi hạ về tổ tiên, khi khớp theo vùng giải phẫu, khi suy từ bệnh, khi bỏ một cụm, và khi câu hỏi có bên hoặc mức độ.
- **Dữ liệu UMLS không vào Git, không vào Docker image.**
- **Không chẩn đoán.** Điểm là độ khớp truy vấn, không phải xác suất bệnh; diễn giải không dùng ngôn ngữ chẩn đoán. Dòng "Research and educational prototype" giữ nguyên.
- **Ghi vết.** Mỗi lượt chạy ghi lại cụm từ, ứng viên, CUI mà LLM chọn, khái niệm sau mở rộng, trọng số, diễn giải và thời gian từng bước. Vết này dùng cho UI và error analysis.

## Kế hoạch đánh giá

Đo bốn lớp riêng để biết lỗi nằm ở đâu.

| Lớp | Câu hỏi | Gold | Metric |
| --- | --- | --- | --- |
| 1. Chất lượng KG | Liên kết và trọng số có đúng không | Duyệt tay mẫu chuỗi và mẫu cạnh | Precision của linker; tỉ lệ cạnh `SUGGESTS`, `TYPICALLY_AT` hợp lý |
| 2. Hiểu câu hỏi | LLM có chọn đúng CUI, và ra đúng tập khái niệm sau mở rộng không | Mỗi câu một tập CUI gold viết tay | Độ chính xác của bước chọn CUI; tỉ lệ khớp hoàn toàn; F1 trên tập khái niệm |
| 3. End-to-end | Top-10 có đúng câu hỏi không | Pooling, LLM chấm trên chữ của report, kiểm bằng kappa với người chấm; thêm nhãn bạc từ RadGraph | P@10, nDCG@10, vi phạm exclude@10 |
| 4. Diễn giải | Diễn giải có trung thực với đường đi và điểm không | Kiểm tự động, cộng người chấm một mẫu | Tỉ lệ câu có căn cứ (mục tiêu 100%); tỉ lệ câu có ngôn ngữ chẩn đoán (mục tiêu 0) |

**Benchmark câu hỏi.** Khoảng 60–80 câu EN/VI, chia dev và heldout. Heldout chạy một lần, có khai báo, như D24. Các nhóm: đồng nghĩa và tiếng Việt, khái niệm hẹp hơn, khái niệm rộng hơn, bệnh không thấy trực tiếp, phủ định khái niệm rộng, vị trí giải phẫu. Thêm một nhóm đối chứng chỉ gồm khái niệm chấm được trực tiếp, để kiểm KG không làm tụt.

**Hệ thống so sánh**

| Hệ thống | Mô tả |
| --- | --- |
| B0 | Dịch thô câu hỏi sang tiếng Anh, rồi CXR-CLIP |
| B1 | Pipeline hiện tại: parser, template H1, CXR-CLIP |
| B2 | B1 cộng KG: LLM chọn khái niệm, KG mở rộng, LLM diễn giải (đề xuất chính) |
| B3 | B2 với agent (ablation) |
| Tham chiếu | Lọc theo nhãn CheXpert rồi xếp bằng CLIP. Có đọc nhãn, nên chỉ dùng để biết mức trần |

Hiệu B2 − B1 là đóng góp riêng của KG; hiệu B1 − B0 là đóng góp của parser.

**Pooling.** Lấy hợp top-10 của mọi hệ thống, chấm relevance trên chữ của report. LLM chấm toàn bộ; người chấm một mẫu để báo Cohen's kappa. Chấm trên report nên không cần bác sĩ đọc phim.

**Ablation.**

- Tra cứu tất định so với LLM chọn ứng viên.
- Tắt từng loại mở rộng: hẹp hơn, rộng hơn, vị trí giải phẫu, bệnh.
- Gộp khái niệm con bằng max so với noisy-OR.
- Trọng số `SUGGESTS` đều nhau so với trọng số từ `p`.
- Build chỉ nguồn mức 0 so với có SNOMED CT, khi MLDS cho phép.
- Ba kiểu câu sinh logit (P13).

**Chi phí.** Ghi số lượt gọi LLM mỗi câu, latency p50 và p95 của từng bước.

## Lộ trình và rủi ro

Bắt đầu bằng probe chỉ đọc. Mỗi bước có cổng dừng; ngưỡng của cổng chốt trước khi đo.

1. **Tải UMLS Level 0 Subset, lọc thành Parquet.** Kiểm MD5. Cổng: cả 14 nhãn CheXpert có CUI hoặc ID dự án.
2. **Probe từ vựng trên pool.** Đường cong độ phủ, ánh xạ 14 nhãn. Cổng: phần đầu bảng phủ 90% đủ nhỏ để duyệt tay.
3. **Linker.** Cổng: precision trên mẫu đạt ngưỡng.
4. **Mở rộng và trọng số.** Đếm nút, cạnh; đo tỉ lệ khái niệm nối được về khái niệm `scorable`. Bước này cho biết nguồn mức 0 có đủ, hay cần SNOMED CT.
5. **Ma trận điểm.** Hiệu chỉnh logit trên pool-val cho mọi khái niệm `scorable`. Chạy song song với bước 2–4, vì không phụ thuộc KG.
6. **Neo4j, bốn tool, LLM chọn khái niệm và diễn giải.** Thử khoảng 20 câu EN/VI; B2 chạy được end-to-end.
7. **Benchmark và pooling.**
8. **Đánh giá chính thức.** Heldout chạy một lần, error analysis theo nhóm câu.
9. **Agent (ablation) và UI** hiển thị đường đi, diễn giải và cảnh báo.

| Rủi ro | Giảm thiểu |
| --- | --- |
| Nguồn mức 0 thiếu phân cấp cho finding X-quang | Đo ở bước 4; thêm cạnh tay; bật SNOMED CT sau khi khai báo MLDS |
| MLDS chậm hoặc có phí | Build mức 0 vẫn chạy đủ; SNOMED CT chỉ là ablation |
| Linker sai, nhất là với chữ viết tắt | Duyệt tay phần đầu bảng; ngưỡng SapBERT; giữ chuỗi gốc |
| LLM chọn sai CUI | Chỉ chọn trong ứng viên; đo ở lớp 2; fallback tất định |
| Diễn giải bịa thêm | Kiểm căn cứ tự động; bỏ câu vi phạm; đo ở lớp 4 |
| Trọng số `SUGGESTS` nhiễu, vì RadGraph là output của model | Ngưỡng `n`; duyệt 50 cạnh mạnh nhất; ablation trọng số đều |
| Phân cấp UMLS không khớp CheXpert | Cạnh `source: 'chexpert'` thêm tay |
| KG không tạo ra tín hiệu mà ảnh không có | Cảnh báo khi hạ về tổ tiên; báo cáo riêng nhóm câu này |
| KG làm tụt câu hỏi đã chấm được | Nhóm đối chứng trong benchmark |
| Vi phạm license khi repo công khai | `.gitignore`; image không chứa KG; chỉ commit mã |

## Câu hỏi mở và ngoài phạm vi

Câu hỏi mở:

- Câu trích report để hiển thị (D25): bỏ khỏi runtime, hay tách thành service chỉ hiển thị và tắt trong mọi lần đánh giá?
- Có cần SNOMED CT không? Quyết sau bước 4.
- Gộp khái niệm con bằng max hay noisy-OR?
- Cụm không nhận ra: bỏ kèm cảnh báo, hay hỏi lại người dùng?
- Diễn giải viết cho từng ảnh, hay một đoạn cho cả trang kết quả?
- Bước chọn khái niệm, diễn giải và agent dùng gpt-5-mini hay một model mạnh hơn?

Ngoài phạm vi V2:

- Lọc study theo dữ kiện trong report lúc truy vấn.
- Vị trí, bên, mức độ, cho tới khi model ảnh học được (V1.2).
- Khoanh vùng trên ảnh.
- Chẩn đoán hoặc đề xuất điều trị.
- Triệu chứng lâm sàng (ho, sốt…): không có tín hiệu trên ảnh, và suy từ triệu chứng sang bệnh là suy luận chẩn đoán.
- Cypher do LLM viết.
- RadLex.

Tham chiếu nội bộ: `WORKLOG.md` (D24, D25, D39, T29, P13), `docs/image-improvement-plan.md` mục 12, kế hoạch V1 §24.

Tham chiếu ngoài: [UMLS 2026AA release](https://www.nlm.nih.gov/pubs/techbull/mj26/mj26_umls_aa_release_notes_old.html) · [UMLS file downloads](https://www.nlm.nih.gov/research/umls/licensedcontent/umlsknowledgesources.html) · [SNOMED CT licensing (NLM)](https://www.nlm.nih.gov/healthit/snomedct/snomed_licensing.html) · [Ontology-Based Concept Distillation (Nützel và cộng sự)](https://arxiv.org/abs/2508.19915) · [CheXpert Plus](https://arxiv.org/html/2405.19538v2)

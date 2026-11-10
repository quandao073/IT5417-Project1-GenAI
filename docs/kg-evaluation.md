# Đánh giá truy vấn qua KG (V2-7, phần V2-8 trên dev)

Kế hoạch đánh giá của thiết kế: `docs/v2-kg-design.md`, *Kế hoạch đánh giá*. Lệnh: `cxr kg-bench-build` (bộ câu và gold),
`cxr eval-kg` (chạy, chấm, tính số). Config: `configs/eval_kg.yaml`. Mọi file trung gian ở `data/eval/` (gitignored).
Số đo gốc kèm cách đo ở `docs/measured-facts.md`. **Heldout chưa chạy** (gold heldout chờ duyệt; heldout chạy một lần, D24).

## Bộ câu và gold

`benchmarks/kg_queries_v1.jsonl`: 72 câu, 8 nhóm × 9, nửa EN nửa VI. Nhóm: control (14 nhãn chấm trực tiếp) · synonym_vi
(đồng nghĩa, cách nói thường, tiếng Việt) · narrower (hẹp hơn một cột) · broader (rộng hơn các cột) · disease (bệnh suy từ
dấu hiệu, gồm bệnh mới có đường ở lượt rà đường) · negation_broad (phủ định khái niệm rộng/hẹp) · anatomy (vùng giải phẫu)
· new_column (23 cột mở rộng). Mỗi nhóm ≥ 2 cặp dịch, ≥ 2 câu có view, ≥ 2 câu có cụm không chấm được; không câu nào lấy từ
`parser_queries_v1.jsonl` hay `docs/v2-6-smoke.txt` (kiểm tự động).

- **Soạn:** `claude-opus-5-5` (effort high, prompt `kg_bench_author_v1.txt`), mỗi nhóm một lượt, nhận danh mục khái niệm của
  graph (595 khái niệm lõi + đầu mút cạnh tay), 37 cột kèm cờ yếu và mã cảnh báo. Gold mỗi câu: `terms[]` (cụm, CUI hoặc
  rỗng, polarity), `expected_scorable` (cột cho phần PRESENT, kèm cờ weak), `exclude`, `views`, `expect_warnings`,
  `query_en` (bản dịch chuẩn, cho judge).
- **Kiểm tự động** (`auto_check`): mọi CUI có trong graph; `expand_concepts` trên các CUI gold, so tập cột với gold. Lệch
  thì ghi, không sửa gold: 31 / 72 câu lệch (lệch là khoảng trống của graph hoặc khác ý giữa gold và graph).
- **Duyệt độc lập:** `claude-sonnet-5-5` (prompt `kg_bench_review_v1.txt`) xét từng câu: gold có đúng nghĩa câu hỏi và hợp
  lý về y khoa không. 1 câu `disagree` → `gold_disputed` (kg-broader-07, lý do: "tại giường" không vào gold). Mọi bảng
  tính hai cách, có và không có câu disputed. Sonnet đồng ý gần hết nên phép duyệt này yếu; gold heldout vẫn chờ Claude
  bên chat duyệt.
- **Chia** dev / heldout 36 / 36, phân tầng theo nhóm (5/4 xen kẽ) và ngôn ngữ, cặp dịch cùng split (seed 20261110).
  sha256 của các dòng heldout: `6b7b47468cf52fa24a5d041c4d62bc65fe7f41241e8c93a9656dfefd02d0eef0`.
- Ghi chú của người soạn và lý do duyệt (có tên khái niệm UMLS) ở `data/eval/bench_notes.jsonl`, không commit.

## Hệ thống

| Hệ thống | Cách chạy (in-process, không API, không Qdrant) |
|---|---|
| B0 | Câu VI dịch thô sang EN bằng `claude-haiku-5-5` (prompt `kg_b0_translate_v1.txt`, cache), câu EN giữ nguyên; text encoder FT6; cosine trên 25.000 vector ảnh; không lọc view |
| B1 | Parser B1 (prompt v1, D61) → `query_text` (template H1) → FT6 → cosine, lọc view theo parser |
| B2 | `KGSearch.search` (parser KG, lookup + chooser, `expand_concepts`, ma trận 37 cột, lọc view) |
| Tham chiếu | **Đọc nhãn, chỉ để biết trần.** Các cột gold là nhãn CheXpert lọc study theo nhãn resolve (PRESENT; loại trừ: không PRESENT, không UNCERTAIN), xếp bằng cosine của `query_en`; câu không có nhãn CheXpert nào thì không lọc |
| B3 | Agent (V2-9): chưa dựng, ghi `planned` |

Cosine trên vector shard bằng đúng phép Qdrant tính (vector chuẩn hóa), nên B1 ở đây là B1 của API trừ sai số làm tròn.

## Chấm (pooling)

- Pool: hợp top-10 của mọi hệ thống và mọi ablation trên mỗi câu.
- Judge `claude-opus-5-5` (effort medium, prompt `kg_judge_v1.txt`) chỉ thấy câu hỏi, `query_en` và FINDINGS / IMPRESSION của
  report; không thấy gold, khái niệm KG hay hệ thống nào trả study. Mỗi lượt gọi chấm tối đa 10 report của một câu (xếp
  theo study id); cache theo (câu, study, model, sha prompt), nên một cặp chỉ chấm một lần dù xuất hiện ở hệ thống nào.
  Thang 2 / 1 / 0 và cờ `violates_exclusion`.
- Chấm chéo `claude-sonnet-5-5`, cùng prompt, 10 % cặp chọn theo seed → tỉ lệ đồng thuận, kappa Cohen (thường và trọng số
  bậc hai) giữa hai máy. `data/eval/human_sample.csv`: 100 cặp cân theo điểm judge, đáp án tách ở
  `human_sample_key.csv`, để người chấm sau.
- Luật view (khai báo trước khi chạy): câu có view thì study khác view tính 0 điểm (report không ghi view).

## Metric

- **Lớp 2 (B2):** độ chính xác chọn CUI (mỗi CUI gold, đúng polarity, có trong intent của B2); khớp hoàn toàn tập cột có dấu
  (+ cột include, − cột exclude); F1 của tập đó; tỉ lệ câu có đủ cảnh báo kỳ vọng và recall cảnh báo.
- **Lớp 3:** P@10 (tỉ lệ điểm 2), P@10 lenient (điểm ≥ 1), nDCG@10 graded (gain 2^g − 1, ideal là pool đã chấm của câu),
  exclude-violation@10 (chỉ câu có phủ định), tỉ lệ câu không có kết quả. Hiệu theo câu B2 − B1, B1 − B0, Tham chiếu − B2,
  CI 95 % bootstrap (10.000 lần, seed 20261110).
- **Lớp 4:** diễn giải hiện tại (`kg_explain_v3`) của B2: tỉ lệ câu có căn cứ, tỉ lệ bị loại, loại vì ngôn ngữ chẩn đoán.
  Đây là mốc "không có tri thức".
- **Chi phí, độ trễ:** chi phí thật cộng từ `usage` (bảng giá trong `configs/eval_kg.yaml`), lượt gọi LLM mỗi câu, USD mỗi
  1.000 câu; p50 / p95 từng bước.

## Kết quả trên dev (2026-11-10)

36 câu dev, 0 câu `gold_disputed` (câu disputed duy nhất nằm ở heldout), nên bảng "không có câu disputed" trùng bảng dưới.
Số máy ghi ở `data/eval/metrics_dev.json`, bảng thô `data/eval/report_dev.md`.

### Judge

Pool 1.000 cặp (36 câu), judge chấm đủ: điểm 0 / 1 / 2 = 367 / 252 / 381. Chấm chéo Sonnet trên 99 cặp (10 % theo seed;
1 cặp Sonnet không trả): đồng thuận 0,798, kappa 0,702, kappa trọng số bậc hai 0,833; đồng thuận "điểm 2 hay không" 0,838.
Hai máy cùng họ, nên đây là độ ổn định của cách chấm, không thay được người chấm: `human_sample.csv` (100 cặp, 34 / 33 / 33
theo điểm judge) chờ người.

### Lớp 3: end-to-end

| Hệ thống | nDCG@10 | P@10 | P@10 (≥ 1) | excl-viol@10 | không kết quả |
|---|---|---|---|---|---|
| B0 | 0,396 | 0,250 | 0,439 | 0,088 | 0 |
| B1 | 0,512 | 0,328 | 0,592 | 0,025 | 0 |
| **B2** | **0,541** | **0,369** | 0,583 | **0,000** | 0,111 |
| Tham chiếu (đọc nhãn) | 0,663 | 0,436 | 0,736 | 0,000 | 0 |

Hiệu theo câu, trung bình [CI 95 % bootstrap] (thắng / thua / hòa):

| Cặp | nDCG@10 | P@10 |
|---|---|---|
| B2 − B1 (đóng góp của KG) | +0,029 [−0,065; +0,114] (18 / 9 / 9) | +0,042 [−0,019; +0,103] (14 / 6 / 16) |
| B1 − B0 (đóng góp của parser) | +0,116 [−0,009; +0,245] (21 / 12 / 3) | +0,078 [−0,008; +0,172] (13 / 9 / 14) |
| Tham chiếu − B2 | +0,122 [0,000; +0,242] (19 / 17 / 0) | +0,067 [−0,031; +0,167] |
| Tham chiếu − B1 | +0,151 [+0,053; +0,251] (22 / 12 / 2) | +0,108 [+0,036; +0,192] |

Trên 36 câu dev, **B2 hơn B1 chưa phân biệt được với 0** (CI chứa 0); B2 không vi phạm exclude nào nhưng không trả kết quả ở
4 câu (11 %). nDCG@10 theo nhóm:

| Nhóm (n) | B0 | B1 | B2 | Tham chiếu |
|---|---|---|---|---|
| control (5) | 0,502 | 0,807 | **0,919** | 0,852 |
| synonym_vi (4) | 0,442 | 0,402 | 0,327 | 0,323 |
| narrower (5) | 0,557 | 0,591 | 0,649 | 0,707 |
| broader (4) | 0,145 | 0,384 | 0,316 | 0,906 |
| disease (5) | 0,175 | 0,511 | **0,157** | 0,512 |
| negation_broad (4) | 0,417 | 0,683 | **0,881** | 0,945 |
| anatomy (5) | 0,548 | 0,193 | 0,384 | 0,578 |
| new_column (4) | 0,335 | 0,513 | **0,710** | 0,480 |

B2 thắng rõ ở control, negation_broad và new_column (các cột mở rộng), thua ở disease và synonym_vi. Mỗi nhóm chỉ 4–5
câu: số theo nhóm là chỉ dẫn, không phải kết luận.

### Lớp 2: hiểu câu hỏi (B2)

| Nhóm | CUI đúng | tập cột khớp hoàn toàn | F1 tập cột | đủ cảnh báo | recall cảnh báo |
|---|---|---|---|---|---|
| tất cả | 0,588 | 0,472 | 0,676 | 0,389 | 0,463 |
| control | 1,000 | 0,800 | 0,933 | 0,800 | 0,667 |
| synonym_vi | 0,167 | 0,250 | 0,250 | 0,500 | 0,500 |
| narrower | 0,400 | 1,000 | 1,000 | 0,400 | 0,600 |
| broader | 1,000 | 0,000 | 0,595 | 0,000 | 0,125 |
| disease | 0,667 | 0,000 | 0,190 | 0,000 | 0,150 |
| negation_broad | 0,667 | 0,500 | 0,833 | 0,750 | 0,875 |
| anatomy | 0,300 | 0,400 | 0,667 | 0,200 | 0,400 |
| new_column | 0,667 | 0,750 | 0,917 | 0,500 | 0,500 |

"CUI đúng" đòi đúng CUI của gold: ở narrower, B2 hay chọn khái niệm cha hoặc anh em mà vẫn ra đúng cột (tập cột khớp
1,000). Kiểm tự động của gold (graph đi từ CUI gold) cũng lệch gold ở 14 / 36 câu dev, nên một phần thua ở broader và
disease nằm ở graph, không ở bước chọn.

### Lớp 4: diễn giải hiện tại (`kg_explain_v3`, mốc "không có tri thức")

36 câu: 30 `ok`, 2 `partial`, 4 `no_results` (không có ảnh để diễn giải). 384 câu diễn giải, 99,48 % có căn cứ, 0,52 % bị
loại (2 câu, cả 2 vì ngôn ngữ chẩn đoán); không câu nào bị loại vì khái niệm hay con số ngoài facts.

### Chi phí và độ trễ

| Hệ thống | lượt LLM mỗi câu | USD mỗi 1.000 câu | độ trễ (p50 / p95) |
|---|---|---|---|
| B0 | 0,47 (chỉ câu VI) | 0,02 | dịch 0 / 1,71 s; encode 23 / 30 ms; cosine 1 ms |
| B1 | 1,06 | 0,15 | parse 1,46 / 2,05 s; encode 20 / 25 ms; cosine 1 ms |
| B2, tìm | 1,75 | 0,28 | tổng 3,08 / 4,25 s (parse 1,68 / 2,53 s; term + chooser 1,33 / 2,24 s; chấm điểm < 1 ms) |
| B2, diễn giải | 0,94 | 0,40 | 2,63 / 4,20 s |

N1 (< 5 s chưa tính diễn giải) đạt trên dev: p95 4,25 s. Chi phí judge không thuộc hệ thống: 2,84 USD cho 1.000 cặp.

## Ablation trên dev (V2-8)

Cùng parse với B2 (các ablation tìm lại chạy chung process, dùng cache parser); các ablation còn lại chấm lại intent của
B2. Cặp mới vào pool được chấm bằng cùng judge. nDCG@10, hiệu với B2 [CI 95 %] (thắng / thua / hòa):

| Cấu hình | tất cả | hiệu với B2 | control | synonym_vi | narrower | broader | disease | negation | anatomy | new_column |
|---|---|---|---|---|---|---|---|---|---|---|
| B2 | 0,541 | — | 0,919 | 0,327 | 0,649 | 0,316 | 0,157 | 0,881 | 0,384 | 0,710 |
| lookup tất định (không chooser) | 0,571 | +0,030 [−0,017; +0,093] (3/2/31) | 0,919 | 0,361 | 0,649 | 0,316 | 0,424 | 0,880 | 0,309 | 0,710 |
| SapBERT tắt | 0,533 | −0,008 [−0,037; +0,012] (2/1/33) | 0,919 | 0,215 | 0,649 | 0,316 | 0,186 | 0,882 | 0,384 | 0,710 |
| tắt hẹp hơn | 0,487 | −0,054 [−0,111; −0,008] (0/5/31) | 0,919 | 0,215 | 0,521 | 0,316 | 0,103 | 0,881 | 0,267 | 0,710 |
| tắt rộng hơn | 0,488 | −0,053 [−0,126; −0,003] (0/4/32) | 0,919 | 0,327 | 0,649 | 0,000 | 0,157 | 0,722 | 0,384 | 0,710 |
| tắt vị trí | 0,541 | 0 (0/0/36) | | | | | | | | |
| tắt bệnh | 0,526 | −0,015 [−0,044; 0,000] (0/2/34) | 0,919 | 0,327 | 0,649 | 0,316 | 0,054 | 0,877 | 0,384 | 0,710 |
| gộp max thay noisy-OR | 0,547 | +0,006 [0,000; +0,019] (2/0/34) | 0,919 | 0,327 | 0,649 | 0,316 | 0,157 | 0,937 | 0,384 | 0,710 |
| SUGGESTS đều | 0,541 | 0 (0/0/36) | | | | | | | | |

- Mở rộng **hẹp hơn** và **rộng hơn** là hai đóng góp đo được của KG (tắt mỗi loại làm tụt có CI dưới 0).
- **Vị trí** không đổi gì trên dev: không câu dev nào giữ lại intent vùng (vùng đi cạnh dấu hiệu thì chỉ cảnh báo, D67);
  `anatomy_top` vì vậy không có gì để chỉnh trên dev.
- **SUGGESTS đều** không đổi gì: các bệnh của dev hoặc chỉ có một cột, hoặc không ra intent bệnh.
- **Lookup tất định** hơn B2 ở nhóm disease (0,424 so với 0,157) nhưng gần như do tình cờ: lấy ứng viên đầu (thường là
  giải phẫu "phổi") cho cụm tiếng Việt không có alias, trong khi chooser trả `none`. Không đổi luật chooser vì một nhóm 5 câu.
- Tham số khai báo được chỉnh trên dev, giữ nguyên: `anatomy_top` 3, `lookup.sapbert.min_cos` 0,8, chooser bỏ qua LLM khi
  có đúng một ứng viên exact (SapBERT tắt chỉ lệch −0,008, không có tín hiệu để chỉnh `min_cos`).

## Error analysis: mọi câu B2 thua B1 trên dev (9 câu)

| Câu | Bước sai | nDCG B2 / B1 | Chuyện gì xảy ra |
|---|---|---|---|
| kg-control-07 | mở rộng | 0,874 / 0,882 | loại Fracture theo noisy-OR cha + 2 con (D63) thay vì chỉ cột cha; lệch nhỏ |
| kg-synonym_vi-01 | parse/lookup | 0,446 / 0,475 | "scarring in the lungs" → chooser chọn C0034069 (xơ phổi) thay vì cột sẹo C2004491 |
| kg-synonym_vi-02 | parse/lookup | 0,000 / 0,524 | "air tracking under the skin" không ra ứng viên tràn khí dưới da; chooser trả `none` |
| kg-broader-03, -04 | mở rộng | 0,158 / 0,158; 0,104 / 0,457 | "chấn thương" chỉ mở rộng xuống rib fracture; gold muốn gãy xương, tràn khí, tràn khí dưới da (graph không có các cạnh đó) |
| kg-disease-01 | mở rộng | 0,117 / 0,861 | xơ phổi C0034069 leo `IS_A` tới `PRJ:pleural_other` qua cụm "fibrosis" của CheXpert (D57), không tới các cột mô kẽ |
| kg-disease-02 | parse/lookup | 0,000 / 0,284 | "xơ phổi" (VI) không có alias; chooser trả `none` |
| kg-disease-03 | mở rộng | 0,516 / 0,525 | bóc tách động mạch chủ suy từ widened mediastinum (cạnh tay), gold thêm Enlarged Cardiomediastinum (cột yếu) |
| kg-disease-04 | parse/lookup | 0,000 / 0,756 | "bóc tách động mạch chủ" (VI) không có alias; chooser trả `none` |

Theo bước: mở rộng 5, parse/lookup 4 (chọn sai ứng viên 0 lần ở dạng "có trong danh sách mà không chọn"). Hai nguyên nhân
lặp lại: (1) **thiếu alias tiếng Việt** cho bệnh (alias VI chỉ có cho 14 nhãn và 200 khái niệm hay gặp nhất), nên cặp dịch
EN thắng mà VI trắng; (2) **cạnh của graph** cho bệnh và khái niệm rộng (xơ phổi → Pleural Other qua D57; chấn thương chỉ
tới rib fracture). Cả hai là việc của KG, không phải của ranking.

## Chạy heldout

Chưa chạy. Điều kiện: gold heldout đã được Claude bên chat duyệt (sửa gold heldout thì ghi lại sha256 mới vào WORKLOG
trước khi chạy). Chạy một lần, có khai báo; mọi hệ thống, mọi ablation đã khai báo trong `configs/eval_kg.yaml`
(`lookup_deterministic`, `sapbert_off`, `no_narrower`, `no_wider`, `no_anatomy`, `no_disease`, `combine_max`,
`suggests_equal`), cùng judge, cùng prompt, cùng tham số như dev:

```bash
uv run cxr eval-kg --split heldout --heldout-declared
```

Ước chi phí (theo dev): các lượt Haiku của pipeline và B0 < 0,1 USD; judge Opus ≈ 3 USD (≈ 0,0026 USD mỗi cặp); chấm chéo
Sonnet ≈ 0,2 USD; tổng ≈ 3,5 USD. Cần Neo4j với `kg_build_id` kg_1637d5c0ee06 và `ANTHROPIC_API_KEY`. Kết quả ở
`data/eval/metrics_heldout.json`, `report_heldout.md`.

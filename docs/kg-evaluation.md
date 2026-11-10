# Đánh giá truy vấn qua KG (V2-7, V2-8)

Kế hoạch đánh giá của thiết kế: `docs/v2-kg-design.md`, *Kế hoạch đánh giá*. Lệnh: `cxr kg-bench-build` (bộ câu và gold),
`cxr eval-kg` (chạy, chấm, tính số). Config: `configs/eval_kg.yaml`. Mọi file trung gian ở `data/eval/` (gitignored).
Số đo gốc kèm cách đo ở `docs/measured-facts.md`. **Heldout đã chạy một lần (D24, D75), xem mục *Heldout*.** Phần *Dev lượt 2* là trạng thái dev hiện tại; các phần trước là lượt 1
(36 câu dev, trước D72–D74).

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

## Dev lượt 2 (phiên C, 2026-11-10): 72 câu, ma trận probe, KG đã vá, trộn điểm

**Dev đã được dùng để chỉnh** (error analysis lượt 1 → alias VI, cạnh xơ phổi và chấn thương, fallback văn bản; lưới trộn
điểm chọn trên chính dev này), nên mọi số dưới đây **lạc quan**; số để báo cáo là heldout. Cấu hình: ma trận
`..._ext1_probe1` (D72), graph kg_32e035ed37b0 (D73), gold dev sha256 `cae72e64…` (`scripts/kg_bench_apply_review.py`), cùng
judge Opus, cùng prompt `kg_judge_v1`, cùng cache với lượt 1. Pool 2.986 cặp (0/1/2 = 1.271 / 788 / 927); chấm chéo Sonnet 299
cặp: đồng thuận 0,789, kappa 0,680, kappa bậc hai 0,838. Khai báo trước khi đo: D74 (commit `b0ec9f5`).

### Lớp 3 theo nhóm (nDCG@10)

| nhóm | n | B0 | B1 | B2 | B2 + trộn (b0, w 0,2) | Tham chiếu |
|---|---|---|---|---|---|---|
| tất cả | 72 | 0,396 | 0,419 | 0,470 | **0,503** | 0,625 |
| control | 9 | 0,448 | 0,735 | 0,766 | 0,774 | 0,867 |
| synonym_vi | 9 | 0,391 | 0,371 | 0,408 | 0,382 | 0,430 |
| narrower | 9 | 0,590 | 0,466 | 0,428 | 0,489 | 0,709 |
| broader | 9 | 0,327 | 0,266 | 0,322 | 0,343 | 0,707 |
| disease | 9 | 0,317 | 0,435 | 0,394 | 0,402 | 0,495 |
| negation_broad | 9 | 0,287 | 0,485 | 0,532 | 0,544 | 0,740 |
| anatomy | 9 | 0,475 | 0,256 | 0,303 | 0,442 | 0,636 |
| new_column | 9 | 0,336 | 0,341 | 0,608 | 0,647 | 0,420 |

P@10: B0 0,233 · B1 0,264 · B2 0,292 · Tham chiếu 0,385. excl-viol@10: 0,184 · 0,037 · 0,079 · 0,063. Không kết quả: 0 ở mọi
hệ; B2 đi đường văn bản (D73) ở 8 / 72 câu. Bỏ 2 câu `gold_disputed`: B2 0,480, B2 + trộn 0,513.

- **B2 − B1** nDCG@10 +0,051 [−0,010; +0,112] (35 thắng / 21 thua / 16 hòa): chưa tách khỏi 0.
- **B2 + trộn − B1** +0,083 [+0,017; +0,149]: tách khỏi 0. B1 − B0 +0,023 [−0,069; +0,113]; Tham chiếu − B2 +0,155 [+0,071; +0,241].
- So với lượt 1 trên đúng 36 câu dev cũ: B2 0,541 → 0,544, B1 0,512 → 0,482 (parse LLM không tái lập, T18; judge thêm cặp
  làm đổi pool lý tưởng), B2 − B1 +0,029 → +0,063 [+0,012; +0,114]. Trên 36 câu dev mới: B2 − B1 +0,039 [−0,070; +0,147].

### Trộn điểm (D74)

Hiệu nDCG@10 so với B2 (w = 0), CI 95% paired, 10.000 lần bootstrap:

| w | câu mẫu B1 (b1) | EN thô B0 (b0) |
|---|---|---|
| 0,1 | +0,011 [−0,002; +0,025] | +0,027 [+0,008; +0,049] |
| 0,2 | +0,012 [−0,006; +0,030] | **+0,033 [+0,006; +0,062]** |
| 0,3 | +0,013 [−0,005; +0,032] | +0,028 [−0,004; +0,062] |
| 0,5 | +0,005 [−0,018; +0,029] | +0,010 [−0,031; +0,050] |
| 1 | +0,000 [−0,032; +0,034] | +0,008 [−0,044; +0,060] |

Luật khai báo trước giữ **b0, w = 0,2** (hơn w = 0 ≥ 0,005 và CI không chứa 0). Trộn giúp nhất ở anatomy (+0,139) và
narrower (+0,061), làm hại synonym_vi (−0,026). Nguồn b0 cần một lượt dịch Haiku cho câu VI lúc truy vấn (p95 1,71 s ở lượt 1), nên khi đưa vào API phải chạy song song với parser.

### Ablation (hiệu nDCG@10 với B2)

tắt bệnh −0,040 [−0,078; −0,005]; tắt rộng hơn −0,012 [−0,031; 0]; tắt vị trí −0,011 [−0,033; 0]; tắt hẹp hơn −0,002
[−0,012; +0,007]; SapBERT tắt −0,007 [−0,020; 0]; lookup tất định +0,008 [−0,020; +0,039]; gộp max +0,001; SUGGESTS đều 0;
**probe_off** (intents của B2 chấm trên ma trận zero-shot) −0,004 [−0,034; +0,022]: ma trận probe chưa cho lợi đo được
trên xếp hạng dev, dù AUROC từng cột tăng (D72).

### Lớp 2 và lớp 4

Lớp 2 (B2, 72 câu): CUI đúng 0,692 (lượt 1 0,588 trên 36 câu), F1 tập cột 0,723 (0,676), đủ cảnh báo kỳ vọng 0,704 (0,463).
Theo nhóm (CUI / F1 cột): control 1,000 / 1,000; synonym_vi 0,500 / 0,556; narrower 0,778 / 0,611; broader 0,818 / 0,771;
disease 0,917 / 0,799; negation_broad 0,688 / 0,663; anatomy 0,375 / 0,667; new_column 0,571 / 0,719. Lớp 4
(`kg_explain_v4`, knowledge bật): 56 ok, 7 partial, 9 không diễn giải (8 câu đi đường văn bản + 1); 759 câu, 98,95 % có căn cứ,
1 câu bị loại vì từ chẩn đoán; không câu nào còn bị loại vì markup (D73 e).

### Error analysis lượt 2: 21 câu B2 thua B1

ranking 8 (control-07/10/11, narrower-03/04/12, broader-04, negation_broad-03: cột đúng, thứ hạng kém hơn caption B1),
parse/lookup 5 (synonym_vi-01/02, narrower-13, broader-19, negation_broad-12), expansion 5 (disease-01/05/10/11,
negation_broad-16: SUGGESTS của bệnh chọn cột khác gold), choice 3 (narrower-02, anatomy-10, new_column-16). Không sửa KG
thêm theo lượt này.

## Heldout: khai báo (D24, D75), ghi trước khi chạy

Cấu hình khóa: B2 = ma trận `..._ext1_probe1` + graph kg_32e035ed37b0 (`matrix_version` `..._ext1_probe1@e878b151d823`) +
fallback văn bản + trộn b0, w = 0,2; `explain.knowledge` bật (prompt v4). Commit code của lượt chạy: commit khóa này. sha256
(bản trong working copy lúc chạy):

```
b9e7eccbefecfda8371f6041e7d18a3097c2c8ce63655eea07337f8853518348  dòng heldout (kb.split_sha256, 100 câu)
b3ce4dcef6597a183cbb278c45285a3bb2345350843ece1fca6887830ccf800e  benchmarks/kg_queries_v1.jsonl
6afff35044156d7739291722ed1e53fed19fc0ad7c0948eb024af1a2e75ed6aa  configs/eval_kg.yaml
47e87f860f26544a105d5d002707ea975215b8ccbdad715a9aa738d6d23ee41c  configs/kg.yaml
1e91735d6b55fec3936b8aac081ee39f05043e7fb3269f861406f3ad8148a0e8  configs/models.yaml
e0cacc2cb6634ad6b99c8c02474b324382c1f8ed34067d3ba07c538f30bcfdad  configs/concepts.yaml
1a1ed039b55c4dd5bc5bd66b7090f5746bb7dd8887cb87382ca349e2d909db5f  configs/kg/aliases_manual.yaml
b93212d8dc875d8050dd3eeb0d8337321d601f997c17ef7e0a08e2d6ac8254cd  configs/kg/chexpert_cuis.yaml
8280508edc18ab35a49f7c957ca0f9d1a7b3babdd573a79285c5a7e5886273f6  configs/kg/linker_overrides.yaml
9e798cc5fe0d1fc82768098891c6a4071c12981729e8055b3581718c83b86918  configs/prompts/kg_judge_v1.txt
d22db8aeab4931f9056b5cc7c84bb56eec7ac77fa0366af51ec2cdcc2b3f99ba  configs/prompts/kg_b0_translate_v1.txt
8f647955ca123d19b52091e44e22ab5657b8475dae1f2ef6799a28d975e88a8f  configs/prompts/query_parser_v1.txt
38e161cee59c0abfadc4d9e8ef4bce750fcf45414fab8992a4fbc323210d59e0  configs/prompts/query_parser_v3.txt
7ae3b5452940a6988866417df574dc0aa2eff1b89bd3eb3c64a88135578a7058  configs/prompts/kg_choose_concept_v2.txt
83610c273291d72867347544519aa69e3d8ffbb39bdd8b2580eb3692f9586bcd  configs/prompts/kg_explain_v3.txt
a7e51fbbcea9a4c7a075784ec6571b0f995a373761e0c02b3cd8fae3ac87276d  configs/prompts/kg_explain_v4.txt
```

Cùng lượt: B0, B1, Tham chiếu, `fusion_off` (= B2 không trộn), `probe_off`, mọi ablation của `eval.ablations`, lưới 10 ô trộn
điểm. **Lưới trên heldout chỉ là phân tích độ nhạy: kết quả chính luôn là cấu hình đã khóa, kể cả khi ô khác cao hơn.**
Lớp 4 đo cả có (v4) và không có knowledge (v3, cùng kết quả tìm). Chạy một lần; sau đó không sửa gì theo heldout.

## Heldout (2026-11-10, chạy một lần theo khai báo D75)

**Kết quả chính là cấu hình đã khóa B2 + trộn (b0, w = 0,2).** Lưới trộn bên dưới chỉ là phân tích độ nhạy: kết quả chính
luôn là cấu hình đã khóa, kể cả khi một ô khác cao hơn. Sau lượt này không sửa gì theo heldout.

Chạy: `uv run cxr eval-kg --split heldout --heldout-declared` (pha run, judge, score), commit `3d9d551`, `kg_build_id`
kg_32e035ed37b0, `matrix_version` `..._ext1_probe1@e878b151d823`. 100 câu (49 EN / 51 VI), 0 câu `gold_disputed`.
23 hệ trong cùng một lượt: B0, B1, B2 (= `fusion_off`), Tham chiếu, 9 ablation, lưới 10 ô. Judge là Opus `kg_judge_v1`,
giống dev. Pool có 4.602 cặp (0/1/2 = 2.204 / 1.154 / 1.244). Sonnet chấm chéo 458 cặp: đồng thuận 0,817, kappa 0,708,
kappa bậc hai 0,852, đồng thuận nhị phân "liên quan" 0,893.

### Lớp 3 theo nhóm (nDCG@10)

| nhóm | n | B0 | B1 | B2 | **B2 + trộn (khóa)** | Tham chiếu |
|---|---|---|---|---|---|---|
| tất cả | 100 | 0,355 | 0,359 | 0,482 | **0,497** | 0,537 |
| control | 13 | 0,493 | 0,521 | 0,627 | 0,615 | 0,763 |
| synonym_vi | 12 | 0,347 | 0,415 | 0,409 | 0,432 | 0,513 |
| narrower | 13 | 0,225 | 0,155 | 0,319 | 0,326 | 0,343 |
| broader | 12 | 0,439 | 0,501 | 0,606 | 0,593 | 0,603 |
| disease | 13 | 0,295 | 0,181 | 0,311 | 0,369 | 0,474 |
| negation_broad | 12 | 0,518 | 0,523 | 0,659 | 0,708 | 0,736 |
| anatomy | 13 | 0,237 | 0,380 | 0,539 | 0,538 | 0,441 |
| new_column | 12 | 0,299 | 0,211 | 0,398 | 0,405 | 0,434 |

P@10: B0 0,206 · B1 0,235 · B2 0,303 · B2 + trộn 0,320 · Tham chiếu 0,351.
excl-viol@10: 0,046 · 0,054 · 0,054 · 0,069 · 0,035.
Không hệ nào trả về rỗng. B2 đi đường văn bản (D73) ở 12 / 100 câu.

### Hiệu (paired, CI 95%, 10.000 lần bootstrap, seed 20261110; thắng/thua/hòa)

| cặp | nDCG@10 | P@10 |
|---|---|---|
| **B2 + trộn − B1** | **+0,138 [+0,078; +0,198]** (67/28/5) | +0,085 [+0,038; +0,134] |
| B2 − B1 | +0,123 [+0,060; +0,188] (62/30/8) | +0,068 [+0,016; +0,121] |
| B1 − B0 | +0,004 [−0,046; +0,055] (44/41/15) | +0,029 [−0,011; +0,071] |
| Tham chiếu − B2 + trộn | +0,040 [−0,028; +0,107] (54/40/6) | +0,031 [−0,022; +0,082] |
| B2 + trộn − B2 (`fusion_off`) | +0,015 [−0,018; +0,045] (46/38/16) | +0,017 [−0,007; +0,041] |

- Trên heldout, cả B2 lẫn B2 + trộn đều hơn B1 và CI tách khỏi 0. Mức hơn lớn hơn trên dev (dev: +0,051 và +0,083).
- B1 không hơn B0.
- Khoảng cách tới Tham chiếu chưa tách khỏi 0. Riêng nhóm anatomy, B2 cao hơn Tham chiếu.
- Lợi của trộn trên B2 là +0,015, CI chứa 0: nhỏ hơn trên dev (+0,033 [+0,006; +0,062]).

### Ablation (hiệu nDCG@10 với B2, không trộn)

| ablation | hiệu |
|---|---|
| **`probe_off`** (intents B2 chấm trên ma trận zero-shot) | **−0,071 [−0,118; −0,027]** (29/50/21) |
| `no_anatomy` | −0,032 [−0,058; −0,011] |
| `no_disease` | −0,029 [−0,055; −0,007] |
| `no_narrower` | −0,026 [−0,053; −0,005] |
| `lookup_deterministic` | −0,021 [−0,045; −0,002] |
| `combine_max` | −0,004 [−0,015; +0,006] |
| `no_wider` | −0,002 [−0,006; 0] |
| `suggests_equal` | 0 |
| `sapbert_off` | +0,014 [−0,019; +0,049] |

- Ma trận probe (D72) giúp rõ trên heldout, dù trên dev chưa đo được lợi (dev −0,004).
- Các route anatomy, disease, narrower và lookup có chọn bằng LLM đều có lợi với CI không chứa 0.
- Tắt SapBERT không làm kém đi.

### Lưới trộn điểm: chỉ phân tích độ nhạy

Hiệu nDCG@10 so với w = 0 (= B2), CI 95% paired. **Kết quả chính luôn là ô đã khóa (b0, w = 0,2), kể cả khi ô khác cao
hơn.**

| w | câu mẫu B1 (b1) | EN thô B0 (b0) |
|---|---|---|
| 0,1 | +0,000 [−0,020; +0,020] | +0,009 [−0,015; +0,030] |
| 0,2 | −0,005 [−0,033; +0,020] | **+0,015 [−0,018; +0,045]** (khóa) |
| 0,3 | −0,018 [−0,051; +0,012] | +0,006 [−0,032; +0,043] |
| 0,5 | −0,032 [−0,069; +0,003] | −0,013 [−0,059; +0,031] |
| 1 | −0,062 [−0,105; −0,020] | −0,048 [−0,100; +0,002] |

- Ô đã khóa cũng là ô cao nhất trên heldout, nhưng CI của mọi ô đều chứa 0, trừ b1, w = 1 (làm hại).
- w lớn kéo kết quả về phía B1/B0.
- Trộn giúp ở disease (+0,058) và negation_broad (+0,049), làm hại nhẹ ở control (−0,012) và broader (−0,013).

### Lớp 2 (B2, 100 câu)

Toàn bộ: CUI đúng 0,654, tập cột đúng 0,510, F1 tập cột 0,713, đủ cảnh báo kỳ vọng 0,660, recall cảnh báo 0,669.

| nhóm | CUI đúng | F1 cột |
|---|---|---|
| control | 1,000 | 1,000 |
| synonym_vi | 0,579 | 0,622 |
| narrower | 0,600 | 0,628 |
| broader | 0,412 | 0,609 |
| disease | 0,786 | 0,758 |
| negation_broad | 0,609 | 0,764 |
| anatomy | 0,588 | 0,500 |
| new_column | 0,565 | 0,822 |

`lookup_deterministic`: CUI 0,621, F1 0,653 (synonym_vi chỉ còn 0,368 / 0,372). `sapbert_off`: CUI 0,660, F1 0,696.

### Lớp 4: diễn giải, có và không có knowledge (cùng kết quả tìm)

| | ok / partial / tắt | câu | có căn cứ | bị loại |
|---|---|---|---|---|
| v4, knowledge bật (cấu hình khóa) | 82 / 6 / 12 | 1.064 | 99,44 % | 6 |
| v3, không knowledge | 85 / 3 / 12 | 1.058 | 99,72 % | 3 |

- 12 câu "tắt" là 12 câu đi đường văn bản: khi đó không có diễn giải (D73).
- Lý do bị loại ở v4: 3 thiếu trích dẫn, 2 khái niệm không có trong facts, 1 con số không có trong mục được trích. Không câu
  nào bị loại vì từ chẩn đoán.
- Ở v3: 2 câu bị loại vì từ chẩn đoán, 1 vì con số.
- Knowledge không làm tăng tỉ lệ có căn cứ: hai bản chênh 0,3 điểm, trên ~1.060 câu, nên chưa phân biệt được.

### Chi phí và độ trễ (heldout)

**Chi phí** (USD trên 1.000 câu hỏi):

| bước | USD / 1.000 câu | lượt gọi LLM / câu |
|---|---|---|
| B0 | 0,036 | 0,86 |
| B1 | 0,395 | 2,83 |
| B2, tìm | 0,705 | 4,25 |
| B2, diễn giải | 1,754 | 3,39 |

Lượt heldout tốn ≈ 14,3 USD (run 0,14; judge 13,45; chấm chéo 0,70).

**Độ trễ p50 / p95:**

| hệ | bước | p50 (ms) | p95 (ms) |
|---|---|---|---|
| B2 | tổng | 2.472 | 4.469 |
| B2 | parse | 1.898 | 2.462 |
| B2 | terms | 163 | 2.313 |
| B2 | chấm | 0,4 | 36 |
| B2 | diễn giải (luồng riêng) | 3.476 | 4.701 |
| B1 | parse | 1.673 | 2.387 |
| B0 | dịch | 1.049 | 1.933 |

Muốn dùng cấu hình khóa trong API, phải thêm lượt dịch B0 cho câu VI và chạy nó song song với parser của B2. Code trong
API hiện chưa trộn điểm.

### Error analysis heldout: 30 câu B2 thua B1 (không sửa gì theo phần này)

| loại | số câu | các câu và diễn giải |
|---|---|---|
| ranking | 9 | control-06/08/18/19, narrower-06, disease-06/12/16, new_column-08. Cột đúng với gold, nhưng thứ hạng kém caption B1. Ở control-08, B1 đạt 1,0 còn B2 0,626 |
| parse/lookup | 12 | synonym_vi-04/12/16 (B2 không nhận ra term VI nào; xem lại ở dưới), narrower-16 (chọn sai cột: C0016655 thay C0016658), broader-12/13 (chỉ lấy cột chung, bỏ các khái niệm con gold muốn), negation_broad-07/08, anatomy-20/22, new_column-15/19 |
| expansion | 4 | narrower-17, broader-05, disease-09, anatomy-21. Graph không có đường từ term tới cột gold, hoặc đường đi tới cột khác |
| choice | 5 | narrower-21, broader-15, disease-18 (đảo cực: cột gold PRESENT bị chọn thành ABSENT), negation_broad-09, new_column-06 |

**Ba câu synonym_vi trả nDCG 0:** B2 không có intent nào. Alias VI chỉ được quét trên 63 bệnh có trong pool (D73), nên
chỗ hổng này đúng như đã biết.

**Lỗi tìm thấy sau lượt chạy (chưa sửa): đường văn bản mã hóa chuỗi `""`.**

- Khi parser không nhận ra khái niệm nào, nó trả `semantic_query` là chuỗi `""`, tức đúng hai dấu nháy.
- `TextFallback.text` mã hóa nguyên chuỗi đó. Kết quả đường văn bản vì vậy gần như ngẫu nhiên theo câu hỏi.
- Gặp ở 9 / 12 câu fallback của heldout và 5 / 9 câu của dev (`fallback_text` trong `runs_*.json`). Lỗi đã có mặt từ dev lượt
  2, nhưng không được phát hiện trước khi khóa cấu hình.
- Số heldout ở trên là số của hệ còn lỗi này. Ảnh hưởng chỉ nằm ở 9 câu đó; chưa đo xem sửa xong thì
  số thay đổi bao nhiêu.
- Cách sửa đề xuất: khi `semantic_query` rỗng hoặc chỉ là dấu nháy, dùng câu mẫu B1 hoặc bản dịch EN của câu hỏi. Đây là
  việc cho lượt sau và cần đo lại trên dev; không đo lại heldout.

**Lỗ hổng KG đã biết từ duyệt gold heldout, chưa sửa** (theo D24, chỉ sửa KG theo dev):

- Tràn khí trung thất đi IS_A tới C0013990, không tới cột gold.
- Hạch vôi hóa không tới cột vôi hóa.
- C1881236, C3809515 và C2102813 không có đường tới cột nào.
- `auto_check` lệch gold ở 18 câu heldout (55 / 172 trên cả hai split).
- Ở broader-05 và anatomy-22, gold muốn bộ ba cột kẽ phổi C1849561 / C4476748 / C5539772. B2 không ra cột nào, vì từ term
  của câu không có đường tới bộ ba này.

### Giới hạn

1. **Dev đã dùng để chọn cấu hình.** Gồm: vá KG (alias VI, cạnh xơ phổi và chấn thương), fallback văn bản, λ của probe trên
   pool-val, và lưới trộn. Vì vậy số dev lạc quan. Số heldout là số báo cáo, nhưng lựa chọn "trộn b0, w = 0,2" chỉ dựa trên
   72 câu dev.
2. **Parser LLM không tái lập được** (T18): cùng câu, cùng prompt, kết quả có thể khác giữa hai lượt. Ví dụ: trên đúng 36
   câu dev cũ, B1 0,512 (lượt 1) → 0,482 (lượt 2), và B2 0,541 → 0,544. Mọi hiệu ở trên vì thế còn mang phương sai của
   parser, ngoài phương sai bootstrap đã báo.
3. Hai giới hạn của probe (D72): encoder FT6 đã thấy pool-train; λ và cổng cùng chọn trên pool-val.
4. Judge là LLM. Kappa với Sonnet là 0,708, nhưng chưa có mẫu người chấm (P9; mẫu 100 cặp đã rút ở
   `data/eval/human_sample.csv`).

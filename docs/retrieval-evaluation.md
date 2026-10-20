# Retrieval ba mode — Phase 5

**Ngày:** 2026-10-21 · **Git:** heldout lần 1 ở `a1361db` (commit chốt fusion), heldout lần 2 (số trích dẫn) ở `47c2943`; lưới dev ở `241881e` · **build_id:** `20261011_chexpert25k_biomedclip_v1` · **Report index `corpus_sha256`:** `60b72389ef6e56579e7d34a958e9fa22b5b3c27e339658e08cda695e923c3ee8` · **Parser `auto`:** OpenAI `gpt-5-mini-2025-08-07`, prompt_sha256 `c975e4ca2e2172d376915e17926ab8c28a9110876186e600bb178124e4caac35`

Nguồn của mọi số dưới đây (đều gitignored). Mỗi lần `cxr eval-retrieval` ghi đè
`artifacts/metrics/retrieval_eval.json`, nên heldout được đóng băng:

- `artifacts/metrics/retrieval_eval_heldout_20261021_run2.json`, sha256
  `e61862395586d30b575ebe386f9b14d8b73227fb56823d54945eefd26b103969`: heldout **lần 2**, lần được
  trích dẫn.
- `artifacts/metrics/retrieval_eval_heldout_20261021.json`, sha256
  `b35a6b762a9d84f8baabe2c9c66df6ac3cb8f52bc8466f438054adabf93befdd`: heldout lần 1, giữ lại. Vì sao
  có hai lần: *Cách đo*.
- `artifacts/metrics/retrieval_dev.json` (section `grid` và `eval` của dev), sha256
  `b56c0d34fd3888a54c482b7e1b9b5158e9e48bfb7a10c454931cda3fd2385a2a`.
- HTTP: `vector_baseline_phase5_{A_vector,A_hybrid,B_hybrid}.json`. API smoke: `smoke_phase5.json`.
  Lọc Qdrant: `qdrant_filter_probe.json`.

**Kết luận:** heldout **PASS** (parser `gold`, policy mặc định). Ba ngưỡng của
`configs/evaluation.yaml:thresholds`: p@10 của hybrid trên nhóm `single` 1,000 (ngưỡng 0,70);
constraint satisfaction = p@10 của hybrid 1,000 (0,95); `(nDCG_hybrid − nDCG_vector) / nDCG_vector`
= (0,607 − 0,235) / 0,235 = **1,579** (0,05). Hai câu hỏi không có ngưỡng nhưng là câu hỏi thật của
luận văn:

| Câu hỏi | Số (gold, heldout) | Đọc |
|---|---|---|
| Thứ hạng của hybrid có ích hơn thứ tự cố định của label-only không? | nDCG@10 **0,607** so với **0,531**; nDCG nhị phân 1,000 ở cả hai | Có, theo proxy graded: trên cùng tập thỏa ràng buộc, hybrid đưa study được nhiều nguồn nhãn khẳng định lên trước. Relevance nhị phân không tách được hai mode (WORKLOG D24) |
| Hybrid có chữa điểm mù phủ định của vector không? | exclude_violation@10 vector **0,319**, hybrid **0,000** | Có: gần một phần ba kết quả vector của câu có exclude mang đúng finding bị loại ở trạng thái PRESENT; hybrid lọc nhãn trước nên bằng 0 theo construction |

Hybrid thắng vector ở p@10 (1,000 so với 0,359) **theo thiết kế**: nó lọc trên đúng nhãn định nghĩa
relevance (`docs/evaluation-design.md`). Con số đó là tính chất của hệ thống, không phải kết quả
retrieval. Câu hỏi retrieval là dòng đầu của bảng trên.

## Cách đo

- **Benchmark** `benchmarks/parser_queries_v1.jsonl`, sha256
  `8d19c619fd159ebbad2254b3ec5db7bc7b8b6ba197b6c6a9cf04633ed2c9058a`, không đổi từ Phase 4. `dev` 40
  câu để tune, `heldout` 160 câu. 10 câu nhóm `other` (hỏi thứ ngoài 14 nhãn, như hiatal hernia,
  cervical rib; 2 ở dev, 8 ở heldout) không có gold nên không có tập relevant: bỏ khỏi metric, đếm ở
  `skipped_no_gold`. Còn lại **38 câu dev, 152 câu heldout**. Tập 400 câu của kế hoạch §18 để Phase 7
  (WORKLOG D24).
- **Ba cấu hình parser.** `gold`: gold của benchmark, `semantic_query` dựng bằng
  `QueryParser.finalize` theo luật của rule parser. Đây là bảng chính và là cấu hình duy nhất dùng để
  tune. `rule`: `QueryParser.parse(mode="rule")`, tất định. `auto`: parser mặc định (gpt-5-mini, rule
  fallback, WORKLOG D22), không tất định (T23, T25).
- **Relevance luôn lấy từ gold**, kể cả với `rule`/`auto`: `rel(s) = 1` khi `s` qua bộ lọc nhãn của
  gold dưới policy đang xét **và** ảnh của study nằm trong view của gold. Request search thì mang view
  của parse, nên `rule`/`auto` đo đầu-cuối. **Graded:** `grade = 1 + trung bình (trên các concept bị
  ràng buộc) của số nguồn khẳng định đúng trạng thái yêu cầu`, trong bốn nguồn `CHEXPERT_V1`,
  `CHEXBERT_IMPRESSION`, `CHEXBERT_FINDINGS`, `CHEXBERT_REPORT`. Include PRESENT đếm nguồn ghi
  PRESENT, include UNCERTAIN đếm UNCERTAIN, exclude đếm **ABSENT** (để "không có" nói rõ xếp trên im
  lặng).
- **Metric**, top 10, trung bình macro trên câu:
  - `p@5`, `p@10` chia cho k kể cả khi trả ít hơn k. Với `gold`, constraint satisfaction@10 ≡ p@10
    nên chỉ báo p@10.
  - `nDCG@10` graded, IDCG từ 10 grade cao nhất của **toàn** tập relevant. `nDCG nhị phân` (mọi
    study relevant có grade 1) để đối chiếu.
  - MRR; `returned@10` = số kết quả / 10.
  - `exclude_violation@10` = số kết quả có một concept của `gold.exclude` ở PRESENT / số kết quả
    **trả về**. Câu không có exclude hoặc không trả gì không được tính.
  - `evidence_coverage` = tỉ lệ cặp (kết quả, concept include) có câu evidence;
    `evidence_polarity_match` = tỉ lệ câu evidence có polarity mong muốn.
  - Latency là thời gian search trong tiến trình, **không tính parse**, bỏ câu đầu (n = 151).
- **Policy.** Mọi số dùng `(exclude_present, EXCLUDE)` (WORKLOG D9). Nhóm `negation` chạy thêm
  `(explicit_absence_required, EXCLUDE)`.
- **Thứ được tune trên dev: chỉ fusion.** Lưới `rrf_k ∈ {10, 60, 200}` × `leg_weights (vector,
  report) ∈ {(1, 1), (1, 0,5), (0,5, 1)}`, parser `gold`, mode `hybrid`, cùng relevance với lần đánh
  giá. Luật chọn viết trước: nDCG@10 cao nhất; chênh < 0,005 so với mặc định `(60, (1, 1))` thì giữ
  mặc định. `vector_top_k = report_top_k = 100` cố định. Từ vựng report (`concepts.yaml:report`) cố
  định trước lưới. Giá trị chọn (`rrf_k 10`, vector 0,5 / report 1,0) được commit ở `a1361db`
  **trước** heldout.
- **Heldout chạy hai lần, cả hai được báo cáo; lần 2 là lần chạy lại có khai báo.** Lần 1 (2026-10-21 00:49–00:56, `a1361db`) tính
  relevance trong view **của parse** cho `rule`/`auto`, trái spec Phase 5 §11.3 (view của gold). Lỗi
  evaluator được sửa ở `47c2943`, rồi lệnh đầy đủ chạy lại (01:13–01:20). Benchmark, ngưỡng, từ vựng
  và fusion không đổi, không chạy `--grid`. Trước lần 2, report index được dựng lại để `meta` mang
  `concepts_sha256`/`git_commit` hiện tại; `corpus_sha256` không đổi nên nội dung giống hệt. Số `gold`
  (mọi metric, mọi breakdown, `per_query` và verdict) **trùng từng bit** giữa hai lần: so bằng dict
  trên `metrics`, `by_group`, `by_language`, `by_policy`, `per_query`, không tính latency.
- **Máy và dịch vụ.** Qdrant trong Docker (`docker compose up -d qdrant`), collection
  `cxr_images_20261011_chexpert25k_biomedclip_v1`; `uv run cxr validate-indexes` 7/7 PASS trước khi
  đo. `auto` gọi OpenAI qua internet, `.env` chỉ chứa key; Ollama không tham gia. API đo qua
  `127.0.0.1:8001`, vì cổng 8000 đang bị một uvicorn cũ (code trước Phase 5) giữ.
- **Lệnh:** xem *Tái lập*.

## Kết quả heldout (152 câu có gold, policy mặc định, lần 2)

| Parser | Mode | p@5 | p@10 | nDCG@10 | nDCG nhị phân | MRR | violation@10 | returned@10 | phủ evidence | polarity | fallback | p50 / p95 ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `gold` | label | 1,000 | 1,000 | 0,531 | 1,000 | 1,000 | 0,000 | 1,000 | 0,537 | 0,849 | — | 18 / 24 |
| `gold` | vector | 0,382 | 0,359 | 0,235 | 0,371 | 0,528 | 0,319 | 0,908 | 0,510 | 0,837 | — | 179 / 196 |
| `gold` | **hybrid** | 1,000 | 1,000 | **0,607** | 1,000 | 1,000 | 0,000 | 1,000 | 0,922 | 0,862 | — | 207 / 259 |
| `rule` | label | 0,655 | 0,657 | 0,349 | 0,657 | 0,673 | 0,063 | 0,730 | 0,561 | 0,814 | 0,000 | 16 / 24 |
| `rule` | vector | 0,297 | 0,284 | 0,179 | 0,294 | 0,433 | 0,359 | 0,789 | 0,525 | 0,793 | 0,000 | 175 / 194 |
| `rule` | hybrid | 0,730 | 0,722 | 0,427 | 0,722 | 0,752 | 0,136 | 0,868 | 0,924 | 0,819 | 0,000 | 201 / 252 |
| `auto` | label | 0,980 | 0,980 | 0,520 | 0,980 | 0,980 | 0,000 | 0,993 | 0,536 | 0,848 | 0,000 | 18 / 23 |
| `auto` | vector | 0,378 | 0,356 | 0,234 | 0,368 | 0,526 | 0,327 | 0,908 | 0,511 | 0,827 | 0,000 | 173 / 193 |
| `auto` | hybrid | 0,980 | 0,980 | 0,595 | 0,980 | 0,980 | 0,000 | 1,000 | 0,921 | 0,855 | 0,000 | 204 / 256 |

```
verdict: PASS {"single_p10": {"value": 1.0, "threshold": 0.7, "passed": true}, "constraint_satisfaction": {"value": 1.0, "threshold": 0.95, "passed": true}, "hybrid_gain_over_vector": {"value": 1.5791510231504176, "threshold": 0.05, "passed": true}, "passed": true}
```

Bốn chữ số thập phân (cùng file): gold hybrid nDCG@10 0,6072, vector 0,2354, label 0,5309;
exclude_violation@10 vector 0,3192.

- **Gold vector trả dưới 10 kết quả ở 14/152 câu**, đúng là 14 câu tiếng Việt chỉ có exclude của
  heldout: `semantic_query` rỗng nên vector leg không có gì để encode (WORKLOG D26). 14 câu tiếng Anh
  cùng loại encode nguyên văn câu phủ định.
- **`auto` trượt đúng 3/152 câu** (p@10 = 0 ở cả label lẫn hybrid): `vi-044`, `vi-052`, `vi-079`. Đó
  là ba lỗi parse của heldout Phase 4c ("vôi hóa màng phổi" không ra concept, "có thể thấy rõ gãy
  xương đòn" thành `UNCERTAIN`, "tư thế trước-sau" thành PA; WORKLOG T24, D21).
- **`rule` hybrid có p@10 < 1 ở 61/152 câu**, 28 câu bằng 0: lỗi parse của rule parser (Phase 4:
  concept F1 0,793), không phải lỗi retrieval.
- **`rule` hybrid trả nhiều hơn và vi phạm nhiều hơn `rule` label** (returned 0,868 so với 0,730,
  violation 0,136 so với 0,063). Khi rule parse không ra ràng buộc nào, label trả rỗng còn hybrid xếp
  hạng cả corpus, và những kết quả không ràng buộc đó có thể mang concept bị loại. Đây là cách đọc
  đường code `search_hybrid(label_order=None)`, khớp với số, không đo riêng.

## Theo nhóm và ngôn ngữ (gold, lần 2)

Với `gold`, label và hybrid có p@10 = MRR = 1,000, violation 0 và returned 1,000 ở mọi nhóm và mọi
ngôn ngữ, nên bảng chỉ ghi nDCG của chúng.

| Tập con (n) | nDCG label | nDCG vector | nDCG hybrid | vector p@10 | vector MRR | vector violation | vector returned | phủ evidence label / vector / hybrid | polarity hybrid |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| single (70) | 0,557 | 0,309 | **0,664** | 0,440 | 0,614 | n/a | 1,000 | 0,489 / 0,474 / 0,989 | 0,812 |
| multi (60) | 0,580 | 0,199 | **0,635** | 0,290 | 0,479 | 0,307 | 0,950 | 0,600 / 0,557 / 0,835 | 0,926 |
| negation (40) | 0,395 | 0,140 | **0,435** | 0,302 | 0,447 | 0,319 | 0,650 | 0,650 / 0,600 / 0,950 | 0,876 |
| uncertainty (16) | 0,590 | 0,079 | **0,640** | 0,125 | 0,261 | n/a | 1,000 | 0,512 / 0,450 / 0,988 | 0,292 |
| view (20) | 0,570 | 0,449 | **0,657** | 0,620 | 0,933 | 0,100 | 1,000 | 0,565 / 0,560 / 0,960 | 0,976 |
| en (76) | 0,531 | 0,259 | **0,615** | 0,414 | 0,606 | 0,320 | 1,000 | 0,538 / 0,512 / 0,922 | 0,862 |
| vi (76) | 0,531 | 0,212 | **0,599** | 0,304 | 0,450 | 0,317 | 0,816 | 0,536 / 0,509 / 0,921 | 0,862 |

- Hybrid hơn label-only ở **mọi** nhóm: +0,107 (single), +0,055 (multi), +0,040 (negation), +0,050
  (uncertainty), +0,087 (view).
- Vector yếu nhất ở `uncertainty` (p@10 0,125) và `negation` (0,302). `view` là nhóm vector khá nhất
  (0,620), vì mode vector cũng lọc view (WORKLOG G3).
- Với `gold`, câu `en-NNN` và `vi-NNN` có cùng gold ở 79/80 cặp heldout (Phase 4), và câu có include
  dựng cùng một `semantic_query` từ template. Khoảng cách EN–VI của vector vì vậy chủ yếu đến từ 14 câu
  VI chỉ có exclude không có vector leg (returned 0,816). Với 14 câu đó hybrid cũng không có report leg
  (không có term tiếng Anh để MATCH), nên trả theo thứ tự label (D18): hybrid VI 0,599 so với EN 0,615.
- Parser thật, hybrid nDCG@10 EN / VI: `rule` 0,478 / 0,375 (vector VI p@10 chỉ 0,182, returned
  0,579); `auto` 0,619 / 0,572.

## Negation theo hai policy (gold, nhóm `negation`, 40 câu, lần 2)

| Policy | Mode | p@10 | nDCG@10 | MRR | violation@10 | returned@10 | phủ evidence | polarity |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `exclude_present` / `EXCLUDE` | label | 1,000 | 0,395 | 1,000 | 0,000 | 1,000 | 0,650 | 0,882 |
| | vector | 0,302 | 0,140 | 0,447 | 0,319 | 0,650 | 0,600 | 0,900 |
| | hybrid | 1,000 | **0,435** | 1,000 | 0,000 | 1,000 | 0,950 | 0,876 |
| `explicit_absence_required` / `EXCLUDE` | label | 1,000 | 0,612 | 1,000 | 0,000 | 1,000 | 0,633 | 0,830 |
| | vector | 0,082 | 0,059 | 0,200 | 0,319 | 0,650 | 0,600 | 0,900 |
| | hybrid | 1,000 | **0,640** | 1,000 | 0,000 | 1,000 | 1,000 | 0,900 |

- Violation của vector giống nhau dưới hai policy (0,319), vì nó chỉ đếm kết quả PRESENT. p@10 của
  vector rơi 0,302 → 0,082 vì tập relevant co lại (Phase 4: median 10.223 → 1.013 study).
- Hai policy cho hai tập relevant khác nhau, nên nDCG giữa hai policy không so trực tiếp được. Trong
  cùng một policy, hybrid hơn label-only ở cả hai.
- Parser thật, hybrid nDCG@10 `exclude_present` / `explicit_absence_required`: `rule` 0,377 / 0,525;
  `auto` 0,442 / 0,643.
- **Cùng câu qua API** ("Find edema with no pleural effusion.", parse `auto`, top 10): vector **6/10**
  kết quả mang `Pleural Effusion: PRESENT` dưới cả hai policy (Phase 4 cũng 6/10), hybrid **0/10**.
  `total_matches` 2.760 / 1.532, bằng số mode label của Phase 4.

## Evidence

- **Coverage trên heldout** (gold): hybrid 0,922, label 0,537, vector 0,510; polarity khớp 0,862 /
  0,849 / 0,837. Coverage của hybrid cao vì report leg xếp report nhắc đúng term của include lên
  trước (hiệu ứng chọn mẫu), không phải vì evidence tốt hơn.
- **Polarity yếu ở hedge:** nhóm `uncertainty` của hybrid chỉ khớp 0,292 (vector 0,142, label 0,450).
  Polarity do rule parser gán trên từng câu, nên câu kiểu `"QUESTION\nPULMONARY EDEMA VERSUS
  INFECTION."` (smoke, use case 4, mode vector) bị gắn Pneumonia `affirmed`.
- **Faithfulness.** `evidence_faithful` của spec Phase 5 §11.4 không có trong JSON. Faithfulness đúng
  **theo cấu trúc**: `sentences()` cắt mỗi câu thành lát của chính `search_text` (offset, không strip hay
  ghép), và pipeline gọi `EvidenceMatcher.build` với `verify=True`, kiểm lại đúng lát đó (raise, không bị
  nuốt; cả hai lần heldout chạy xong với exit 0). Đi kèm là việc sqlite lưu `search_text` nguyên văn
  (`test_search_text_is_stored_verbatim`). Hai phép kiểm trên dữ liệu thật (`uv run pytest -m
  integration`, cả hai pass):
  1. **Độc lập:** `test_hybrid_fifty_images_round_trip` chạy hybrid trên 50 ảnh thật, lấy evidence từ
     sqlite qua pipeline và assert `text[start:end] == e.text` với `search_text` đọc từ parquet, cho mọi
     hit.
  2. **Regression check:** `test_evidence_is_faithful_on_a_thousand_real_reports` lấy 1.000 report đầu,
     mỗi study một `SearchQuery` include mọi concept PRESENT (nhãn resolve), và assert cùng đẳng thức,
     nhưng trên chính text vừa cắt. Theo cấu trúc trên thì nó không thể sai, nên nó chỉ khóa cách tách
     câu trên report thật (WORKLOG T26), không phải bằng chứng độc lập.
- **Độ phủ không chọn mẫu** (1.000 report đầu, mọi concept PRESENT; "có evidence" = ít nhất một câu
  nhắc concept theo `en + report`, polarity bất kỳ): **2.375 / 3.803 cặp = 62,5%**.

| Concept | PRESENT | có evidence | độ phủ |
|---|---:|---:|---:|
| No Finding | 161 | 38 | 23,6% |
| Enlarged Cardiomediastinum | 91 | 4 | 4,4% |
| Cardiomegaly | 275 | 118 | 42,9% |
| Lung Opacity | 715 | 482 | 67,4% |
| Lung Lesion | 84 | 38 | 45,2% |
| Edema | 364 | 227 | 62,4% |
| Consolidation | 107 | 62 | 57,9% |
| Pneumonia | 57 | 33 | 57,9% |
| Atelectasis | 284 | 166 | 58,5% |
| Pneumothorax | 134 | 89 | 66,4% |
| Pleural Effusion | 571 | 412 | 72,2% |
| Pleural Other | 51 | 23 | 45,1% |
| Fracture | 99 | 52 | 52,5% |
| Support Devices | 810 | 631 | 77,9% |

Ô thấp (Enlarged Cardiomediastinum 4/91, No Finding 38/161) trông như lỗ từ vựng cho cách report
diễn đạt. Đó là giả thuyết, chưa đo.

## Latency

**Trong tiến trình** (heldout lần 2, gold, chỉ search, n = 151): label 18 / 24 ms, vector 179 / 196,
hybrid 207 / 259 (p50 / p95). Lần 1 chậm hơn ở **mọi** ô, cả `rule` và `auto` (gold: 25 / 43, 232 /
269, 300 / 386). Hai lần chạy trên cùng máy, không có server nào của lần đo chạy song song; nguyên
nhân chưa đo.

**HTTP** (WORKLOG T19; `cxr bench-vector --api-url http://127.0.0.1:8001 --mode …`, 30 câu × 4 lần
lặp được tính, n = 120, git `a1361db`, torch 14 luồng):

| Đo | p50 ms | p95 ms | Ghi chú |
|---|---:|---:|---|
| `http_vector`, server `LLM_DISABLED=true` | 156,1 | 171,2 | Phase 3: 150 / 158, khi chưa có parse; giờ có thêm parse rule và evidence |
| `http_hybrid`, LLM tắt | 173,9 | 218,8 | |
| `http_hybrid`, LLM bật | 188,0 | 221,5 | Lần lặp 0 của mỗi câu, lần duy nhất gọi LLM, bị bỏ, nên đây là search + cache hit của parse. Một lần gọi LLM thật mất 1.079–1.756 ms (`llm_ms` của 6 câu smoke) |
| Encode + Qdrant trong tiến trình, ba lần bench trên | 139,5 / 141,0 / 140,4 | 142,3 / 143,6 / 142,8 | |

**Cold start:** từ lúc gọi `uv run uvicorn` tới `/api/v1/health` 200 đầu tiên (poll 0,5 s) là **7,6 s**
khi LLM tắt và **8,1 s** khi LLM bật. `cold_ms` trong tiến trình của bench (dựng state + câu đầu):
7.014,7 / 7.405,8 / 6.995,2 ms (Phase 3: 5.289 ms).

**Lọc Qdrant theo tập cho phép** (spec Phase 5 §8.3; `LLM_DISABLED=true uv run python
artifacts/metrics/qdrant_filter_probe.py`, 2026-10-20; câu "chest x-ray showing pulmonary edema",
`limit` 100, 20 lần đo mỗi ô bỏ lần đầu, n = 19, một lần chạy):

| Số id cho phép | (a) `HasIdCondition` p50 / p95 ms | (b) `MatchAny` + payload index | (c) không lọc, top 2.000 rồi giao (số hit giữ lại) |
|---:|---|---|---|
| 500 | 15,2 / 25,3 | 15,8 / 25,7 | 33,1 / 45,6 (40/100) |
| 2.760 | 30,8 / 51,6 | 24,5 / 31,6 | 30,7 / 39,6 (100) |
| 10.000 | **30,7 / 35,6** | 31,5 / 47,0 | 30,9 / 35,5 (100) |
| 25.000 | 46,0 / 62,0 | 45,4 / 63,7 | 31,4 / 34,7 (100) |

Chọn (a): dưới ngưỡng 100 ms và không cần bước dựng index lúc khởi động. Ở 10.000–25.000 id, (a) và
(b) nằm trong nhiễu của một lần chạy. (c) rẻ nhưng chỉ giữ 40/100 hit khi tập cho phép nhỏ.

**Nhánh report.** Tập cho phép dùng `study_id IN (SELECT value FROM json_each(?))`. Phép này chỉ được
đo trên một chỉ mục **tổng hợp** 25.000 report (500 id 24,6 ms, 10.000 id 37,1 ms), không phải corpus
thật. Probe thiết kế trên report thật dùng bảng TEMP: 500 id nạp 0,4 + truy vấn 11,2 ms, 10.000 id
10,7 + 16,3 ms. Trên chỉ mục thật, nhánh report chưa được bấm giờ riêng; nó nằm trong 207 / 259 ms của
hybrid.

**Chỉ mục report:** 25.000 dòng, 32.759.808 B; `time uv run cxr build-report-index` 1,048 s
(2026-10-20) và 0,998 s ở lần dựng lại (2026-10-21).

## Lưới dev (gold, hybrid, 38 câu)

`uv run cxr eval-retrieval --split dev --parsers gold --modes hybrid --grid --out
artifacts/metrics/retrieval_dev.json` ở `241881e`. p@10 = 1,000 ở cả 9 cấu hình.

| `rrf_k` | (vector, report) | nDCG@10 |
|---:|---|---:|
| 10 | (1, 1) | 0,6406 |
| 10 | (1, 0,5) | 0,6170 |
| **10** | **(0,5, 1)** | **0,6565** ← chọn |
| 60 | (1, 1), mặc định | 0,6441 |
| 60 | (1, 0,5) | 0,6222 |
| 60 | (0,5, 1) | 0,6545 |
| 200 | (1, 1) | 0,6425 |
| 200 | (1, 0,5) | 0,6263 |
| 200 | (0,5, 1) | 0,6476 |

Chênh so với mặc định 0,6565 − 0,6441 = 0,0125 ≥ 0,005, nên không giữ mặc định. Mọi dòng `(0,5, 1)`
hơn dòng `(1, 1)` cùng `rrf_k`, mọi dòng `(1, 0,5)` thấp nhất: phần lợi đến từ việc nâng trọng số
report. `rrf_k` chỉ dịch nDCG 0,004–0,009 trong một cặp trọng số. Hệ quả của trọng số đã chọn: thứ hạng
hybrid chủ yếu do report dẫn (*Cách đọc số*).

Lần chạy dev tham chiếu với fusion đã chọn (`--split dev --parsers gold rule auto`): ba parser cho số
giống hệt nhau; gold label / vector / hybrid nDCG@10 0,524 / 0,302 / 0,657, p@10 1,000 / 0,447 /
1,000, violation 0 / 0,511 / 0; verdict PASS (gain 1,176). Rule parser đã được chỉnh tới 1,000 trên dev
ở Phase 4, nên dev không nói gì về parser.

## API smoke (sáu use case, parse `auto`, top 10, server LLM bật)

`smoke_phase5.py` chạy trên server của lần đo HTTP có LLM. Mỗi câu được parse một lần qua
`/api/v1/query/parse` trước, nên `latency_ms` của search là thời gian search. "Vi phạm" = kết quả mang
concept bị loại ở PRESENT; "không thỏa" = `constraints_satisfied` false.

| # | Câu | `total_matches` | vector: vi phạm / không thỏa | hybrid: vi phạm / có evidence | hybrid `latency_ms` |
|---|---|---:|---|---|---:|
| 1 | Find chest X-rays with cardiomegaly. | 6.618 | 0 / 3 | 0 / 10 | 191 |
| 2 | Find cardiomegaly with pulmonary edema. | 3.421 | 0 / 4 | 0 / 10 | 190 |
| 3 | Find edema with no pleural effusion. | 2.760 | **6** / 7 | **0** / 10 | 202 |
| 3 | cùng câu, `explicit_absence_required` | 1.532 | **6** / 9 | **0** / 10 | 189 |
| 4 | Find cases where pneumonia is uncertain. | 4.435 | 0 / 8 | 0 / 10 | 188 |
| 5 | Show frontal PA images with lung opacity. | 2.408 | 0 / 0 | 0 / 10 | 227 |
| 6 | Tìm ảnh X-quang thẳng có tim to và phù phổi. | 3.421 | 0 / 4 | 0 / 10 | 191 |

Không request nào có warning. Câu 6 parse ra Cardiomegaly + Edema, `vi`, nên cùng tập với câu 2.

## Cách đọc số

- **Hybrid thắng vector theo thiết kế.** Hybrid lọc trên đúng nguồn nhãn định nghĩa relevance, nên p@10
  1,000 và violation 0 là hệ quả của cấu trúc. Gain 1,579 trong verdict cũng vậy. Câu hỏi thật là
  hybrid so với label-only: 0,607 so với 0,531.
- **Proxy graded có thiên lệch.** Bậc đếm nguồn nhãn, mà cả bốn nguồn đều suy từ report. Study có
  findings có tối đa 4 nguồn, study không có findings tối đa 3 (WORKLOG D10), nên proxy thiên về study
  có findings. Report leg (bm25 trên chính report đó) và bậc relevance đọc cùng một văn bản, nên phần
  hơn 0,076 của hybrid so với label-only không phải bằng chứng độc lập. Human review của Phase 7 mới
  là trục độc lập (`docs/evaluation-design.md`).
- **Từ vựng report phủ không đều.** Probe thiết kế (từ vựng thử, porter stem, trên study PRESENT):
  recall 0,27 (Pleural Other) đến 0,76 (Support Devices), precision 0,49–0,93. Với từ vựng cuối, độ
  phủ evidence không chọn mẫu là 4,4–77,9% theo concept (*Evidence*). Concept nào report hiếm khi viết
  đúng term thì report leg ít tín hiệu.
- **Thứ hạng hybrid chủ yếu do report dẫn** dưới trọng số 0,5 / 1,0. Trong smoke, 5/6 hit đầu của
  hybrid (policy mặc định) có `report_rank` 1 và không có `vector_rank`, `rrf` = 1/11 = 0,0909. Một
  study chỉ được vector xếp hạng 1 chỉ được 0,5/11 = 0,045. Hybrid ở đây gần với "bm25 trong tập cho
  phép, vector làm tín hiệu phụ" hơn là hai leg ngang nhau.
- **Câu EN chỉ có exclude vẫn có report leg, và report leg xếp theo chính finding bị loại.** D26 cho
  `semantic_query` = nguyên văn câu, `include` rỗng nên `build_match` quote mọi token: "show me films with
  no evidence of pneumothorax" → `"show" OR "films" OR "with" OR "evidence" OR "pneumothorax"`. Trong tập
  cho phép, report leg vì thế đưa lên trước report nhắc finding bị loại, thường để phủ định nó: ở top 10
  bm25 của 14 câu heldout EN loại này, 44/140 hit trích một câu phủ định concept bị loại (en-023 8/10), và
  37/140 hit trích cùng một câu hai lần, một lần cho concept (`negated`) và một lần `mention` free-text (vd
  `"1. EXPIRATORY FILM SHOWING NO EVIDENCE OF PNEUMOTHORAX."`). Trên 14 câu đó (gold), nDCG@10 hybrid /
  label / vector là 0,409 / 0,328 / 0,210. 14 câu VI tương ứng không có leg nào nên hybrid bằng label
  (0,328), và 62 câu còn lại của mỗi ngôn ngữ có hybrid 0,661 ở cả hai: khoảng cách hybrid EN–VI (0,615 /
  0,599) nằm ở 14 cặp câu này. Bậc relevance của một exclude đếm số nguồn ghi ABSENT, và nguồn thường chỉ
  ghi ABSENT khi report có câu phủ định, nên phần hơn label ở đây là bm25 tìm đúng loại câu đó (cùng văn
  bản với bậc, xem *Proxy graded có thiên lệch*), không phải hybrid hiểu phủ định. Hành vi giữ nguyên;
  câu hỏi mở là WORKLOG P11. Lệnh (chỉ đọc): `uv run python artifacts/metrics/exclude_only_probe.py
  artifacts/metrics/retrieval_eval_heldout_20261021_run2.json`.
- **`auto` không tái lập.** Giữa hai lần heldout, ít nhất 4 parse khác nhau. Đây là cận dưới đọc từ
  `per_query`, vì parse không được lưu: `en-044` và `vi-060` đổi ràng buộc (lần 1 trượt, lần 2 đúng),
  `en-078` và `en-080` chỉ đổi `semantic_query`. Mỗi số `auto` là một lần chạy. `auto` hybrid nDCG
  0,5918 → 0,5951 giữa hai lần là nhiễu LLM cộng hiệu ứng của bản sửa evaluator, không phải cải thiện.
- **Lần 1 và lần 2.** Lần 1 chấm relevance trong view của parse, nên `rule`/`auto` không bị phạt khi
  parse bỏ hay đảo view. Với `rule` (tất định) chỉ `vi-079` đổi (gold AP, parse không có view): `rule`
  hybrid p@10 0,7230 → 0,7224, nDCG 0,4272 → 0,4266. Với `auto`, `vi-079` (parse ra PA) từ p@10 1,0
  xuống 0,0. Số `gold` và verdict trùng từng bit, nên kết luận không đổi.
- **10 câu `other` không vào metric.** Chỉ nhánh report free-text phủ được chúng, và không có gold để
  chấm.
- **Latency giữa hai lần lệch nhau** (lần 2 nhanh hơn ở mọi ô) mà chưa rõ vì sao. Đọc latency trong
  tiến trình như một khoảng, không phải một điểm.
- **Use case 6 của smoke là bản §4.2 của kế hoạch** ("Tìm ảnh X-quang thẳng có tim to và phù phổi.",
  không có exclude), không phải câu mở đầu của kế hoạch có "nhưng không có tràn dịch màng phổi" mà
  Phase 4c dùng.

## Tái lập

```bash
docker compose up -d qdrant
uv run cxr build-report-index        # 25.000 dòng, ≈1 s
uv run cxr validate-indexes          # 7 x PASS
uv run cxr eval-retrieval --split dev --parsers gold --modes hybrid --grid --out artifacts/metrics/retrieval_dev.json
uv run cxr eval-retrieval --split dev --parsers gold rule auto --out artifacts/metrics/retrieval_dev.json
# heldout = make eval-retrieval (mặc định); GHI ĐÈ retrieval_eval.json, nên đóng băng ngay sau đó
uv run cxr eval-retrieval --split heldout --parsers gold rule auto --modes label vector hybrid \
    --policies negation_both --out artifacts/metrics/retrieval_eval.json
cp artifacts/metrics/retrieval_eval.json artifacts/metrics/retrieval_eval_heldout_<ngày>.json
# HTTP và smoke: driver khởi động uvicorn trên 127.0.0.1:8001, đo cold start, chạy bench rồi dừng
uv run python artifacts/metrics/http_phase5.py A 8001   # LLM_DISABLED=true: bench-vector --mode vector, rồi --mode hybrid
uv run python artifacts/metrics/http_phase5.py B 8001   # LLM bật: bench-vector --mode hybrid, rồi smoke_phase5.py
uv run pytest -m integration -q      # 3 passed; cần Qdrant
```

Các số đếm theo câu ở trên (14 câu VI chỉ có exclude, 3 câu `auto` trượt, 61/152 câu `rule`) đọc từ
file đóng băng lần 2 qua `per_query` bằng script chỉ đọc, ví dụ:

```bash
uv run python -c "import json; e=json.load(open('artifacts/metrics/retrieval_eval_heldout_20261021_run2.json'))['eval']['parsers']; print(sorted(q for q,v in e['auto']['modes']['hybrid']['per_query'].items() if v['p@10']==0))"
```

`auto` chạy lại sẽ không ra đúng từng số (WORKLOG T23, T25). `gold` và `rule` tái lập được với cùng
commit, cùng build và cùng report index (`corpus_sha256` ở đầu file).

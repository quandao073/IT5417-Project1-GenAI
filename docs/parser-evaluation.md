# Query parser — Phase 4

**Ngày:** 2026-10-14 · **Git:** chạy trên `cd890d8`, cộng prompt và lexicon chưa commit (nằm trong commit Task 10, xem *Cách đo*) · **Ollama:** 0.40.3 native · **Model:** `qwen2.5:3b-instruct-q4_K_M` · **prompt_sha256:** `c975e4ca2e2172d376915e17926ab8c28a9110876186e600bb178124e4caac35`

Nguồn của mọi số dưới đây: `artifacts/metrics/parser_eval.json` (heldout, kèm `label_filter`; bản
đóng băng `parser_eval_heldout_20261014.json`, sha256 `999fbc70…`, vì mỗi lần `cxr eval-parser` ghi đè
`parser_eval.json`),
`artifacts/metrics/parser_eval_dev.json` (dev, prompt cuối) và `artifacts/metrics/parser_dev_<k>.json`
(từng vòng chỉnh trên dev). Cả ba đều gitignored. Số API lấy từ `artifacts/metrics/smoke_*.json`
và `api_phase4_cold.json`.

**Kết luận:** heldout **FAIL**. Parser `auto` (LLM, rule fallback) đạt concept / status / view
macro-F1 0,880 / 0,718 / 0,890, so với ngưỡng 0,90. Rule parser đạt 0,793 / 0,862 / 0,945. LLM 3B
dẫn ở concept nhưng **thua rule parser ở status và view**. Lợi thế đó của rule parser một phần đến
từ việc nó dùng chung lexicon mà benchmark được viết theo (xem *Cách đọc số*). Bước tiếp theo, nếu
người dùng chọn:
nhánh 7B (`qwen2.5:7b-instruct-q4_K_M`, pull 4,7 GB; WORKLOG P6).

## Cách đo

- **Benchmark** `benchmarks/parser_queries_v1.jsonl`, sha256
  `8d19c619fd159ebbad2254b3ec5db7bc7b8b6ba197b6c6a9cf04633ed2c9058a`. Gồm 200 câu (EN 100 / VI 100).
  `dev` là câu `001–020` của mỗi ngôn ngữ (40 câu), `heldout` là `021–100` (160 câu). Nhóm chính chạy theo
  chu kỳ 20 câu (`parser_benchmark.PATTERN`) nên dev phủ mọi nhóm. Nhóm dùng để tách số dưới đây
  được suy ra từ gold, nên một câu có thể thuộc nhiều nhóm.
- **Ai viết:** các câu mồi lấy từ kế hoạch; phần còn lại, cả câu lẫn gold, do agent của Task 8 viết.
  Phán quyết review đã đổi `en-046`, `vi-072`, `vi-099`, `en-013`, `vi-013`. Gold được sửa cho khớp
  quy ước của CheXpert labeler ở commit `198cf23` (WORKLOG D21). Ba câu heldout `en-043`, `vi-027`,
  `vi-052` được viết lại thành bẫy cho rule parser **sau khi** đã thấy output của rule parser trên
  heldout. Gold của chúng đúng và không mơ hồ (xem *Cách đọc số*). `en-023` và `vi-087` cũng được
  viết lại sau khi thấy output đó, nhưng rule parser xử lý đúng cả hai trên heldout, nên chúng không
  tạo thiên lệch nào chống lại rule parser.
- **Dev** được dùng cho **4 vòng** chỉnh prompt và lexicon (giới hạn 5). **Heldout chạy đúng một
  lần**, sau vòng cuối, và không có chỉnh sửa nào sau đó.
- **Lệnh:**

  ```bash
  uv run cxr eval-parser --split dev --parsers rule llm auto --verbose \
      --out artifacts/metrics/parser_dev_<k>.json                        # k = 0..4
  uv run cxr eval-parser --split heldout --parsers rule llm auto --label-filter   # một lần
  uv run cxr eval-parser --split dev --parsers rule llm auto --out artifacts/metrics/parser_eval_dev.json
  ```

- **Provenance.** `parser_eval.json` ghi `git_commit` = `cd890d8`, nhưng lần chạy dùng prompt và
  lexicon **chưa commit** lúc đó. Commit Task 10 (commit chứa file này) chứa đúng hai file ấy:
  - `configs/prompts/query_parser_v1.txt`: sha256 của prompt **đã render** (thay `{{CONCEPTS}}`) là
    `c975e4ca2e2172d376915e17926ab8c28a9110876186e600bb178124e4caac35`, ghi ở
    `eval.llm.prompt_sha256` của cả hai JSON.
  - `configs/concepts.yaml`: sha256 của file (byte, xuống dòng LF) là
    `735997a20a1cb3fe27c491f78a52fee80d578f38c6148549efeb4b48e7b273f8`.
- **Máy.** i7-13800H, 64 GB RAM. GPU NVIDIA RTX 2000 Ada Laptop 8 GB (driver 596.72, `nvidia-smi`).
  Ollama 0.40.3 chạy native; model nằm trọn trên GPU (`/api/ps`: `size_vram` = `size` =
  2.159.374.499 B), context 4.096. Prompt dài 2.328 token (`prompt_eval_count`), không bị cắt.
  Request đặt `format` = JSON schema với enum 14 concept, `temperature` 0, `seed` 20260101,
  `num_predict` 300, timeout 20 s, retry 1 lần. Rule parser và phần Python chạy CPU trong cùng
  tiến trình.
- **Latency** là thời gian `QueryParser.parse` của mỗi câu, bỏ câu đầu tiên của mỗi parser
  (n = 159 trên heldout, 39 trên dev). Khóa cache có chứa mode, nên `auto` không dùng lại kết quả
  của `llm`.

## Kết quả heldout (160 câu)

| Parser | schema-valid | concept F1 | status F1 | view F1 | exact | fallback | retry | p50 / p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `rule` | 1,000 | 0,793 | **0,862** | **0,945** | 0,606 | — | — | 0,1 / 0,2 |
| `llm` | 1,000 | 0,880 | 0,718 | 0,890 | 0,650 | 0 | 0,0375 | 756 / 1.366 |
| **`auto`** | 1,000 | **0,880** | 0,718 | 0,890 | **0,650** | 0 | 0,0375 | 778 / 1.473 |

```
verdict (auto, split heldout): FAIL {"schema_valid": true, "concept_macro_f1": false, "status_macro_f1": false, "view_macro_f1": false, "passed": false}
```

Ngưỡng lấy từ `configs/evaluation.yaml:thresholds`: macro-F1 ≥ 0,90 cho cả ba trục, schema-valid
≥ 0,99, chấm trên `auto`. FAIL là kết quả báo cáo. Benchmark, ngưỡng và metric không được sửa.

- `llm` và `auto` cho dự đoán giống nhau trên cả 160 câu. Không câu nào phải fallback; 6/160 câu
  cần một lần retry và lần retry cho kết quả hợp lệ.
- Theo status, `auto` có EXCLUDED F1 0,646 (tp 31, fp 19, fn 15), PRESENT 0,818, UNCERTAIN 0,690.
  `rule` có EXCLUDED 0,905, PRESENT 0,778, UNCERTAIN 0,903. Trong 54 fn PRESENT của rule, 53 là
  concept bị bỏ sót hoàn toàn, tức thiếu từ trong lexicon. Chỗ yếu nhất của LLM là **đưa nhầm
  finding dương vào exclude**.
- Theo view, `auto` có 3 dự đoán PA thừa; `rule` không có view thừa nào.
- Theo concept, `auto` yếu nhất ở Support Devices 0,71 (10 fn) và Edema 0,78. `rule` yếu nhất ở
  Lung Opacity 0,43, Fracture 0,59 và Support Devices 0,62, đều do thiếu từ.

## Theo ngôn ngữ và theo nhóm (heldout)

`auto`:

| Tập con | n | concept | status | view | exact |
|---|---:|---:|---:|---:|---:|
| en | 80 | 0,911 | 0,826 | 0,890 | 0,700 |
| vi | 80 | 0,848 | **0,593** | 0,885 | 0,600 |
| single | 70 | 0,850 | 0,557 | 0,864 | 0,743 |
| multi | 60 | 0,885 | 0,489 | 1,000 | 0,583 |
| negation | 40 | 0,769 | 0,683 | 1,000 | 0,550 |
| uncertainty | 16 | 0,941 | 0,230 | n/a | 0,625 |
| view | 20 | 0,947 | 0,714 | 0,925 | 0,600 |
| other (gold rỗng) | 8 | 0,000 | 0,000 | n/a | 0,500 |

`rule`:

| Tập con | n | concept | status | view | exact |
|---|---:|---:|---:|---:|---:|
| en | 80 | 0,787 | 0,866 | 1,000 | 0,600 |
| vi | 80 | 0,788 | 0,858 | 0,883 | 0,613 |
| single | 70 | 0,597 | 0,736 | 0,928 | 0,486 |
| multi | 60 | 0,834 | 0,952 | 1,000 | 0,667 |
| negation | 40 | 0,901 | 0,914 | 1,000 | 0,800 |
| uncertainty | 16 | 0,778 | 0,467 | n/a | 0,750 |
| view | 20 | 0,800 | 0,936 | 0,945 | 0,700 |
| other (gold rỗng) | 8 | n/a | n/a | n/a | 1,000 |

- F1 trên một tập con tính mọi lớp xuất hiện trong gold **hoặc** trong dự đoán của tập con đó.
  Vì vậy `other` cho F1 0 khi parser đoán ra concept cho câu có gold rỗng (4/8 câu với `auto`), và
  cho n/a khi không có lớp nào (rule). Status của `uncertainty` thấp vì macro chia đều cho mọi
  lớp có mặt. Ngoài UNCERTAIN (`auto` 0,69, `rule` 0,93), các lớp nhỏ đều F1 0: 2 include PRESENT
  trong gold mà không parser nào bắt được, và với `auto` thêm 3 EXCLUDED đoán thừa.
- Tiếng Việt kéo status của LLM xuống 0,593 (EN 0,826). Rule parser thì đều giữa hai ngôn ngữ
  (0,866 / 0,858). So sánh EN–VI ở đây là có kiểm soát: câu `en-NNN` và `vi-NNN` có cùng gold ở
  79/80 cặp heldout.
- Ba câu bẫy `en-043`, `vi-027`, `vi-052` là **những câu duy nhất** kiểm cấu trúc phủ định/hedging
  thay vì từ vựng: `en-043` và `vi-027` có cue phủ định đứng sau, cách concept nhiều từ hơn cửa sổ
  3 token ("… in both lower lobes has resolved", "… lượng ít bên trái nay đã hết"), còn `vi-052` có
  "có thể" mang nghĩa "nhìn thấy được", không phải hedge. Rule parser sai cả 3, `auto` sai `vi-027`.
  Các câu negation và uncertainty còn lại dùng cue mà lexicon đã có, nên phần lớn số ở hai nhóm đó
  đo **độ phủ từ vựng**. Lợi thế của rule parser ở `negation` (status 0,914 so với 0,683) phải đọc
  theo nghĩa đó.

## Dev: bốn vòng chỉnh và khoảng cách với heldout

Mỗi vòng sửa `configs/prompts/query_parser_v1.txt` và/hoặc `configs/concepts.yaml`, chạy
`uv run pytest -q` (618 passed mỗi lần) rồi chạy lệnh dev. Số lấy từ `parser_dev_<k>.json`; hash là
12 ký tự đầu của `eval.llm.prompt_sha256`.

| k | Thay đổi | prompt | rule c / s / v | llm c / s / v |
|---|---|---|---|---|
| 0 | gốc (`cd890d8`) | `54a3a0ea990c` | 0,971 / 0,991 / 1,000 | 0,761 / 0,606 / 0,900 |
| 1 | Prompt: quy tắc hai bước (gọi tên finding, kèm gợi ý dụng cụ → Support Devices, bình thường → No Finding, tràn khí/tràn dịch; rồi xếp vào list), thêm 5 ví dụ. Lexicon: `big heart` (Cardiomegaly), `normal chest` (No Finding) | `8aea837d5f0e` | 1,000 / 1,000 / 1,000 | 0,878 / 0,708 / 0,800 |
| 2 | Quy tắc xếp list viết lại thành mẫu "X" (phủ định trước/sau, hedge); đổi ví dụ bị model chép nguyên mẫu | `fd1e14d9d6ad` | 1,000 / 1,000 / 1,000 | 0,872 / 0,777 / 0,583 |
| 3 | Thêm 3 ví dụ chỉ-exclude (negative for, free of, không có), máy tạo nhịp, ví dụ frontal/PA; viết lại quy tắc views | `f398aaf34437` | 1,000 / 1,000 / 1,000 | 0,928 / 0,870 / 0,650 |
| 4 | Bỏ hai ví dụ view thêm ở k=2–3 (chúng mồi model đoán view thừa); cân lại ví dụ VI; thêm ví dụ đa concept có dụng cụ (EN), tràn khí PRESENT (VI), câu không concept (VI) | `c975e4ca2e21` | 1,000 / 1,000 / 1,000 | 0,909 / 0,841 / 0,900 |

`auto` trùng `llm` ở mọi vòng, trừ k=1 (view `auto` 0,667). Prompt cuối có 20 ví dụ (gốc 5). Dừng
ở k=4 vì ứng viên cho vòng 5 không cải thiện trên bộ câu tự viết (dưới đây), nên không có lý do
tiêu thêm một vòng trên dev.

**Live test** (`uv run pytest -m llm`, `tests/integration/test_llm_parser_live.py`). Với prompt
gốc (k=0): 2 passed. Với prompt cuối: 1 passed, 1 failed, lặp lại 4/4 lần. Test đó giờ được đánh
dấu `xfail(strict=True)` (WORKLOG T17, P7), nên lệnh cho `1 passed, 1 xfailed`. Câu use case 6 ("Tìm ảnh
X-quang thẳng có tim to và phù phổi nhưng không có tràn dịch màng phổi.") ra include `[Edema,
Cardiomegaly]` thay vì `[Cardiomegaly, Edema]` và `language` `en`. Exclude và tập concept vẫn
đúng, nên mode `label` không bị ảnh hưởng; `semantic_query` thành "chest x-ray showing pulmonary
edema and cardiomegaly". Assertion được giữ nguyên. Sửa prompt lúc này sẽ cắt đứt liên hệ giữa
prompt và lần chạy heldout duy nhất.

**Giữa các vòng**, các biến thể prompt được so trên câu tự viết, **không lấy từ benchmark**.
Bộ thứ nhất có 12 câu không nêu view và 4 câu có view (`artifacts/metrics/views_probe.py`). Bộ thứ
hai có 32 câu, gồm dương, phủ định, hedge, view và câu không concept
(`artifacts/metrics/probe_wide.py`). Prompt thử nằm ở `artifacts/metrics/probe_prompts/`. **Mỗi số
dưới đây là một lần chạy thăm dò, output không lưu**; theo T18, chạy lại sẽ không ra đúng các số đó.
Chúng chỉ dùng để chọn hướng sửa, không phải số báo cáo.
- Số câu không nêu view mà bị đoán ra view: prompt k=0 0/12, k=3 5/12, k=3 bỏ hai ví dụ view
  0/12. Nguyên nhân là ví dụ, không phải câu chữ của quy tắc.
- Bộ 32 câu: k=0 11/32, k=3 20/32, k=4 23/32. Ứng viên vòng 5 (ví dụ VI vừa include vừa exclude)
  cũng 23/32, và một ví dụ của nó trùng một câu heldout tới token-Jaccard 0,60, nên bị bỏ. Con số
  0,60 đến từ một script nội tuyến không lưu, nhưng phép đo là tất định trên văn bản: prompt ứng viên
  ở `probe_prompts/v5a.txt`.
- Trước mỗi lần cài prompt, mọi dòng `Request:` được so với câu heldout bằng token-Jaccard. Script
  chỉ in điểm lớn nhất, không in câu heldout. Prompt cuối có điểm lớn nhất 0,55, thuộc ví dụ 1 (có
  sẵn trước khi chỉnh trên dev); các ví dụ thêm mới đều ≤ 0,40. Tính lại được bằng
  `uv run python artifacts/metrics/leak.py`.

**Dev cuối và heldout** (dev từ `parser_eval_dev.json`, cùng prompt với k=4):

| | concept | status | view | exact |
|---|---:|---:|---:|---:|
| `auto` dev | 0,912 | 0,827 | 0,833 | 0,725 |
| `auto` heldout | 0,880 | 0,718 | 0,890 | 0,650 |
| `rule` dev | 1,000 | 1,000 | 1,000 | 1,000 |
| `rule` heldout | 0,793 | 0,862 | 0,945 | 0,606 |

```
verdict (auto, split dev): FAIL {"schema_valid": true, "concept_macro_f1": true, "status_macro_f1": false, "view_macro_f1": false, "passed": false}
```

Từ dev xuống heldout, LLM rơi 0,11 ở status và 0,03 ở concept, còn view tăng 0,06. Riêng view tự
dao động 0,067 giữa hai lần chạy dev cùng prompt (xem *Cách đọc số*). Rule parser rơi mạnh hơn
(concept 1,000 → 0,793): lexicon đã được chỉnh trên dev, và benchmark được viết dựa trên chính
lexicon đó. Con số 1,000 trên dev của nó không nói lên gì.

## Label filter trên corpus thật

`--label-filter` chạy `filter_studies_by_labels` với **gold** của mỗi câu heldout có ràng buộc
(152/160) trên 25.000 study, dưới cả 2 absence × 2 uncertainty policy. Bộ lọc chỉ dùng include và
exclude; view không áp ở đây. Ô bảng: median số study khớp (min). Không câu nào khớp 0 study ở
bất kỳ cặp policy nào. Vì `en-NNN` và `vi-NNN` có cùng gold ở 79/80 cặp, mỗi gold gần như được đếm
hai lần; median không đổi vì thế, nhưng n thực chất là số câu chứ không phải số gold khác nhau.

| Nhóm (n) | `exclude_present` / `EXCLUDE` | `exclude_present` / `INCLUDE` | `explicit_absence_required` / `EXCLUDE` | `explicit_absence_required` / `INCLUDE` |
|---|---:|---:|---:|---:|
| single (70) | 3.814 (128) | 5.505 (128) | 3.814 (128) | 5.505 (128) |
| multi (60) | 2.465,5 (155) | 5.840 (235) | 1.904,5 (65) | 3.654 (182) |
| **negation (40)** | **10.223** (489) | 13.390 (2.367) | **1.013** (65) | 1.244 (182) |
| uncertainty (16) | 3.062 (128) | 3.062 (128) | 3.062 (128) | 3.062 (128) |
| view (20) | 6.891,5 (2.302) | 10.373 (2.430) | 6.891,5 (2.302) | 8.910,5 (2.430) |

**Đọc theo D9.** Với câu phủ định, bắt buộc `ABSENT` (`explicit_absence_required`) hạ median từ
10.223 xuống 1.013 study. Tính theo từng câu, policy này chỉ giữ median 0,199 số study mà
`exclude_present` trả (min 0,01, max 0,664; 40 câu, 28 trong đó chỉ có exclude). Phần bị loại là
study `NOT_MENTIONED` với concept bị phủ định: báo cáo không nhắc tới, chứ không phải có bệnh. Đây là
bằng chứng ở mức truy vấn cho D9, khớp lập luận từ prevalence (median `ABSENT` 5,0%). Benchmark này
không có câu nào sụp về 0, nên cái giá của `explicit_absence_required` ở đây là recall, không phải
trang kết quả rỗng. Với `single` và `uncertainty` absence policy không có tác dụng, vì không câu
nào trong hai nhóm có exclude. Include `UNCERTAIN` chỉ khớp study `UNCERTAIN` ở mọi policy (spec
§5.2), nên median và min của `uncertainty` không đổi giữa bốn cặp. Chỉ hai câu có thêm include
PRESENT (`en-045`, `vi-045`) đổi theo uncertainty policy: 3.991 với `EXCLUDE`, 4.092 với `INCLUDE`.

**Cùng một câu qua API** (`"Find edema with no pleural effusion"`, parser `llm`: include
Edema/PRESENT, exclude Pleural Effusion). Mode `label` có `total_matches` 2.760 với `exclude_present`
và 1.532 với `explicit_absence_required`; `latency_ms` 2 và 1. Mode `vector` (top 10, không áp nhãn
theo G3) cho **6/10** kết quả mang `Pleural Effusion: PRESENT` và 1/10 `UNCERTAIN`. Chỉ 3/10 thỏa ràng
buộc theo policy mặc định (Edema `PRESENT`, Pleural Effusion `ABSENT` hoặc `NOT_MENTIONED`).
`latency_ms` của lần vector đầu là 886. Đây là chỗ vector mode vi phạm ràng buộc phủ định mà luận
văn lấy làm điểm xuất phát.

## Cách đọc số

- **Gold chủ yếu do một tác giả viết** (agent Task 8; review chỉ đổi 5 câu), không có người thứ hai
  chấm độc lập toàn bộ. Ba câu bẫy (ở trên) được viết
  sau khi đã thấy output của rule parser trên heldout. Chúng không được dùng để chỉnh gì, nhưng tác
  giả đã biết rule parser sẽ sai ở đó.
- **3B, temperature 0, decoding ràng buộc bằng schema.** Enum buộc concept nằm trong 14 nhãn, nhưng
  model hay chọn nhãn theo **chữ cái đầu** của từ gốc: pacemaker / máy tạo nhịp thành Pneumothorax,
  Pneumonia hoặc Pleural Other. Model cũng gắn concept với list mà nó thấy trong ví dụ: sau ví dụ
  "không có xẹp phổi" → exclude, nó đưa câu dương có atelectasis vào exclude. Trường `language`
  thường ra `en` cho câu tiếng Việt; trường này không được chấm.
- **Rule parser dùng chung lexicon mà benchmark được viết theo.** Số của nó lạc quan với từ có
  trong lexicon và bi quan với từ ngoài lexicon (Lung Opacity, Fracture).
- **View chỉ có 2 lớp** và 20 câu heldout có view. Một view đoán thừa làm F1 đổi vài điểm.
- **Latency bỏ lần gọi đầu.** Lần gọi đầu sau khi server Ollama khởi động được ghi ở WORKLOG T16
  (5,3 s ngày 2026-10-14, 72 s ở lần nạp model đầu tiên sau khi cài).
- **Không tái lập được giữa hai lần chạy** (WORKLOG T18). Cùng prompt (`c975e4ca…`), cùng 40 câu dev,
  temperature 0, seed cố định, Ollama 0.40.3, nhưng vòng k=4 và lần chạy dev cuối cho **5/40 dự đoán
  khác nhau** (en-020, vi-002, vi-008, vi-016, vi-020). Dev view đổi 0,900 → 0,833, status 0,841 →
  0,827, concept 0,909 → 0,912. Mỗi con số LLM ở đây là một lần chạy, nên mang nhiễu cỡ đó (khoảng
  1–2 câu trên 40). Phase 7 nên chạy LLM parser ≥ 3 lần và báo cáo độ dao động. Trong cùng một tiến
  trình, `llm` và `auto` trùng nhau ở 160/160 câu heldout, nhưng ở vòng k=1 thì lệch nhau.
- **Giới hạn của nhãn, không phải của parser.** Đếm trên `search_text` (impression + findings) của
  25.000 study, nhãn resolve theo G1:
  - Report có "sternotomy wire" mà không có từ dụng cụ nào khác: Support Devices `NOT_MENTIONED`
    28/80 (`CHEXPERT_V1` riêng: 73/80). "Từ dụng cụ" là các chuỗi con `tube line catheter pacemaker
    pacer picc port device lead stent valve clip drain wire`, tìm trong text sau khi bỏ cụm
    "sternotomy wire(s)". Vì khớp chuỗi con, "portable" hay "baseline" cũng loại report đó, nên 80 là
    một tập bảo thủ.
  - "pleural plaque" hoặc "pleural calcification": Pleural Other `NOT_MENTIONED` 43/52.
  - "pulmonary congestion": Edema `UNCERTAIN` 7/11.

  Vì vậy mode `label` trả thiếu cho `en-066`, `vi-066` (dây thép xương ức), `en-044`, `en-074`,
  `vi-044` (vôi hóa/mảng màng phổi). Bộ lọc Edema `PRESENT` với `EXCLUDE` bỏ sót phần lớn study ứng
  với `en-061`, `vi-061` (sung huyết phổi). Parse của các câu này vẫn đúng.
- `semantic_query` không được chấm (văn bản tự do). Quy tắc dựng nó dùng chung cho cả hai parser
  (spec §4.2).

## Tái lập

```bash
curl -s http://localhost:11434/api/version          # 0.40.3; .env phải rỗng (WORKLOG T13)
uv run pytest -m llm                                # 1 passed, 1 xfailed (use case 6, P7)
uv run cxr eval-parser --split heldout --parsers rule llm auto --label-filter   # ghi đè parser_eval.json
uv run cxr eval-parser --split dev --parsers rule llm auto --out artifacts/metrics/parser_eval_dev.json
docker compose up -d qdrant && uv run uvicorn src.api.main:app --port 8000   # cho phần API
```

Do T18, chạy lại sẽ không cho đúng từng số ở trên. Phải dùng cùng commit, cùng Ollama và cùng model
mới so sánh được.

# Image concept probe — nhánh ảnh trong truy vấn

**Ngày:** 2026-10-23 · **Nhánh:** `feat/image-concept-probe`, merge vào `main` 2026-10-23 ·
**Git:** probe train ở `6f774bb`, sinh lại ở `667cafb` chỉ để thêm `selection.n_val` (`probe.npz`,
`corpus_scores.parquet` trùng từng byte); lưới dev, dev tham chiếu, ablation, heldout và bench chạy ở `6f774bb`
với `configs/retrieval.yaml` chỉ sửa comment (trọng số không đổi, xem *Lưới dev và ablation*) ·
**build_id corpus:** `20261011_chexpert25k_biomedclip_v1` · **build_id pool:**
`20261023_pool88k_biomedclip_v1` · **probe_id:** `20261011_chexpert25k_biomedclip_v1_probe_lam0.1` ·
**Thiết kế:** bản gốc không nằm trong git; các quyết định C1–C10 của nó ghi ở D29 trong WORKLOG cũ, xem
`git log -p WORKLOG.md` (commit `59b5db6`)

Nguồn của mọi số dưới đây (đều gitignored, đều là **file mới** của nhánh; không lệnh nào của nhánh ghi vào
`retrieval_eval.json`, `retrieval_dev.json` hay `vector_baseline.json` của `main`, spec C8):

- `artifacts/metrics/image_probe.json` (`cxr train-probe`: chọn λ, AUROC trên corpus), sha256
  `4311ce32d1b1daa167c11040ad4a91caa6858cf18699513a59525cafbca41fd6`; `data/probe_pool/pool.json`;
  `data/indexes/image_probe/probe.json`.
- `artifacts/metrics/retrieval_dev_image_probe.json` (section `grid`, `eval`, `ablation` của dev), sha256
  `b8f8ebc153fe73a27e190706a9c2a59af117c8bfa26631289e9fea1b39b5758c`.
- `artifacts/metrics/retrieval_eval_heldout_image_probe_20261023.json`: heldout **một lần**, đóng băng
  bằng `cp` ngay sau khi chạy, sha256 `af0197893734c4527adca833c8cc4e830e4c1c97ea97ad9cd0f6b7d79a4ff013`
  (trùng `retrieval_eval_image_probe.json` lúc đóng băng).
- `artifacts/metrics/vector_baseline_image_probe.json` (`cxr bench-vector`), sha256
  `df2714ab686e82c86664de70cceda546eba42909641a1960add9bacabe24eb04`.

**Tóm tắt:** S1 **PASS** (probe hơn zero-shot +0,060 AUROC), S2 **PASS** (mode `image` hơn `vector` trên
138 câu chung: nDCG@10 0,297 so với 0,259, vi phạm exclude 0,165 so với 0,319), S3 **FAIL** (lưới dev
giữ trọng số ảnh 0 trong `hybrid`), S4 **PASS** (`hybrid` heldout 0,607, giống hệt `main`). Ảnh giờ tự
trả lời được truy vấn tốt hơn baseline vector, nhưng chưa góp được gì đo thấy vào `hybrid`.

## Mục tiêu và tiêu chí

Mục tiêu: kết quả truy vấn tốt nhất có thể trong phạm vi V1 (không KG, WORKLOG D28; không finetune
encoder), với **ảnh góp tín hiệu thật** thay vì để report dẫn gần hết thứ hạng (WORKLOG P12). Cách làm:
giữ nguyên BiomedCLIP, train 14 bộ phân loại tuyến tính (probe) trên embedding ảnh của các bệnh nhân
**ngoài corpus**, rồi dùng logit "ảnh trông như có X" ở hai chỗ: mode mới `image` (chỉ ảnh, không nhãn,
không report) và một nhánh thứ ba `image` trong `hybrid`. Mode `vector`, `label`, nhánh report, bộ lọc
nhãn và evidence không đổi.

Bốn tiêu chí được viết **trước khi đo** (spec §8.3), parser `gold`, policy mặc định
`(exclude_present, EXCLUDE)`:

| # | Tiêu chí | Ngưỡng | Kết quả |
|---|---|---|---|
| S1 | Probe hơn zero-shot | macro strict AUROC probe − zero-shot > 0 trên corpus, CI 95% paired không chứa 0 | **PASS**: +0,060, CI [0,050; 0,069] (*Probe*) |
| S2 | Mode `image` hơn `vector` **trên cùng tập câu** | Trên các câu heldout mà `vector` trả kết quả: macro nDCG@10 của `image` > của `vector` **và** trung bình exclude_violation@10 của `image` < của `vector` (trên các câu có exclude trong tập đó). Số `vector` lấy từ chính lần chạy này | **PASS**: 138 câu, nDCG@10 0,297 > 0,259; violation 0,165 < 0,319 trên 26 câu (*Mode `image` trên heldout*) |
| S3 | Ảnh góp vào `hybrid` | Lưới dev chọn trọng số `image > 0` theo luật `choose` (hơn cấu hình đang chạy ≥ 0,005 nDCG@10) | **FAIL**: cấu hình `image > 0` tốt nhất hơn mốc 0,0040 < 0,005, giữ mốc (*Lưới dev và ablation*) |
| S4 | `hybrid` không tụt | heldout nDCG@10 ≥ 0,597 (0,607 của Phase 5 − 0,01) | **PASS**: 0,607 ≥ 0,597, trùng từng bit với Phase 5 (*Mode `image` trên heldout*) |

Trượt tiêu chí nào thì vẫn báo đủ số; nhánh vẫn được giữ, chỉ kết luận thay đổi (spec §8.3).

## Dữ liệu train

**Pool ngoài corpus** (`uv run cxr build-probe-pool`, ra `data/probe_pool/`): cùng luật đủ điều kiện
của corpus (frontal AP/PA, có impression, file ảnh tồn tại, một ảnh mỗi study theo view index nhỏ nhất),
bỏ **mọi bệnh nhân có trong corpus**, chỉ lấy `train/` (không bao giờ `valid/`, WORKLOG D12). Số từ
`data/probe_pool/pool.json`:

| | |
|---|---|
| Study / bệnh nhân | **88.853 / 46.682** (`studies`, `patients`) |
| View | AP 71.692 · PA 17.161 (`projection_mix`) |
| `pool_sha256` | `a9cca8aed9ed0ed2cc8328226a5f272a79aa11caf1e62b4ad7a47f0feaea2549`, trùng `probe.json:pool_sha256` |
| `corpus_images_sha256` | `634b22f92c5821a4f34428fb433751ec71e83b80a14037e370bbb280fceab914` (`data/canonical/images.parquet` dùng để loại bệnh nhân) |
| Kiểm lại trong lần viết này (chỉ đọc, polars trên `pool_images.parquet`) | 0 path `valid/`, 0 path ngoài `train/`, **0 bệnh nhân chung** với `data/canonical/images.parquet` |

**Embed pool** (`data/probe_pool/embeddings/20261023_pool88k_biomedclip_v1/shards.json`): 89 shard,
88.853 ảnh embed, 0 bỏ qua, CPU (`device: cpu`), 0,0815 s/ảnh, `started_at` 2026-10-22T17:27:34Z →
`finished_at` 19:28:16Z (7.242 s ≈ 2 giờ); cùng `model_id`/`revision` (`9f341de`)/`dim` 512 với corpus,
`cxr train-probe` từ chối pool nếu lệch.

**Nhãn của pool** sau resolve (G1, bộ lọc T11), đếm study trên `pool_labels.parquet` bằng `group_by(concept,
status)` (chỉ đọc; trùng bảng §1 của spec). Đích train theo C5: PRESENT = 1; ABSENT và NOT_MENTIONED =
0; UNCERTAIN bỏ.

| Concept | PRESENT | ABSENT | UNCERTAIN | NOT_MENTIONED |
|---|---:|---:|---:|---:|
| No Finding | 17.287 | **0** | 0 | 71.566 |
| Enlarged Cardiomediastinum | 8.773 | 26.282 | 12.306 | 41.492 |
| Cardiomegaly | 21.554 | 19.554 | 5.771 | 41.974 |
| Lung Opacity | 62.239 | 4.741 | 2.335 | 19.538 |
| Lung Lesion | 9.581 | 1.252 | 1.311 | 76.709 |
| Edema | 31.794 | 16.028 | 8.703 | 32.328 |
| Consolidation | 8.910 | 23.921 | 18.920 | 37.102 |
| Pneumonia | 4.471 | 2.880 | 15.777 | 65.725 |
| Atelectasis | 25.162 | 825 | 23.926 | 38.940 |
| Pneumothorax | 10.598 | 46.011 | 1.819 | 30.425 |
| Pleural Effusion | 43.622 | 27.128 | 5.908 | 12.195 |
| Pleural Other | 3.911 | **123** | 2.290 | 82.529 |
| Fracture | 8.883 | 4.988 | 615 | 74.367 |
| Support Devices | 69.077 | 3.088 | 407 | 16.281 |

## Probe

`uv run cxr train-probe` (`src/embeddings/probe.py`, config `configs/models.yaml:image_probe`):
logistic regression phạt L2 trên `[x, 1]` (x là vector ảnh L2-norm 512 chiều, bias không bị phạt),
Newton-IRLS bằng numpy, tối đa 50 vòng, dừng khi bước lớn nhất < 1e-6. Pool chia **theo bệnh nhân**
(`sha256("20260101:<patient_id>") mod 1000 < 100`): 9.037 study / 4.645 bệnh nhân vào val, 79.816
study vào train (đếm lại bằng `probe.patient_split` trên `pool_images.parquet`). Một λ chung cho 14
concept, chọn trong lưới **cố định trước khi đo** `{0,1; 1; 10; 100}` theo macro strict AUROC trên val
(PRESENT so với ABSENT, chỉ concept có ≥ 10 ca mỗi lớp trên val: 13 concept, No Finding không có
ABSENT), rồi fit lại trên toàn pool. Kết quả ở `data/indexes/image_probe/` (`probe.npz`,
`corpus_scores.parquet` 25.000 dòng × 14 logit, `probe.json`) và `artifacts/metrics/image_probe.json`.

**Chọn λ** (`image_probe.json:probe.selection.per_lambda`):

| λ | macro strict AUROC, pool-val |
|---:|---:|
| **0,1** ← chọn | **0,6196** |
| 1 | 0,6166 |
| 10 | 0,6114 |
| 100 | 0,6051 |

AUROC tăng đều khi λ giảm, và λ được chọn là **giá trị nhỏ nhất của lưới**: điểm tối ưu có thể nằm
dưới 0,1. Lưới được chốt trước khi đo và cố ý không nới ra sau khi thấy kết quả, nên λ nhỏ hơn chưa
được thử. Bốn λ nằm trong 0,0145 AUROC của nhau trên val, nên λ ít ảnh hưởng trong khoảng đã thử.

**Trên corpus** (25.000 study, ảnh đã chọn của mỗi study; cùng cách chấm với `cxr score-corpus`: strict
AUROC PRESENT so với ABSENT trên 14 concept đủ mẫu, bootstrap 1.000 lần seed 20260101, **paired**;
`image_probe.json:probe.comparison`). Probe chưa từng thấy bệnh nhân nào của corpus.

| Dòng | Điểm mỗi ảnh | macro strict AUROC |
|---|---|---:|
| zero-shot một prompt (mốc Phase 3) | cosine với `"chest x-ray showing {en[0]}"` | 0,599 |
| A1: prompt ensemble | cosine với trung bình vector text qua mọi dạng `en` | 0,599 |
| A2: prompt đối | cosine(dương) − cosine(`"chest x-ray showing no {en[0]}"`) | 0,640 |
| **probe** | logit `z_c` | **0,659** |

Probe − zero-shot = **+0,060, CI 95% paired [0,050; 0,069]** (`probe_minus_zero_shot`), `s1_passed: true`.

Theo concept (`comparison.<dòng>.per_concept`; train = `probe.n_train` PRESENT / 0 / UNCERTAIN bỏ trên
toàn pool; val dương / âm = `probe.selection.n_val`, PRESENT / ABSENT của pool-val, tức hai lớp mà
`selection.val_auroc` ở λ đã chọn chấm):

| Concept | train dương / âm / bỏ | val dương / âm | val AUROC | zero-shot | A1 | A2 | probe | probe − zero-shot |
|---|---|---|---:|---:|---:|---:|---:|---:|
| No Finding | 17.287 / 71.566 / 0 | 1.751 / 0 | n/a | 0,585 | 0,602 | 0,390 | **0,665** | +0,081 |
| Enlarged Cardiomediastinum | 8.773 / 67.774 / 12.306 | 924 / 2.550 | 0,671 | 0,602 | 0,602 | 0,667 | **0,655** | +0,053 |
| Cardiomegaly | 21.554 / 61.528 / 5.771 | 2.282 / 1.939 | 0,723 | 0,681 | 0,683 | 0,719 | **0,718** | +0,037 |
| Lung Opacity | 62.239 / 24.279 / 2.335 | 6.324 / 488 | 0,744 | 0,646 | 0,620 | 0,731 | **0,757** | +0,112 |
| Lung Lesion | 9.581 / 77.961 / 1.311 | 921 / 134 | 0,495 | 0,469 | 0,447 | 0,587 | **0,533** | +0,064 |
| Edema | 31.794 / 48.356 / 8.703 | 3.327 / 1.607 | 0,730 | 0,681 | 0,617 | 0,720 | **0,739** | +0,058 |
| Consolidation | 8.910 / 61.023 / 18.920 | 900 / 2.377 | 0,740 | 0,655 | 0,675 | 0,733 | **0,757** | +0,103 |
| Pneumonia | 4.471 / 68.605 / 15.777 | 417 / 308 | 0,578 | 0,564 | 0,578 | 0,644 | **0,589** | +0,025 |
| Atelectasis | 25.162 / 39.765 / 23.926 | 2.517 / 77 | 0,615 | 0,559 | 0,652 | 0,631 | **0,629** | +0,070 |
| Pneumothorax | 10.598 / 76.436 / 1.819 | 1.033 / 4.719 | 0,630 | 0,564 | 0,564 | 0,613 | **0,655** | +0,090 |
| Pleural Effusion | 43.622 / 39.323 / 5.908 | 4.452 / 2.724 | 0,827 | 0,763 | 0,761 | 0,794 | **0,827** | +0,064 |
| Pleural Other | 3.911 / 82.652 / 2.290 | 428 / 11 | 0,222 | 0,496 | 0,492 | 0,564 | **0,563** | +0,067 |
| Fracture | 8.883 / 79.355 / 615 | 919 / 458 | 0,476 | 0,520 | 0,509 | 0,565 | **0,487** | −0,033 |
| Support Devices | 69.077 / 19.369 / 407 | 7.131 / 279 | 0,602 | 0,604 | 0,580 | 0,608 | **0,650** | +0,045 |

- Probe hơn zero-shot ở **13/14** concept; **Fracture kém hơn** (0,487 so với 0,520).
- **Ba concept yếu, gần ngẫu nhiên:** Fracture 0,487, Lung Lesion 0,533, Pleural Other 0,563. Câu chỉ hỏi
  một trong các concept này thì mode `image` gần như xếp ngẫu nhiên; macro che chuyện đó nên bảng theo
  concept là bảng phải đọc.
- **A2 (prompt đối) mạnh hơn dự kiến:** 0,640, hơn zero-shot ở 13/14 concept; probe chỉ hơn A2 ở 7/14
  concept, và A2 hơn probe ở Lung Lesion (0,587 so với 0,533) và Fracture (0,565 so với 0,487). A1 không
  đổi gì (0,599). Theo C1 của spec, A chỉ là dòng đo, không vào runtime; A2 trên No Finding chỉ 0,390
  (prompt "no no finding" vô nghĩa).
- **val AUROC của Pleural Other (0,222) đứng trên 11 ca ABSENT** của pool-val (`selection.n_val`; 428
  PRESENT), vì cả pool chỉ có 123 ca ABSENT. Con số đó gần như không có thông tin; AUROC trên corpus
  (0,563) mới là số để đọc.

## Mode `image` trên heldout

`uv run cxr eval-retrieval --split heldout --parsers gold rule auto --modes label vector image hybrid
--policies negation_both --out artifacts/metrics/retrieval_eval_image_probe.json`, chạy **một lần**,
2026-10-23 02:43:32 → 02:51:33 (8 phút 2 giây), exit 0, không có lần chạy dở nào. Trước đó: Qdrant
Docker, `uv run cxr validate-indexes` 8/8 PASS, không có API server nào chạy. 152 câu có gold (8 câu
`other` bỏ, `skipped_no_gold`); benchmark sha256 `8d19c619…` (`eval.benchmark`); report index
`corpus_sha256` `60b72389…` (`eval.report_index`); fusion `rrf_k` 10, (vector 0,5, report 1, image 0)
(`eval.fusion`); `auto` = OpenAI `gpt-5-mini-2025-08-07`, prompt_sha256 `c975e4ca…` (`eval.llm`). Mode
`image` chỉ dùng ảnh: không nhãn, không report lúc xếp hạng (spec §6.1).

Bảng chính (`eval.parsers.<parser>.modes.<mode>.metrics`, `latency_ms`, `fallback_rate`; policy mặc định):

| Parser | Mode | p@5 | p@10 | nDCG@10 | nDCG nhị phân | MRR | violation@10 | returned@10 | phủ evidence | polarity | fallback | p50 / p95 ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `gold` | label | 1,000 | 1,000 | 0,531 | 1,000 | 1,000 | 0,000 | 1,000 | 0,537 | 0,849 | — | 20 / 31 |
| `gold` | vector | 0,382 | 0,359 | 0,235 | 0,371 | 0,528 | 0,319 | 0,908 | 0,510 | 0,837 | — | 181 / 200 |
| `gold` | **image** | 0,516 | 0,507 | **0,299** | 0,510 | 0,640 | **0,165** | 1,000 | 0,597 | 0,812 | — | **22 / 33** |
| `gold` | hybrid | 1,000 | 1,000 | **0,607** | 1,000 | 1,000 | 0,000 | 1,000 | 0,922 | 0,862 | — | 215 / 279 |
| `rule` | label | 0,655 | 0,657 | 0,349 | 0,657 | 0,673 | 0,063 | 0,730 | 0,561 | 0,814 | 0,000 | 17 / 26 |
| `rule` | vector | 0,297 | 0,284 | 0,179 | 0,294 | 0,433 | 0,359 | 0,789 | 0,525 | 0,793 | 0,000 | 177 / 200 |
| `rule` | image | 0,411 | 0,407 | 0,233 | 0,411 | 0,530 | 0,192 | 0,868 | 0,632 | 0,784 | 0,000 | 24 / 190 |
| `rule` | hybrid | 0,730 | 0,722 | 0,427 | 0,722 | 0,752 | 0,136 | 0,868 | 0,924 | 0,819 | 0,000 | 212 / 264 |
| `auto` | label | 0,974 | 0,974 | 0,516 | 0,974 | 0,975 | 0,000 | 0,993 | 0,534 | 0,848 | 0,000 | 20 / 26 |
| `auto` | vector | 0,376 | 0,357 | 0,233 | 0,368 | 0,518 | 0,319 | 0,908 | 0,513 | 0,827 | 0,000 | 182 / 206 |
| `auto` | image | 0,513 | 0,503 | 0,297 | 0,507 | 0,636 | 0,165 | 1,000 | 0,602 | 0,804 | 0,000 | 24 / 33 |
| `auto` | hybrid | 0,976 | 0,977 | 0,589 | 0,977 | 0,979 | 0,000 | 1,000 | 0,923 | 0,855 | 0,000 | 217 / 281 |

```
verdict: PASS {"single_p10": {"value": 1.0, "threshold": 0.7, "passed": true}, "constraint_satisfaction": {"value": 1.0, "threshold": 0.95, "passed": true}, "hybrid_gain_over_vector": {"value": 1.5791510231504176, "threshold": 0.05, "passed": true}, "passed": true}
```

- **`gold` và `rule` ở label / vector / hybrid trùng từng bit với heldout lần 2 của Phase 5**
  (`retrieval_eval_heldout_20261021_run2.json`; so dict trên `metrics`, `by_group`, `by_language`,
  `by_policy`, `per_query`, không tính latency), verdict cũng trùng. Mode `vector` không đổi (spec C2) và
  `hybrid` với trọng số ảnh 0 chạy đúng như `main`; **S4 PASS** (0,607 ≥ 0,597). Chỉ `auto` lệch (LLM
  không tái lập, WORKLOG T23, T25): `auto` label p@10 0,974 so với 0,980 ở Phase 5; p@10 = 0 ở đúng ba
  câu cũ `vi-044`, `vi-052`, `vi-079` (mode label, đọc từ `per_query`).
- **`rule` image p95 190 ms:** khi rule parse không ra concept nào, mode `image` lùi về cosine text–ảnh qua
  Qdrant (spec §6.1), nên những câu đó tốn như mode `vector`; câu VI không có concept thì trả rỗng (`rule`
  image không trả gì ở 20/152 câu, `rule` vector 32/152, đọc từ `per_query`). Đây là cách đọc đường code,
  khớp với số, không đo riêng.

**`image_vs_vector` (S2)**, từ `eval.image_vs_vector` (parser `gold`, policy mặc định), tính trên các câu
mà `vector` trả ít nhất một kết quả:

| | image | vector |
|---|---:|---:|
| Số câu chung (`n_common`) | 138 | 138 |
| nDCG@10 macro (`ndcg@10`) | **0,297** | 0,259 |
| exclude_violation@10 (`exclude_violation@10`, 26 câu chung có exclude) | **0,165** | 0,319 |
| p@10 macro trên cùng 138 câu (tính từ `per_query`) | 0,480 | 0,396 |
| Thắng / hòa / thua theo câu, image so với vector (`wins`/`ties`/`losses`) | 77 / 12 / 49 | |

`s2_passed: true`, tức **S2 PASS**: hơn ở nDCG **và** vi phạm exclude giảm gần một nửa. Theo ngôn ngữ trên
tập chung (`per_query`): EN 76 câu 0,302 so với 0,259; VI 62 câu 0,292 so với 0,260.

**14 câu `vector` không trả lời** (`vector_unanswered`, báo riêng, không gộp vào S2): đúng 14 câu VI chỉ có
exclude (`vi-023`, `vi-027`, `vi-031`, `vi-038`, `vi-043`, `vi-047`, `vi-051`, `vi-058`, `vi-071`,
`vi-078`, `vi-087`, `vi-091`, `vi-095`, `vi-098`), `semantic_query` rỗng nên vector không có gì để encode
(WORKLOG D26). Mode `image` trả đủ 10 kết quả cho cả 14 câu: nDCG@10 **0,312**, so với **0,328** của
thứ tự label (`vector_unanswered.ndcg@10`); p@10 0,764 so với 1,000 của label, vi phạm 0,164 (`per_query`).
Ảnh trả lời được thứ vector không trả lời được, nhưng xếp kém hơn một mẫu ngẫu nhiên cố định của tập đã
lọc nhãn.

**"Ảnh không có report"** (spec §8.4): `vector` và `image` không đọc nhãn hay report lúc truy vấn, nên số
heldout của chúng **chính là** chất lượng tìm kiếm trên ảnh chưa có report. Trên cả 152 câu (gold):
`image` nDCG@10 0,299, p@10 0,507, violation 0,165, returned 1,000; `vector` 0,235, 0,359, 0,319, 0,908.

**Theo nhóm và ngôn ngữ** (gold, `by_group`, `by_language`; label và hybrid có p@10 1,000 và violation 0 ở
mọi dòng):

| Tập con (n) | nDCG label | nDCG vector | nDCG image | nDCG hybrid | p@10 vector / image | violation vector / image | returned vector / image |
|---|---:|---:|---:|---:|---|---|---|
| single (70) | 0,557 | 0,309 | 0,343 | 0,664 | 0,440 / 0,517 | n/a | 1,000 / 1,000 |
| multi (60) | 0,580 | 0,199 | 0,257 | 0,635 | 0,290 / 0,407 | 0,307 / 0,144 | 0,950 / 1,000 |
| negation (40) | 0,395 | 0,140 | **0,281** | 0,435 | 0,302 / **0,655** | 0,319 / 0,165 | 0,650 / 1,000 |
| uncertainty (16) | 0,590 | 0,079 | 0,103 | 0,640 | 0,125 / 0,175 | n/a | 1,000 / 1,000 |
| view (20) | 0,570 | **0,449** | 0,432 | 0,657 | 0,620 / 0,630 | **0,100** / 0,200 | 1,000 / 1,000 |
| en (76) | 0,531 | 0,259 | 0,302 | 0,615 | 0,414 / 0,511 | 0,320 / 0,165 | 1,000 / 1,000 |
| vi (76) | 0,531 | 0,212 | 0,296 | 0,599 | 0,304 / 0,503 | 0,317 / 0,165 | 0,816 / 1,000 |

- `image` hơn `vector` ở mọi nhóm **trừ `view`** (nDCG 0,432 so với 0,449, violation 0,200 so với 0,100).
- Lợi lớn nhất ở `negation`: p@10 0,655 so với 0,302. Mode `image` xử lý exclude bằng log(1 − σ(z)) của
  probe, còn vector chỉ encode câu phủ định thành một vector text.
- `uncertainty` kém ở cả hai (image 0,103, vector 0,079): mode `image` coi include UNCERTAIN như PRESENT (spec §5.3), còn
  relevance đòi đúng UNCERTAIN.
- `image` không phụ thuộc `semantic_query`, nên EN và VI gần bằng nhau (0,302 / 0,296); `vector` tụt ở VI.

**Negation dưới hai policy** (gold, 40 câu, `by_policy`):

| Mode | `exclude_present`: p@10 / nDCG@10 | `explicit_absence_required`: p@10 / nDCG@10 | violation@10 |
|---|---|---|---:|
| label | 1,000 / 0,395 | 1,000 / 0,612 | 0,000 |
| vector | 0,302 / 0,140 | 0,082 / 0,059 | 0,319 |
| image | 0,655 / 0,281 | 0,200 / 0,124 | 0,165 |
| hybrid | 1,000 / 0,435 | 1,000 / 0,640 | 0,000 |

**Latency.** Trong tiến trình, chỉ search, không tính parse, bỏ câu đầu (n = 151, `latency_ms` của gold):
label 20 / 31, vector 181 / 200, **image 22 / 33**, hybrid 215 / 279 ms (p50 / p95). Mode `image` rẻ ngang
`label` (numpy trên 25.000 × 14 logit, không encoder, không Qdrant). Hybrid chạy cùng đường code với `main`
(nhánh ảnh trọng số 0 không chạy), nên 215 / 279 so với 207 / 259 của Phase 5 lần 2 đọc như dao động giữa các
lần chạy (Phase 5 lần 1 và lần 2 cũng lệch ở mọi ô); nguyên nhân không đo. `uv run cxr bench-vector --out
artifacts/metrics/vector_baseline_image_probe.json` (`latency`, 30 câu × 5 lần, bỏ lần đầu, n = 120, torch
14 luồng): cold 7.450 ms; warm encode 140,2 / 147,1, Qdrant 6,7 / 28,3, tổng **152,2 / 171,5 ms**. HTTP
không đo (cần server; không có server nào được bật trong lần đo này).

## Lưới dev và ablation

Nguồn: `artifacts/metrics/retrieval_dev_image_probe.json`, section `grid`, `eval`, `ablation` (38 câu dev
có gold, 2 câu `other` bỏ; policy mặc định; benchmark sha256 `8d19c619…`).

**Lưới** (`uv run cxr eval-retrieval --split dev --parsers gold --modes hybrid --grid`, 4 phút 2 giây):
`rrf_k ∈ {10, 60}` × vector `{0; 0,5; 1}` × report `{0,5; 1}` × image `{0; 0,5; 1}` = 36 cấu hình, gold,
hybrid. **Mốc là cấu hình đang chạy** `rrf_k` 10, (vector 0,5, report 1, image 0) (`grid.fusion_current`),
không phải mặc định của plan. Luật `choose`: nDCG@10 cao nhất, nhưng phải hơn mốc ≥ 0,005, ngược lại giữ
mốc. p@10 = 1,000 ở cả 36 cấu hình (lọc nhãn trước, D23). `grid.configs`, xếp theo nDCG@10:

| # | `rrf_k` | vector | report | image | nDCG@10 |
|---:|---:|---:|---:|---:|---:|
| 1 | 10 | 0 | 1 | **0,5** | **0,6605** ← cao nhất |
| 2 | 10 | 0,5 | 1 | 0,5 | 0,6583 |
| 3 | 10 | 0 | 0,5 | 0 | 0,6579 |
| 4 | 10 | 0 | 1 | 0 | 0,6579 |
| 5 | 60 | 0 | 0,5 | 0 | 0,6579 |
| 6 | 60 | 0 | 1 | 0 | 0,6579 |
| **7** | **10** | **0,5** | **1** | **0** | **0,6565** ← mốc, **chọn** |
| 8 | 60 | 0,5 | 1 | 0 | 0,6545 |
| 9 | 10 | 0 | 0,5 | 0,5 | 0,6535 |
| 10 | 10 | 0 | 1 | 1 | 0,6535 |
| 11 | 10 | 0,5 | 1 | 1 | 0,6521 |
| 12 | 60 | 0,5 | 1 | 0,5 | 0,6504 |
| 13 | 60 | 0 | 1 | 0,5 | 0,6490 |
| 14 | 10 | 1 | 1 | 0,5 | 0,6483 |
| 15 | 60 | 0,5 | 0,5 | 0 | 0,6441 |
| 16 | 60 | 1 | 1 | 0 | 0,6441 |
| 17 | 10 | 0,5 | 0,5 | 0 | 0,6406 |
| 18 | 10 | 1 | 1 | 0 | 0,6406 |
| 19 | 10 | 0,5 | 0,5 | 0,5 | 0,6392 |
| 20 | 10 | 1 | 1 | 1 | 0,6392 |
| 21 | 60 | 1 | 1 | 0,5 | 0,6375 |
| 22 | 60 | 0 | 0,5 | 0,5 | 0,6368 |
| 23 | 60 | 0 | 1 | 1 | 0,6368 |
| 24 | 10 | 0 | 0,5 | 1 | 0,6324 |
| 25 | 60 | 0 | 0,5 | 1 | 0,6317 |
| 26 | 10 | 1 | 0,5 | 1 | 0,6316 |
| 27 | 10 | 0,5 | 0,5 | 1 | 0,6300 |
| 28 | 60 | 0,5 | 0,5 | 0,5 | 0,6290 |
| 29 | 60 | 1 | 1 | 1 | 0,6290 |
| 30 | 60 | 0,5 | 1 | 1 | 0,6257 |
| 31 | 60 | 0,5 | 0,5 | 1 | 0,6254 |
| 32 | 60 | 1 | 0,5 | 0 | 0,6222 |
| 33 | 60 | 1 | 0,5 | 0,5 | 0,6220 |
| 34 | 10 | 1 | 0,5 | 0,5 | 0,6206 |
| 35 | 10 | 1 | 0,5 | 0 | 0,6170 |
| 36 | 60 | 1 | 0,5 | 1 | 0,6144 |

**Áp luật:** cao nhất là `rrf_k` 10, (0, 1, 0,5) với 0,6605; hơn mốc 0,6605 − 0,6565 = **0,0040 < 0,005**,
nên `grid.chosen` là chính mốc (`baseline_ndcg` 0,6565). `configs/retrieval.yaml` giữ nguyên `rrf_k` 10,
`leg_weights {vector: 0.5, report: 1.0, image: 0.0}`; chỉ comment ghi số của lưới này. Với trọng số 0,
nhánh image **không chạy** (spec §6.2), nên `hybrid` của nhánh này cho cùng kết quả với `main`. **S3
FAIL.**

Kiểm tính nhất quán: các dòng `image` 0 tái lập đúng số của lưới Phase 5 (`retrieval_dev.json` của `main`):
mốc 0,6565, `rrf_k` 60 (1, 1) 0,6441, `rrf_k` 10 (1, 1) 0,6406, `rrf_k` 60 (0,5, 1) 0,6545. So từng bộ
(`rrf_k`, vector, report) với chính nó ở `image` 0 (12 bộ, đọc từ `grid.configs`): `image` 0,5 hơn ở
**4/12** bộ, cả bốn đều ở `rrf_k` 10 (+0,0017 đến +0,0077); `image` 1 hơn ở **1/12** (`rrf_k` 10, vector
1, report 0,5: +0,0146). Ở `rrf_k` 60 nhánh ảnh làm tụt ở mọi bộ (tới −0,029). Lợi theo cặp lớn nhất là khi
ảnh pha loãng một nhánh vector nặng (vector 1), còn cấu hình tốt nhất là khi ảnh thay hẳn nhánh vector
(vector 0: +0,0027); ở mốc (vector 0,5) nó chỉ thêm +0,0017, và mọi chênh lệch này nằm trong cỡ nhiễu của
38 câu dev (WORKLOG T25).

**Dev tham chiếu** (`--modes label vector image hybrid --policies default`, section `eval`, fusion đang
chạy):

| Mode | p@5 | p@10 | nDCG@10 | nDCG nhị phân | MRR | violation@10 | returned@10 | p50 / p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| label | 1,000 | 1,000 | 0,524 | 1,000 | 1,000 | 0,000 | 1,000 | 18 / 28 |
| vector | 0,484 | 0,447 | 0,302 | 0,466 | 0,614 | 0,511 | 0,921 | 179 / 201 |
| **image** | 0,600 | 0,626 | **0,367** | 0,615 | 0,678 | 0,217 | 1,000 | 23 / 32 |
| hybrid | 1,000 | 1,000 | 0,657 | 1,000 | 1,000 | 0,000 | 1,000 | 212 / 268 |

label / vector / hybrid trùng dev tham chiếu của Phase 5 (0,524 / 0,302 / 0,657). `eval.image_vs_vector`
trên dev: 35 câu chung, nDCG@10 image 0,368 so với vector 0,328, thắng / hòa / thua 22 / 2 / 11,
violation 0,278 so với 0,511; 3 câu vector không trả lời.

**Ablation** (`--ablation`, section `ablation`, cùng fusion đã chọn; "lọc + chỉ X" = hybrid với mọi nhánh
khác trọng số 0, bộ lọc nhãn vẫn bật):

| Cấu hình (`ablation.configs`) | nDCG@10 | p@10 | violation@10 |
|---|---:|---:|---:|
| label | 0,524 | 1,000 | 0,000 |
| vector | 0,302 | 0,447 | 0,511 |
| image | 0,367 | 0,626 | 0,217 |
| lọc + chỉ vector (`filter+vector`) | 0,611 | 1,000 | 0,000 |
| lọc + chỉ report (`filter+report`) | **0,658** | 1,000 | 0,000 |
| lọc + chỉ image (`filter+image`) | 0,632 | 1,000 | 0,000 |
| hybrid `main` (10; 0,5 / 1 / 0) | 0,657 | 1,000 | 0,000 |
| hybrid đã chọn | 0,657 | 1,000 | 0,000 |

- Trong tập đã lọc, image đứng riêng (0,632) hơn vector đứng riêng (0,611) và hơn label (0,524), nhưng
  thua report đứng riêng (0,658).
- Hybrid đã chọn trùng hybrid `main` ở cả 38 câu (`ablation.per_query`), đúng như trọng số 0 dự báo.
  So với lọc + chỉ report: thắng / hòa / thua 12 / 19 / 7 câu, như ablation 2026-10-22 của `main`.

## Kết luận trung thực

| # | Kết quả | Số |
|---|---|---|
| S1 | **PASS** | probe 0,659 so với zero-shot 0,599 trên corpus, +0,060, CI 95% paired [0,050; 0,069]; hơn ở 13/14 concept |
| S2 | **PASS** | 138 câu chung: nDCG@10 image 0,297 so với vector 0,259 (77 thắng / 12 hòa / 49 thua), violation 0,165 so với 0,319 |
| S3 | **FAIL** | lưới dev giữ mốc; cấu hình `image > 0` tốt nhất (`rrf_k` 10, 0 / 1 / 0,5) hơn mốc 0,0040, dưới ngưỡng 0,005 |
| S4 | **PASS** | gold hybrid heldout 0,607 ≥ 0,597, trùng từng bit với `main` |

- **Ảnh tự đứng được, và đã hơn baseline của đồ án.** Không đọc nhãn hay report, mode `image` hơn `vector` ở
  nDCG@10 (0,299 so với 0,235 trên 152 câu; 0,297 so với 0,259 trên 138 câu chung) và giảm vi phạm exclude
  gần một nửa (0,165 so với 0,319). Đó là kịch bản "ảnh chưa có report" (spec §8.4) và câu trả lời có số cho
  P12: **tín hiệu ảnh** (mode `image`, ảnh đứng riêng) làm mạnh được mà không finetune encoder.
- **Nhưng ảnh còn xa nhãn.** `image` 0,299 so với label-only 0,531 và hybrid 0,607; p@10 0,507, tức khoảng
  một nửa top 10 không relevant. Ở 14 câu VI chỉ có exclude, `image` (0,312) còn kém thứ tự cố định của tập đã
  lọc nhãn (0,328); ở nhóm `view` nó kém `vector`. Với ba concept gần ngẫu nhiên (Fracture 0,487, Lung Lesion
  0,533, Pleural Other 0,563), phần điểm ảnh của concept đó gần như không mang thông tin.
- **Ảnh chưa góp gì đo thấy vào `hybrid` (S3 FAIL).** Hybrid lọc nhãn trước (D23), nên mọi nhánh chỉ xếp
  thứ tự trong một tập đã đúng; trong tập đó report đứng riêng (0,658) vẫn hơn image đứng riêng (0,632) và
  vector đứng riêng (0,611). Trong lưới, nhánh ảnh giúp chủ yếu ở `rrf_k` 10: cấu hình tốt nhất (`rrf_k` 10,
  vector 0 / report 1 / image 0,5 = 0,6605) là khi ảnh **thay** nhánh vector (+0,0027 so với cùng cấu hình
  không có ảnh), còn lợi theo cặp lớn nhất là khi ảnh pha loãng một nhánh vector nặng (vector 1: +0,0077,
  +0,0146). Lợi lớn nhất so với mốc (+0,0040) vẫn dưới luật min-gain viết trước. `retrieval.yaml` giữ `image: 0.0`, nhánh không chạy, nên
  hybrid của nhánh git này là hybrid của `main`.
- **Prompt đối (A2) lấy được phần lớn phần lợi của probe trên AUROC** (0,640 so với 0,659, probe chỉ hơn A2
  ở 7/14 concept). Theo C1 nó chỉ là một dòng đo; dùng A2 lúc chạy là một thí nghiệm chưa làm.
- **Giới hạn của chính phép đo:** λ = 0,1 nằm ở **mép** lưới chọn trước `{0,1; 1; 10; 100}` và AUROC val
  còn tăng khi λ giảm, nên probe có thể chưa ở điểm tốt nhất; lưới cố ý không nới sau khi đo. Lưới hybrid
  36 cấu hình chọn trên 38 câu dev (T25). S2 không có khoảng tin cậy: `eval-retrieval` không bootstrap trên
  câu, chỉ có đếm thắng / hòa / thua. `auto` không tái lập. Latency của mode `image` chỉ đo trong tiến trình,
  không qua HTTP.
- **Thiên lệch của phép đo (spec §8.5).** Relevance là nhãn trích từ report (`configs/evaluation.yaml` trục
  1–2). Probe học dự đoán chính các nguồn nhãn đó, trên bệnh nhân khác, nên không rò rỉ (pool và corpus không
  chung bệnh nhân nào, kiểm lại ở *Dữ liệu train*); nhưng thứ S1 và S2 đo vẫn là "ảnh đoán nhãn report tốt
  đến đâu", một proxy dựa trên text, và probe có lợi thế cấu trúc so với zero-shot vì được train đúng trên
  định nghĩa relevance. Trục độc lập duy nhất vẫn là human review của Phase 7.

## Tái lập

```bash
docker compose up -d qdrant
# Pool ngoài corpus: data/probe_pool/{pool_images,pool_labels}.parquet + pool.json
uv run cxr build-probe-pool
# Embed pool trên CPU (≈2 giờ, shards.json đo 7.242 s); shard đã có thì được bỏ qua (D14)
uv run cxr embed --images data/probe_pool/pool_images.parquet --out data/probe_pool/embeddings \
    --build-id 20261023_pool88k_biomedclip_v1
# Probe: data/indexes/image_probe/ + artifacts/metrics/image_probe.json (S1)
uv run cxr train-probe
uv run cxr validate-indexes          # 8 x PASS (thứ 8: image_probe)
uv run pytest -m integration tests/integration/test_image_probe_live.py -q
# Dev: lưới (S3), rồi dev tham chiếu và ablation với retrieval.yaml đã chốt; cùng một file, ba section
uv run cxr eval-retrieval --split dev --parsers gold --modes hybrid --grid \
    --out artifacts/metrics/retrieval_dev_image_probe.json
uv run cxr eval-retrieval --split dev --parsers gold --modes label vector image hybrid --policies default \
    --out artifacts/metrics/retrieval_dev_image_probe.json
uv run cxr eval-retrieval --split dev --parsers gold --ablation \
    --out artifacts/metrics/retrieval_dev_image_probe.json
# Heldout, một lần (S2, S4; auto ≈0,04 USD, câu hỏi đi ra OpenAI), rồi đóng băng ngay
uv run cxr eval-retrieval --split heldout --parsers gold rule auto --modes label vector image hybrid \
    --policies negation_both --out artifacts/metrics/retrieval_eval_image_probe.json
cp artifacts/metrics/retrieval_eval_image_probe.json \
    artifacts/metrics/retrieval_eval_heldout_image_probe_20261023.json
sha256sum artifacts/metrics/retrieval_eval_heldout_image_probe_20261023.json
# af0197893734c4527adca833c8cc4e830e4c1c97ea97ad9cd0f6b7d79a4ff013
uv run cxr bench-vector --out artifacts/metrics/vector_baseline_image_probe.json
```

Không lệnh nào ở trên dùng `--out` mặc định của `eval-retrieval` hay `bench-vector` (file của `main`). Các số
đếm theo câu (138 câu chung theo ngôn ngữ, p@10 trên tập chung, 14 câu chưa trả lời, 20/152 câu `rule`) đọc
từ file đóng băng qua `per_query` bằng script chỉ đọc, ví dụ:

```bash
uv run python -c "import json; g=json.load(open('artifacts/metrics/retrieval_eval_heldout_image_probe_20261023.json'))['eval']['parsers']['gold']['modes']; print(sorted(q for q,m in g['vector']['per_query'].items() if m['returned@10']==0))"
```

`gold` và `rule` tái lập được với cùng commit, cùng build, cùng report index và cùng probe
(`probe.json` sha256 `aaa7cc00…`; bản lúc chạy heldout là `e9deb1ae…`, chỉ khác ở `selection.n_val`,
`git_commit`, `trained_at`); `auto` chạy lại sẽ không ra đúng từng số (WORKLOG T23, T25).

# Model decision — Phase 2 gate

**Ngày:** 2026-10-03 · **Git:** embed tại `0cb15d43cf99fd95083e92e761e5e106ea080638`, báo cáo `--rescore` tại `07314cdca7c4d1277c00e8919814a2cd982931f0` · **Mẫu:** manifest rank 200–1199 (1.000 study, không có ca `valid/`; `patient22720/study1` → `patient41387/study4`)

Nguồn của mọi số dưới đây: `artifacts/metrics/model_spike.json`, sinh bởi
`uv run cxr spike-models --rescore` sau ba lần embed riêng tiến trình
(`--models biomedclip`, `--models chexzero`, `--models xrayclip`). Chỉ báo cáo
`--rescore` là quyết định: một lần chạy `--models <tên>` đơn lẻ ghi đè file đó bằng
báo cáo chỉ có một model.

Thư viện (`library_versions` trong JSON): torch 2.4.1, torchvision 0.19.1, open_clip_torch
2.24.0, transformers 4.44.2, numpy 2.5.3.

## Kết quả

**BiomedCLIP thắng** với macro strict AUROC **0,636** (CI 95% bootstrap 0,607–0,663).
XrayCLIP đứng thứ hai với 0,616. Hiệu là **0,0198**, dưới ngưỡng tie 0,02 chỉ **0,0002**, nên
luật tie-break chọn model rẻ hơn: BiomedCLIP nhanh hơn XrayCLIP 6,9 lần trên CPU (0,093 so với
0,637 s/ảnh). License MIT (thay vì CC-BY-NC-4.0) là lợi ích đi kèm, không phải đầu vào của quyết
định: bước license không được xét vì chi phí chênh xa ngưỡng 10%. CheXzero trượt hard gate
(0,461 ≤ 0,55).

CI 95% của hiệu BiomedCLIP − XrayCLIP là −0,015–0,053 (chứa 0). `top2_diff` được tính cho hai
model có macro cao nhất trong mọi model đã chấm, không riêng các model qua hard gate; ở đây đúng
là cặp đứng đầu của gate. Bootstrap chỉ để báo cáo, không tham gia gate.

## Bảng tổng hợp

Đo trên CPU, 14 luồng torch, batch 16. s/ảnh là trung bình warm (bỏ batch đầu). Dự báo 25k =
s/ảnh × 25.000. Peak RSS là peak working set của cả tiến trình embed.

| Model | Macro strict (CI 95%) | Macro lenient (14 concept) | s/ảnh CPU | Dự báo 25k | ms/query | Peak RSS | Weights | Dim | License |
|---|---|---|---|---|---|---|---|---|---|
| **biomedclip** | **0,636** (0,607–0,663) | 0,609 | 0,093 | 0,644 h | 158 | 1.960 MiB | 784.611.126 B | 512 | MIT |
| chexzero | 0,461 (0,432–0,489) | 0,540 | 0,031 | 0,215 h | 42 | 1.494 MiB | 353.548.709 B | 512 | không ghi |
| xrayclip | 0,616 (0,587–0,645) | 0,626 | 0,637 | 4,421 h | 19 | 1.568 MiB | 603.029.989 B | 512 | CC-BY-NC-4.0 |

Cả ba: 0 ảnh lỗi / 1.000, dim ảnh = dim text = 512, vector 25k = 51.200.000 B (float32).
Thời gian load model: 5,0 s / 3,6 s / 2,8 s.

Macro lenient lấy trung bình trên 14 concept, còn macro strict trên 11 concept đủ mẫu, nên hai
cột không cùng tập concept và không so trực tiếp được; macro lenient không tham gia gate. Xếp
theo lenient, XrayCLIP đứng trên BiomedCLIP (ngược với strict). Trên cùng 11 concept
(`eligible_concepts`), lenient là 0,613 (BiomedCLIP), 0,528 (CheXzero), 0,627 (XrayCLIP), tính
từ `auroc_lenient` từng concept trong `model_spike.json`.

## AUROC strict theo concept

Prompt: `"chest x-ray showing {}"` với `concepts.yaml:<c>.en[0]`. Dương = `PRESENT`, âm strict =
`ABSENT`; cần ≥ 10 mỗi phía. 11 concept đủ mẫu tạo nên macro.

| Concept | n_pos / n_neg_strict | biomedclip | chexzero | xrayclip |
|---|---|---|---|---|
| No Finding | 149 / 0 | thiếu mẫu | thiếu mẫu | thiếu mẫu |
| Enlarged Cardiomediastinum | 97 / 248 | 0,614 | 0,435 | 0,564 |
| Cardiomegaly | 276 / 177 | 0,669 | 0,494 | 0,654 |
| Lung Opacity | 726 / 48 | 0,660 | 0,326 | 0,593 |
| Lung Lesion | 97 / 8 | thiếu mẫu | thiếu mẫu | thiếu mẫu |
| Edema | 408 / 141 | 0,690 | 0,602 | 0,714 |
| Consolidation | 115 / 244 | 0,678 | 0,346 | 0,670 |
| Pneumonia | 49 / 27 | 0,587 | 0,426 | 0,584 |
| Atelectasis | 287 / 12 | 0,478 | 0,322 | 0,559 |
| Pneumothorax | 135 / 462 | 0,603 | 0,583 | 0,598 |
| Pleural Effusion | 538 / 253 | 0,769 | 0,540 | 0,719 |
| Pleural Other | 43 / 1 | thiếu mẫu | thiếu mẫu | thiếu mẫu |
| Fracture | 93 / 44 | 0,642 | 0,364 | 0,454 |
| Support Devices | 807 / 25 | 0,607 | 0,634 | 0,670 |

Vài concept có rất ít âm strict (Atelectasis 12, Support Devices 25, Pneumonia 27), nên AUROC
từng concept ở đó nhiễu; macro trên 11 concept mới là chỉ số của gate.

## Luật gate và lý do

`gate.reasons`, nguyên văn:

```
chexzero: rejected (macro strict AUROC 0.4610196176554344 <= 0.55)
top two within 0.0198 AUROC; biomedclip is cheaper
selected biomedclip (macro strict AUROC 0.636)
```

Luật (cố định trước khi chạy, `src/evaluation/zero_shot.py:gate`):

1. Hard gate — trượt một điều là loại: chạy CPU; 0 ảnh lỗi; dim ảnh == dim text;
   macro strict AUROC **> 0,55**; dự báo 25k **≤ 8 h** CPU.
2. Trong số còn lại, macro strict cao nhất thắng.
3. Nếu hai model đầu cách nhau **< 0,02**: chọn s/ảnh CPU thấp hơn.
4. Nếu chi phí cũng hòa (chênh **< 10%**): license thoáng hơn (MIT > không ghi > CC-BY-NC-4.0).

Ở đây bước 3 quyết định (hiệu 0,0198 < 0,02; dòng reason in làm tròn thành `0.020`). Bước 4
không được xét. Kết quả không phụ thuộc vào tie-break: không có nó, bước 2 cũng chọn BiomedCLIP
vì macro cao hơn. Bootstrap (`n_boot` 1.000, `seed` 20260101 trong JSON, paired) chỉ để báo cáo,
không tham gia gate.

## Cảnh báo khi đọc số

- Nhãn là CheXbert/CheXpert labeler sinh từ report, không phải bác sĩ; 200 ca nhãn bác sĩ bị
  loại có chủ đích (không tune trên test).
- XrayCLIP: dữ liệu train của checkpoint HF không được công bố. Encoder trong paper CheXagent
  train trên MIMIC-CXR/PadChest/BIMCV (không có CheXpert), nhưng checkpoint này có kiến trúc
  khác. Contamination chưa xác nhận, không loại trừ.
- XrayCLIP: processor resize thẳng về 512×512, **không giữ tỉ lệ khung** (`do_center_crop`
  false), nên ảnh không vuông bị bóp méo.
- CheXzero: 1 checkpoint, không ensemble 10 như paper; decode ảnh bằng PIL thay vì cv2.
  Thứ tự tiền xử lý theo repo gốc: decode màu → RGB → letterbox LANCZOS lên canvas `L` 320 →
  thang 0–255 → normalize → resize 224.
- CheXzero dưới 0,5 ở 7/11 concept (thấp hơn ngẫu nhiên, không phải phẳng quanh 0,5), và
  trên cùng 11 concept lenient (0,528) cao hơn strict (0,461). Một lần chấm lại tạm thời, ngoài gate, trên **chính
  embedding ảnh đã cache** của CheXzero bằng cặp prompt dương/âm (`"{c}"` vs `"no {c}"`) cho
  điểm cao hơn. Lần đó khác cả giao thức lẫn cách viết concept, không nằm trong
  `model_spike.json`, và script không được giữ lại, nên không so sánh được với bảng trên. Nó chỉ
  gợi ý rằng tiền xử lý ảnh của CheXzero nhiều khả năng không hỏng, và giao thức một prompt dương
  có thể bất lợi cho model này. Gate giữ nguyên (Q3 của spec: đo đúng phép vector mode sẽ chạy);
  việc có xem lại giao thức chấm hay không đang chờ người dùng (WORKLOG P4).
- BiomedCLIP: `config.json` của text tower tải từ repo không pin
  (`microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract`, chỉ là kiến trúc; weights đã pin).
- 14 prompt đơn concept, không phải 30–50 query như §17 (query phủ định/đa concept cần parser
  Phase 4).
- Test `model` in các cảnh báo của bên thứ ba, đã biết và vô hại: `FutureWarning` của timm
  (`timm.models.layers`), `FutureWarning` `torch.load` `weights_only` (từ open_clip và từ
  adapter CheXzero), cảnh báo symlink của HF hub trên Windows.

## Pin

| | |
|---|---|
| model | `biomedclip` |
| model_id | `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` |
| framework | open_clip_torch 2.24.0 (`library_versions`), nạp từ snapshot HF đã pin (loader `hf-hub:` bỏ qua revision) |
| revision | `9f341de24bfb00180f1b847274256e9b65a3a32e` |
| image_size | 224 |
| resize | open_clip `image_transform`: cạnh ngắn, bicubic, center crop |
| mean | `[0.48145466, 0.4578275, 0.40821073]` |
| std | `[0.26862954, 0.26130258, 0.27577711]` |
| context_length | 256 |
| tokenizer | `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224@9f341de24bfb00180f1b847274256e9b65a3a32e` |
| text prompt | `"chest x-ray showing {}"` |
| dim | 512 (đo từ output; `models.yaml:embedding_dim` vẫn `null`, D6) |

# WP2 — adapter `cxrclip` và gỡ gắn cứng encoder phía offline (phiên chính, 2026-10-23)

Nhánh tích hợp `feat/label-aware-image-retrieval`, commit `b71e587` (adapter) và `e5904e4` (CLI).

## Checkpoint

| | |
|---|---|
| Tar gốc | `data/cache/models/cxrclip/swint_m.tar`, 1.635.490.293 B, sha256 `a2e5dec1…c13f`, tải 12 phút từ CDN Kakao |
| Bản chuyển | `data/cache/models/cxrclip/swint_m.pt`, 546.876.850 B, sha256 **`743e2333…a5d5`** (pin trong `models.yaml:cxrclip.revision`); chỉ `model` + 4 section config (`model`, `tokenizer`, `transform`, `base`) + `source`; nạp bằng `torch.load(weights_only=True)` |
| Lần chuyển đầu | sha `2c3c0e4a…` **sai**: `OmegaConf.to_container` chỉ chuyển node ngoài, `transform.train.RandomResizedCrop.scale` còn là `ListConfig` → `weights_only` từ chối. Sửa `convert_checkpoint` chuyển đệ quy mọi node; hash đổi, pin lại |
| Lệnh chuyển | `uv run --with omegaconf python -c "from pathlib import Path; from src.embeddings.cxrclip import convert_checkpoint as c; print(c(Path('data/cache/models/cxrclip/swint_m.tar'), Path('data/cache/models/cxrclip/swint_m.pt')))"` |

## Kiểm fidelity

- `uv run pytest -m model tests/integration/test_encoders.py -k cxrclip`: **2 passed**. `test_cxrclip_matches_the_authors_recipe` dựng lại đúng công thức của repo gốc (`SwinModel.from_pretrained` + `AutoModel.from_pretrained` Bio_ClinicalBERT + hai `nn.Linear(768, 512)`, `load_state_dict(ckpt["model"], strict=False)`, Swin token 0, text token `attention_mask.sum(-1) − 1`) rồi so với adapter trên 10 ảnh corpus và 5 câu template: **cosine ≥ 0,999 ở mọi cặp**.
- Tải thêm để chạy test đó: `pytorch_model.bin` của Swin (≈110 MB) và Bio_ClinicalBERT (≈440 MB) vào HF cache; adapter sản phẩm không cần hai file này.
- Smoke trên dữ liệu thật (CPU, 14 luồng, batch 32, bỏ batch đầu): nạp 4,9 s; **0,0345 s/ảnh** (32 ảnh 1,1 s); dim 512, norm 1,0. Cos ảnh–chữ trong khoảng −0,24…0,37. Cos chữ–chữ zero-shot: `pleural effusion` ↔ `no pleural effusion` **0,903**, `cardiomegaly` ↔ `cardiomegaly and no pleural effusion` 0,859: model gốc gần như không phân biệt phủ định — đúng điểm fine-tune phải sửa.
- Tokenizer: `AutoTokenizer` trên snapshot Bio_ClinicalBERT không có `tokenizer_config.json` → `BertTokenizer` với `do_lower_case=True` dù vocab là cased (28.996 từ). Repo gốc nạp y hệt nên giữ nguyên; ghi trong `preprocessing.do_lower_case`.

## Gỡ gắn cứng

- `cxr embed --encoder <tên> --scope <gallery>`; `shards.json` ghi `encoder` (tên candidate, không phải tên adapter, để bản fine-tune `cxrclip_ft1` phân biệt được với `cxrclip`).
- `cxr train-probe --encoder --corpus-build-id`: không cờ → đọc `build_manifest.json` như cũ; có cờ → đọc `shards.json` của build chỉ định. Thêm kiểm `ENCODER MISMATCH` (revision encoder ≠ revision build) trước khi train. `probe.json` thêm `encoder.name`, `source: "probe"`.
- `tests/unit/test_spike.py::test_candidates_include_the_three_spike_models`: nới từ "đúng ba" thành "ba model gate ⊆ candidates ⊆ ba model + cxrclip*".
- Bit-identity của đường mặc định: chạy trong `notes/wp2_chain.log` (`train-probe --out data/indexes/image_probe_check …` rồi `cmp` với `data/indexes/image_probe/`). Kết quả: xem log, chép vào đây khi xong.

## Chuỗi nền (bắt đầu 2026-10-23 ~20:50 giờ máy, `notes/wp2_chain.log`)

1. `cxr embed --encoder cxrclip --build-id 20261023_chexpert25k_cxrclip_v1 --batch-size 32` (ước 25.000 × 0,035 s ≈ 15 phút).
2. `cxr train-probe --out data/indexes/image_probe_check --report artifacts/metrics/image_probe_check.json --n-boot 20` + `cmp`.
3. `cxr embed --encoder cxrclip --images data/probe_pool/pool_images.parquet --out data/probe_pool/embeddings --scope pool88k --build-id 20261023_pool88k_cxrclip_v1 --batch-size 32` (ước 88.853 × 0,035 s ≈ 52 phút).

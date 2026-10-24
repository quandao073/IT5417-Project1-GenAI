# V1.1 — bảng trạng thái từng bước (nguồn sự thật về tiến độ)

> **Cách dùng.** Một bước một lúc. Trước khi bắt đầu bước nào: đổi trạng thái của nó thành `🔄` và ghi giờ.
> Sau khi xong: `✅` + kết quả (commit, file, số đo chính, log). Lỗi/chặn: `⛔` + lý do. Bước chưa tới: `⏳`.
> Mỗi phiên mới (hoặc chat mới) đọc theo thứ tự: `WORKLOG.md` (*Trạng thái*, *Quyết định* D30–D36, *Bẫy*) →
> **file này** → spec `docs/superpowers/specs/2026-10-23-label-aware-image-retrieval-design.md` →
> plan `docs/superpowers/plans/2026-10-23-label-aware-image-retrieval.md` (chi tiết từng task) → mục của bước đang làm.
> Chỉ **một** dòng được đánh dấu **TIẾP THEO**. File này nằm trên nhánh tích hợp; commit cùng các mốc, không commit từng lần sửa.

**Cập nhật lần cuối:** 2026-10-24 tối — **M1 đạt** (WP5 Task 15–17 + Task 23, `notes/lair-runs.md`). Fine-tune chạy song song trên Colab (người dùng).

**Nhánh tích hợp:** `feat/label-aware-image-retrieval` (HEAD xem `git log -1`), tách từ `main` 90cd62a. Không push, không đụng `main`; squash/PR khi người dùng gọi `/commit`.

## Mốc

| Mốc | Nghĩa | Trạng thái |
|---|---|---|
| M1 | Hệ thống chạy `image_joint` / `image_decomposed` với CXR-CLIP zero-shot; `validate-indexes` PASS; `vector`/`hybrid`/`image` trùng từng bit | ✅ 2026-10-24: `validate-indexes` 15/15, integration 4 passed, mode cũ trùng từng giá trị, API thật phục vụ `image_joint`/`image_decomposed`/`/studies`/CORS; `notes/lair-runs.md` |
| M2 | Heldout một lần với mọi mode, K1–K7, WORKLOG + `docs/image-retrieval.md` | ⏳ |

## Các bước

| # | Gói | Bước | Trạng thái | Kết quả / nơi xem |
|---|---|---|---|---|
| B1 | WP0 | Spec + plan + lộ trình lưu file; H1 `templates.py`, H2 `training.yaml`; WORKLOG D30–D34 | ✅ 2026-10-23 | commit `4ee7f1b`, `c2f3334`, `6b13159`; 10 test template |
| B2 | WP2 | Adapter `cxrclip` (pin sha256 bản chuyển `743e2333…`), fidelity ≥ 0,999 so với công thức gốc | ✅ | commit `b71e587`; `uv run pytest -m model -k cxrclip` 2 passed; `notes/lair-encoder.md` |
| B3 | WP2 | `embed --encoder/--scope`, `train-probe --encoder/--corpus-build-id`, `shards.json.encoder`, `probe.json.source` | ✅ | commit `e5904e4`; 936 passed |
| B4 | WP1 | `template_bench.py` + `cxr eval-image-bench` (P@k, hit@k, HNRR@k, CI95 bootstrap, gallery corpus/corpus5k/pool-val) | ✅ merge `56798f8` + fix `5ff3221` | 957 passed; `notes/lair-bench.md` |
| B5 | WP2 | Chuỗi nền: (a) embed corpus CXR-CLIP → (b) kiểm bit-identity probe BiomedCLIP → (c) embed pool CXR-CLIP | ✅ (a) exit 0 (16 phút, 25 shard); (b) **IDENTICAL** `probe.npz` và `corpus_scores.parquet`; (c) xong 2026-10-23 14:55Z, 88.853 ảnh, 89 shard | `notes/wp2_chain.log`; shard ở `data/indexes/embeddings/20261023_chexpert25k_cxrclip_v1/`, `data/probe_pool/embeddings/20261023_pool88k_cxrclip_v1/` |
| B6 | WP6a | Dòng mốc BiomedCLIP: joint + probe-decomposed, gallery corpus và corpus5k | ✅ 20:58 | `artifacts/metrics/image_bench_biomedclip_joint_{corpus,corpus5k}.json`, `image_bench_probe_biomedclip_{corpus,corpus5k}.json`; log `notes/wp6a_biomedclip.log`; **chưa đọc số** → đọc ở B12 |
| B7 | WP3+4 | Agent `train` (`.worktrees/train`, `feat/lair-train`): captions, cache, venv `training/` (torch cu124), loss, LoRA, train loop | ✅ Task 8–12 xong, người dùng duyệt, **merge `7c94d63`** (959 passed, ruff sạch) | commit `bb6e9e1`, `28bbc71`, `5629060`, `f5a3910`, `84742c2` (+2 merge); xác minh lại ở phiên chính: `uv run --project training pytest training/tests -q` 30 passed, `uv run pytest -q` 959 passed, ruff sạch; venv train `2.4.1+cu124 True`; `notes/lair-train.md` (trong worktree) |
| B8 | WP8 | Agent `ui` (`.worktrees/ui`, `feat/lair-ui`): Next.js, 6 mode, cards, detail, compare, Playwright | ✅ Task 18–20 đã review; **merged** `7361313` (web/ trùng byte với `c283531`) | commit `27e83c4`, `7c6477d`, `c283531`; xác minh lại ở phiên chính: `pnpm lint` sạch, `pnpm test:e2e` **16 passed**; agent báo `docker build` OK (196 MB). Mở: API chưa có CORS (web :3000 gọi API :8000 sẽ bị chặn); 2 mode mới luôn hiện kể cả khi API cũ trả 422; pnpm 12 cài bằng `npm i -g` (Node 25 không có corepack); `nodeLinker: hoisted` để tránh lỗi symlink Windows. `notes/lair-ui.md` |
| B9 | WP5+9 | Agent `serve` (`.worktrees/serve`, `feat/lair-serve`): manifest ảnh, `index --role image`, AppState, 2 mode mới, `/studies/{id}`, `build-image-scores`, `eval-retrieval` mode mới, protocol human review, compose | ✅ Task 13–17 (`ee0466b`, `9c3ce69`, `4acd4c9`, `9342d59`, `6077698`); Task 21–22 (WP9) chưa làm | commit `ee0466b`, `9c3ce69`; xác minh lại: 955 passed, ruff sạch. Mode tắt riêng khi thiếu manifest / collection / scores, API vẫn ready. Mở: report index chỉ lưu `search_text` → `/studies` trả impression/findings null hoặc đọc `reports.parquet`; merge vào nhánh tích hợp sẽ xung đột `main.py` (khối `index`, `validate-indexes`, argparse) |
| B10 | WP6a | Sau B5: `train-probe --encoder cxrclip`; bench joint CXR-CLIP (corpus, corpus5k, pool-val); probe-CXR-CLIP decomposed | ✅ 2026-10-24: probe-CXR-CLIP (pool-val 0,661), bench joint/decomposed/probe cả hai encoder trên corpus, corpus5k, pool-val | `notes/lair-runs.md`, `artifacts/metrics/image_bench_*`, `artifacts/metrics/m1_baseline_table.txt` |
| B11 | tích hợp | Merge WP3+4 → WP5 → (WP8 riêng) vào nhánh tích hợp; `pytest`, `ruff`; dọn worktree | ✅ cả ba đã merge (`7c94d63`, `ff56ddc`, `7361313`), 978 passed; còn dọn worktree | thứ tự plan: WP5 trước WP4; WP8 khi xong |
| B12 | WP6a/M1 | `cxr index --role image`, `build-image-scores --encoder cxrclip`, `validate-indexes` 15 PASS, `pytest -m integration`; **đọc bảng dòng mốc** (B6 + B10 + decomposed CXR-CLIP) và ghi `notes/lair-runs.md` | ✅ 2026-10-24 **M1**: collection ảnh, H6 (`data/indexes/image_scores`), 15/15 PASS sau khi đóng dấu lại manifest BiomedCLIP (WORKLOG T35), dev 6 mode `retrieval_dev_lair.json` | → **M1** |
| B13 | WP4 | cache pool thật; smoke train 5k ảnh 1 epoch (GPU) | ✅ đã review; VRAM vượt 8 GB → sửa ở B13c | cache: 88.853 ảnh, 4,46 GB, 198 s, `pool_sha256` trùng, n_val 9.037. Lần smoke đầu lỗi DataLoader trên Windows (lambda không pickle được) → sửa `88069ba` + test. Smoke lần 2: 39 bước, 273 s (gồm chấm pool-val), loss 7,50 → 7,43, không NaN, val P@5 single/pair/neg 0,617/0,204/0,271 (mean 0,364), `max_memory_allocated` **8.661 MiB > 8.188 MiB GPU** → tràn sang RAM dùng chung. `notes/b13_smoke.log` (`runs/smoke_ft1/` đã xóa) |
| B13c | WP4 | Tối ưu train cho GPU laptop: bf16 (bỏ GradScaler), gradient checkpointing hai tháp (giữ batch 128), cắt gradient 1,0, bỏ qua loss không hữu hạn (dừng nếu > 10/epoch), `--resume` + trộn theo seed từng epoch, giữ máy không ngủ, đo riêng train/val/s mỗi bước/chờ dữ liệu/VRAM; TF32 + cudnn.benchmark | ✅ **chờ review** | code `313f4ea` (35 test training, 959 test chính). Smoke 5k/1 epoch: VRAM train **4.616 MiB** (trước 8.661), val 2.327; epoch **119 s** (trước 273): train 78,9 s = 39 bước × **1,33 s/bước**, trong đó chờ dữ liệu 24,6 s; val 40 s; bf16; 0 bước bỏ; val P@5 mean 0,363 (trước 0,364). Ước epoch đầy đủ (79.115 dòng train, 618 bước): ~14 phút, 10 epoch ~2,4 giờ/lần. `notes/b13c_smoke.log` (thư mục `runs/smoke_ft1*` đã xóa 2026-10-23, số liệu nằm trong log) |
| B13b | WP4 | **Mới (D35), dựng trên FT6 (D36):** `--lora-targets`, train đầu chiếu + `logit_scale`, câu 3 ý trong `CaptionSampler`; thêm FT4/FT5 vào `training.yaml` | ⏳ | |
| B14 | WP6b | Dò lr {1e-5, 3e-5, 1e-4} cho **FT6** (D36; FT1 công thức bài báo chạy ở lr FT6 chọn), chọn trên pool-val (D33), **trên Colab Pro** | 🔄 FT1 lr 3e-5 dừng sau 2 epoch (sụp, P@5 0,255 < zero-shot 0,377) → chẩn đoán (WORKLOG *FT1 kẹt ở ln 128*) → FT6 thử ngắn trên laptop ✅ không sụp, TB 0,386, phủ định 0,267 → 0,358 (WORKLOG *Sửa B14*). Chờ người dùng chạy notebook Colab | `runs/ab_ft6/`, `runs/ab_ft6_scale24/`, `runs/zero_shot_val.json`, `notes/b14_ab.log`; Colab: `training/colab/README.md`, gói `data/colab_bundle/cxr_v11` |
| B14c | WP4 | Bộ Colab: `training/colab/pack.py` (gói + manifest sha256), `colab_run.py` (setup / check-env / train / gate / summary), `train_colab.ipynb`, README; `training/zero_shot_val.py`; `val_text_cos` mỗi epoch | ✅ commit `d355a27`; notebook dựng venv Python 3.12 từ `pylock.colab.toml` (T32), chạy T4 (fp16, không checkpointing) và A100 (bf16); tập dượt trong container Linux GPU: zero-shot trùng laptop, train 15 bước OK | WORKLOG *Colab: môi trường tái lập*; 52 test training |
| B15 | WP6b | Ở lr FT6 chọn, dựng trên FT6 (D36), trên Colab: FT4 (LoRA rộng + đầu chiếu), FT5 (FT4 + câu 3 ý), FT3 (λ 0, ablation); FT2 (+impression) ghép lên biến thể thắng | ⏳ | |
| B16 | WP6b | Embed corpus + pool bằng từng checkpoint (GPU qua venv `training/`), `build-image-scores` từng biến thể, bench pool-val → chọn joint/decomposed (D33); bench corpus → K1–K3 | ⏳ | |
| B17 | WP7 | Trỏ phục vụ vào biến thể đã chọn; heldout **một lần** (khai báo trong WORKLOG trước khi chạy); K4–K7; AUROC `valid/` | ⏳ | → **M2** |
| B18 | WP7 | `docs/image-retrieval.md`, `docs/model-decision.md`, README, WORKLOG (gom `notes/lair-*.md`) | ⏳ | |

## Đang chạy nền (kiểm tra bằng `tail`/`grep` log, không chạy lại khi chưa xem log)

Không có việc nào chạy nền (2026-10-24). Lượt train dài chạy trên Colab (`training/colab/README.md`).

## Agent / worktree

Đã gỡ 2026-10-24 sau khi merge hết vào nhánh tích hợp: bench `56798f8`, train `7c94d63`, serve `ff56ddc`, ui `7361313`. Việc còn lại của WP5/WP9 (Task 15–17, 21–22) làm thẳng trên nhánh tích hợp.

## Lệnh tiếp tục (chạy ở cây chính)

```bash
# Kiểm tra sức khỏe nhanh
uv run pytest -q && uv run ruff check src tests training  # 978 passed (2026-10-24)
uv run --project training pytest training/tests -q        # 49 passed
# B10: chạy khối *Script B10* ở cuối file này (CPU)
# B14/B15: trên Colab, training/colab/README.md (gói: training/colab/pack.py)
# B12 (M1)
uv run cxr index --role image --build-id 20261023_chexpert25k_cxrclip_v1
uv run cxr build-image-scores --encoder cxrclip --corpus-build-id 20261023_chexpert25k_cxrclip_v1 \
    --pool-build-id 20261023_pool88k_cxrclip_v1 --out data/indexes/image_scores --report artifacts/metrics/image_scores_cxrclip.json
uv run cxr validate-indexes && uv run pytest -m integration -q
```

**TIẾP THEO:** người dùng review M1 (WP5 Task 15–17, `notes/lair-runs.md`) và quyết P14; fine-tune chạy trên Colab. Sau đó: B16 khi có `runs/<lượt>/model.pt` (embed từng biến thể, chọn joint/decomposed trên pool-val, K1–K3 trên corpus), song song B13b (FT4/FT5) hoặc Task 21–22.

### Script B10 (bản chép của `wp6a_cxrclip.sh`)

```bash
C=20261023_chexpert25k_cxrclip_v1; P=20261023_pool88k_cxrclip_v1
uv run cxr train-probe --encoder cxrclip --corpus-build-id $C --pool-build-id $P \
    --out data/indexes/image_probe_cxrclip --report artifacts/metrics/image_probe_cxrclip.json
for g in corpus corpus5k; do
  uv run cxr eval-image-bench --gallery $g --scoring joint --encoder cxrclip --build-id $C --out artifacts/metrics/image_bench_cxrclip_joint_$g.json
  uv run cxr eval-image-bench --gallery $g --scoring decomposed --scores-dir data/indexes/image_probe_cxrclip --build-id $C --out artifacts/metrics/image_bench_probe_cxrclip_$g.json
done
uv run cxr eval-image-bench --gallery pool-val --scoring joint --encoder cxrclip --pool-build-id $P --out artifacts/metrics/image_bench_cxrclip_joint_poolval.json
```

# V1.1 — bảng trạng thái từng bước (nguồn sự thật về tiến độ)

> **Cách dùng.** Một bước một lúc. Trước khi bắt đầu bước nào: đổi trạng thái của nó thành `🔄` và ghi giờ.
> Sau khi xong: `✅` + kết quả (commit, file, số đo chính, log). Lỗi/chặn: `⛔` + lý do. Bước chưa tới: `⏳`.
> Mỗi phiên mới (hoặc chat mới) đọc theo thứ tự: `WORKLOG.md` (*Trạng thái*, *Quyết định* D30–D34, *Bẫy*) →
> **file này** → spec `docs/superpowers/specs/2026-10-23-label-aware-image-retrieval-design.md` →
> plan `docs/superpowers/plans/2026-10-23-label-aware-image-retrieval.md` (chi tiết từng task) → mục của bước đang làm.
> Chỉ **một** dòng được đánh dấu **TIẾP THEO**. File này nằm trên nhánh tích hợp; commit cùng các mốc, không commit từng lần sửa.

**Cập nhật lần cuối:** 2026-10-23 21:05 (giờ máy) — WP0, WP2, WP1 xong và đã merge; WP3/4, WP5, WP8 đang chạy ở worktree; chuỗi embed CXR-CLIP đang chạy nền.

**Nhánh tích hợp:** `feat/label-aware-image-retrieval` (HEAD `d13e1cc`), tách từ `main` 90cd62a. Không push, không đụng `main`; squash/PR khi người dùng gọi `/commit`.

## Mốc

| Mốc | Nghĩa | Trạng thái |
|---|---|---|
| M1 | Hệ thống chạy `image_joint` / `image_decomposed` với CXR-CLIP zero-shot; `validate-indexes` PASS; `vector`/`hybrid`/`image` trùng từng bit | ⏳ cần: chuỗi embed xong (B5), WP5 merge (B9), index + build-image-scores (B10) |
| M2 | Heldout một lần với mọi mode, K1–K7, WORKLOG + `docs/image-retrieval.md` | ⏳ |

## Các bước

| # | Gói | Bước | Trạng thái | Kết quả / nơi xem |
|---|---|---|---|---|
| B1 | WP0 | Spec + plan + lộ trình lưu file; H1 `templates.py`, H2 `training.yaml`; WORKLOG D30–D34 | ✅ 2026-10-23 | commit `4ee7f1b`, `c2f3334`, `6b13159`; 10 test template |
| B2 | WP2 | Adapter `cxrclip` (pin sha256 bản chuyển `743e2333…`), fidelity ≥ 0,999 so với công thức gốc | ✅ | commit `b71e587`; `uv run pytest -m model -k cxrclip` 2 passed; `notes/lair-encoder.md` |
| B3 | WP2 | `embed --encoder/--scope`, `train-probe --encoder/--corpus-build-id`, `shards.json.encoder`, `probe.json.source` | ✅ | commit `e5904e4`; 936 passed |
| B4 | WP1 | `template_bench.py` + `cxr eval-image-bench` (P@k, hit@k, HNRR@k, CI95 bootstrap, gallery corpus/corpus5k/pool-val) | ✅ merge `56798f8` + fix `5ff3221` | 957 passed; `notes/lair-bench.md` |
| B5 | WP2 | Chuỗi nền: (a) embed corpus CXR-CLIP → (b) kiểm bit-identity probe BiomedCLIP → (c) embed pool CXR-CLIP | 🔄 bắt đầu 20:43, đang ở (a) | `notes/wp2_chain.log`; shard ở `data/indexes/embeddings/20261023_chexpert25k_cxrclip_v1/`, `data/probe_pool/embeddings/20261023_pool88k_cxrclip_v1/` |
| B6 | WP6a | Dòng mốc BiomedCLIP: joint + probe-decomposed, gallery corpus và corpus5k | ✅ 20:58 | `artifacts/metrics/image_bench_biomedclip_joint_{corpus,corpus5k}.json`, `image_bench_probe_biomedclip_{corpus,corpus5k}.json`; log `notes/wp6a_biomedclip.log`; **chưa đọc số** → đọc ở B12 |
| B7 | WP3+4 | Agent `train` (worktree `.worktrees/train`, nhánh `feat/lair-train`): captions, cache, venv `training/` (torch cu124), loss, LoRA, train loop | 🔄 agent đang chạy | nhánh đã có tới `f5a3910` (Task 11 LoRA xong); còn Task 12 train loop |
| B8 | WP8 | Agent `ui` (`.worktrees/ui`, `feat/lair-ui`): Next.js, 6 mode, cards, detail, compare, Playwright | 🔄 agent đang chạy | nhánh chưa có commit mới (`6b13159`) |
| B9 | WP5+9 | Agent `serve` (`.worktrees/serve`, `feat/lair-serve`): manifest ảnh, `index --role image`, AppState, 2 mode mới, `/studies/{id}`, `build-image-scores`, `eval-retrieval` mode mới, protocol human review, compose | 🔄 agent đang chạy | nhánh có `ee0466b` (Task 13 xong); còn Task 14–17, 21–22 |
| B10 | WP6a | Sau B5: `train-probe --encoder cxrclip`; bench joint CXR-CLIP (corpus, corpus5k, pool-val); probe-CXR-CLIP decomposed | ⏳ script sẵn: `%TMPDIR%/../scratchpad/wp6a_cxrclip.sh` (xem mục *Lệnh*) | |
| B11 | tích hợp | Merge WP3+4 → WP5 → (WP8 riêng) vào nhánh tích hợp; `pytest`, `ruff`; dọn worktree | ⏳ | thứ tự plan: WP5 trước WP4; WP8 khi xong |
| B12 | WP6a/M1 | `cxr index --role image`, `build-image-scores --encoder cxrclip`, `validate-indexes` 15 PASS, `pytest -m integration`; **đọc bảng dòng mốc** (B6 + B10 + decomposed CXR-CLIP) và ghi `notes/lair-runs.md` | ⏳ | → **M1** |
| B13 | WP4 | `training/cache.py` trên pool thật (≈15 phút CPU); smoke train 5k ảnh 1 epoch (GPU, venv `training/`); bấm giờ | ⏳ | |
| B14 | WP6b | Dò lr FT1 {1e-5, 3e-5, 1e-4}, chọn trên pool-val (D33) | ⏳ qua đêm | |
| B15 | WP6b | FT2 (+impression), FT3 (λ 0) với lr đã chọn | ⏳ | |
| B16 | WP6b | Embed corpus + pool bằng từng checkpoint (GPU qua venv `training/`), `build-image-scores` từng biến thể, bench pool-val → chọn joint/decomposed (D33); bench corpus → K1–K3 | ⏳ | |
| B17 | WP7 | Trỏ phục vụ vào biến thể đã chọn; heldout **một lần** (khai báo trong WORKLOG trước khi chạy); K4–K7; AUROC `valid/` | ⏳ | → **M2** |
| B18 | WP7 | `docs/image-retrieval.md`, `docs/model-decision.md`, README, WORKLOG (gom `notes/lair-*.md`) | ⏳ | |

## Đang chạy nền (kiểm tra bằng `tail`/`grep` log, không chạy lại khi chưa xem log)

| Việc | Lệnh đang chạy | Log | Cách biết xong |
|---|---|---|---|
| B5 chuỗi embed | `uv run cxr embed --encoder cxrclip …` ×2 và `train-probe` kiểm bit-identity (xem `notes/lair-encoder.md`) | `notes/wp2_chain.log` | dòng `=== … done`; `IDENTICAL probe.npz` / `IDENTICAL corpus_scores.parquet`; `shards.json` xuất hiện trong hai thư mục build |

## Agent / worktree

| Worktree | Nhánh | Việc | Khi xong |
|---|---|---|---|
| `.worktrees/bench` | `feat/lair-bench` | WP1 — **đã merge** | có thể `git worktree remove .worktrees/bench` |
| `.worktrees/train` | `feat/lair-train` | WP3+WP4 | merge sau WP5 (B11) |
| `.worktrees/serve` | `feat/lair-serve` | WP5+WP9 | merge ngay khi xong (B11), rồi B12 |
| `.worktrees/ui` | `feat/lair-ui` | WP8 | merge riêng khi xong |

## Lệnh tiếp tục (chạy ở cây chính)

```bash
# Kiểm tra sức khỏe nhanh
uv run pytest -q && uv run ruff check src tests          # hiện 957 passed
tail -20 notes/wp2_chain.log                             # B5
git log --oneline feat/lair-train feat/lair-serve feat/lair-ui -1  # tiến độ agent
# B10 (chỉ sau khi B5 có dòng "done" và hai shards.json)
bash "$TMPDIR/../scratchpad/wp6a_cxrclip.sh"              # bản chép của script ở cuối file này
# B11 merge (theo thứ tự)
git merge --no-ff --no-edit feat/lair-serve && uv run pytest -q && uv run ruff check src tests
git merge --no-ff --no-edit feat/lair-train && uv run pytest -q
# B12 (M1)
uv run cxr index --role image --build-id 20261023_chexpert25k_cxrclip_v1
uv run cxr build-image-scores --encoder cxrclip --corpus-build-id 20261023_chexpert25k_cxrclip_v1 \
    --pool-build-id 20261023_pool88k_cxrclip_v1 --out data/indexes/image_scores --report artifacts/metrics/image_scores_cxrclip.json
uv run cxr validate-indexes && uv run pytest -m integration -q
```

**TIẾP THEO:** chờ B5 (chuỗi embed) và báo cáo của ba agent; khi B5 xong → chạy B10; khi `serve` xong → B11 rồi B12 (M1).

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

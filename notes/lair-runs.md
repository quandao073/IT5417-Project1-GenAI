# V1.1 runs on real data (WP6/WP7)

## M1: the image modes with CXR-CLIP zero-shot (2026-10-24, plan Task 23)

**Method.** Commits `4acd4c9`, `9342d59` and `6077698` (Tasks 15–17) on the integration branch. Everything ran on CPU with native settings: `QDRANT_URL=http://localhost:6333`, `INDEXES_DIR=data/indexes`, `IMAGE_*` paths under `data/indexes`. The Qdrant container had stopped (exit 143) and was restarted with `docker compose up -d qdrant`. Logs are in `notes/m1.log` and `notes/baselines.log`; the API smoke log is `notes/m1_api.log`.

| Step | Command | Result |
|---|---|---|
| Image-mode collection | `cxr index --role image --build-id 20261023_chexpert25k_cxrclip_v1 --recreate` | 25,000 vectors, `build_manifest_image.json` with `encoder: cxrclip` |
| H6 scores, served | `cxr build-image-scores --encoder cxrclip … --out data/indexes/image_scores` | 14 concepts calibrated on pool-val (9,037), none uncalibrated |
| H6 scores, baseline row | same with `--encoder biomedclip` → `data/indexes/image_scores_biomedclip` | as above |
| CXR-CLIP probe | copied from the encoder experiment to `data/indexes/image_probe_cxrclip` | pool-val macro AUROC 0.661 (λ 1) |
| `validate-indexes` | first run | **14/15**: `config_hash` FAIL on the 2026-10-11 BiomedCLIP manifest (see below) |
| Re-stamp | `cxr index --build-id 20261011_chexpert25k_biomedclip_v1 --recreate`, old manifest kept as `build_manifest.biomedclip_20261011.json.bak` | **15/15 PASS** |
| Integration tests | `pytest -m integration` | 4 passed |
| No change to the old modes | dev, parser gold: per-query metrics of `label`/`vector`/`image`/`hybrid` against the pre-Task-15 run (`exp_retrieval_dev_biomedclip.json`) and again after the re-stamp | **identical** on every shared key, 38 queries × 4 modes; `hnrr@10` is the only new key |
| Live API | `uvicorn src.api.main:app` | `/health`: `image_joint` ok, `image_decomposed` ok, encoder `cxrclip`. Both modes answer "cardiomegaly without pleural effusion"; `concept_scores` sum to `score`. `/studies/patient50920%2Fstudy3` and the raw-slash path both return 200 with 14 labels and the report. CORS preflight from `localhost:3000` returns 200 |

**Why `config_hash` failed.** D16 hashes all of `models.yaml:image_encoder`, candidates included. Adding the `cxrclip` candidate (WP2) therefore changed the hash of every older build, although nothing that shaped the BiomedCLIP build had changed. Re-indexing from the build's own shards re-stamps the manifest with identical vectors: the `sample` check compares Qdrant with the shards. Every future candidate (FT6, FT4, …) will cause the same failure again (WORKLOG T35, P14).

### Dev benchmark, parser gold, policy default (`artifacts/metrics/retrieval_dev_lair.json`)

| Mode | n | P@10 | nDCG@10 | exclude viol@10 | p95 |
|---|---|---|---|---|---|
| `label` | 38 | 1.000 | 0.524 | 0.000 | 30 ms |
| `vector` (BiomedCLIP) | 38 | 0.447 | 0.302 | 0.511 | 220 ms |
| `image` (probe-BiomedCLIP) | 38 | 0.626 | 0.367 | 0.217 | 34 ms |
| `image_joint` (CXR-CLIP zero-shot) | 38 | 0.468 | 0.309 | 0.400 | 69 ms |
| `image_decomposed` (CXR-CLIP H6) | 38 | 0.474 | 0.272 | 0.383 | 28 ms |
| `hybrid` | 38 | 1.000 | 0.657 | 0.000 | 396 ms |

Mode comparison (K4) on the 35 queries `vector` answered, as `image_vs_vector` (spec §5.1). It uses binary nDCG@10; HNRR@10 is computed on the 6 queries that have both an include and an exclude. `image_modes` in the report names the image build (`cxrclip`, `sha256:743e…`) and the H6 artifact (`20261023_chexpert25k_cxrclip_v1_decomposed`).

| | ndcg_bin (new / base) | wins / ties / losses | HNRR (new / base) |
|---|---|---|---|
| `image_joint` vs `image` | 0.457 / 0.584 | 8 / 2 / 25 | 0.133 / 0.100 |
| `image_joint` vs `vector` | 0.457 / 0.506 | 7 / 6 / 22 | 0.133 / 0.067 |
| `image_decomposed` vs `image` | 0.444 / 0.584 | 4 / 3 / 28 | 0.267 / 0.100 |
| `image_decomposed` vs `vector` | 0.444 / 0.506 | 6 / 6 / 23 | 0.267 / 0.067 |

On the 3 queries `vector` left empty, binary nDCG@10 is 0.641 for `image_joint` and 0.817 for `image_decomposed`, against 0.974 for `image`.

On natural-language dev queries, the two zero-shot image modes are below the BiomedCLIP probe. Closing that gap is the fine-tune's job (D30, FT6).

### Baseline table: template bench S12, policy default (`artifacts/metrics/m1_baseline_table.txt`)

P@5 with bootstrap CI95 (1,000 resamples); HNRR@5 and viol@10 are for the negation queries, lower is better. This is a measurement only: nothing is selected on the corpus (D33).

**corpus (25,000)**

| Row | single | pair | negation | neg HNRR@5 | neg viol@10 | paraphrase P@5 |
|---|---|---|---|---|---|---|
| BiomedCLIP joint | 0.533 [0.32, 0.73] | 0.189 [0.12, 0.26] | 0.236 [0.19, 0.29] | 0.153 | 0.373 | 0.187 |
| BiomedCLIP decomposed (H6) | 0.533 [0.32, 0.73] | 0.182 [0.12, 0.25] | 0.318 [0.26, 0.38] | 0.138 | 0.274 | 0.250 |
| probe-BiomedCLIP | 0.617 [0.45, 0.78] | 0.273 [0.19, 0.35] | 0.407 [0.35, 0.46] | 0.129 | 0.227 | 0.332 |
| **CXR-CLIP joint** (K1–K3 reference) | 0.633 [0.48, 0.78] | 0.255 [0.18, 0.34] | 0.260 [0.21, 0.32] | **0.095** | 0.252 | 0.255 |
| CXR-CLIP decomposed (H6) | 0.600 [0.47, 0.72] | 0.247 [0.19, 0.32] | 0.382 [0.33, 0.43] | 0.167 | 0.278 | 0.327 |
| **probe-CXR-CLIP** | **0.700** [0.53, 0.83] | **0.367** [0.29, 0.45] | **0.542** [0.49, 0.60] | 0.151 | 0.231 | **0.473** |

**corpus5k**

| Row | single | pair | negation | neg HNRR@5 | neg viol@10 | paraphrase P@5 |
|---|---|---|---|---|---|---|
| BiomedCLIP joint | 0.583 [0.40, 0.77] | 0.218 [0.15, 0.29] | 0.224 [0.18, 0.27] | 0.173 | 0.375 | 0.194 |
| BiomedCLIP decomposed (H6) | 0.583 [0.40, 0.77] | 0.233 [0.16, 0.32] | 0.358 [0.30, 0.42] | 0.156 | 0.278 | 0.299 |
| probe-BiomedCLIP | 0.650 [0.50, 0.80] | 0.273 [0.20, 0.35] | 0.402 [0.35, 0.46] | 0.135 | 0.223 | 0.332 |
| CXR-CLIP joint | 0.550 [0.38, 0.72] | 0.225 [0.15, 0.30] | 0.256 [0.21, 0.31] | 0.096 | 0.255 | 0.237 |
| CXR-CLIP decomposed (H6) | 0.567 [0.40, 0.73] | 0.240 [0.17, 0.32] | 0.340 [0.28, 0.39] | 0.149 | 0.282 | 0.290 |
| probe-CXR-CLIP | 0.700 [0.53, 0.85] | 0.309 [0.23, 0.40] | 0.464 [0.41, 0.53] | 0.173 | 0.228 | 0.401 |

Reading the table:
- **K1–K3 reference for the fine-tune:** the starting model is "CXR-CLIP joint". On corpus, K1 asks pair P@5 ≥ 0.305 and negation P@5 ≥ 0.310, each with the lower CI bound of the difference > 0. K2 asks single P@5 ≥ 0.583. K3 asks negation HNRR@5 ≤ 0.095.
- **The bar a fine-tune must clear to be worth it:** probe-CXR-CLIP. It uses no fine-tuning, only 14 logistic probes on frozen CXR-CLIP features, and already beats CXR-CLIP joint by +0.112 on pair and +0.282 on negation (corpus). Its HNRR (0.151) is worse than the joint row's.
- **Calibration of H6 (pool-val):**
  - The strict AUROC of No Finding is n/a: No Finding is never ABSENT.
  - The calibrated slope is negative for CXR-CLIP Pleural Effusion (a = −3.01). The "pleural effusion" caption therefore scores inversely on pool-val under the probe's targets (PRESENT against ABSENT or NOT_MENTIONED).
  - Lung Lesion, Atelectasis and Fracture stay below 0.5 strict AUROC with both encoders. The calibration targets count NOT_MENTIONED as negative, while the AUROC scores only ABSENT as negative.

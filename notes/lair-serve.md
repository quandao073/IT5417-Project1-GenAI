# WP5 (serving): the image modes, `/studies`, `build-image-scores`, `eval-retrieval`

Tasks 13–14 were built on `feat/lair-serve` (merged `ff56ddc`, branch deleted 2026-10-24). Tasks 15–17 were built directly on the integration branch: `4acd4c9`, `9342d59`, and the commit after it. The design is spec §5.1; decisions are D31 and D34.

## What the API serves now

| Mode | Text the encoder reads | Ranking | `score` |
|---|---|---|---|
| `image_joint` | `render(include, exclude)` (H1). Without a constraint: `semantic_query` plus the warning "no concept; ranked by text-image cosine on the image collection". With an empty `semantic_query`: no results | cosine on the second collection (`build_manifest_image.json`) | cosine |
| `image_decomposed` | none (no constraint → the joint path) | `rank_studies` on the H6 scores (`IMAGE_SCORES_DIR`) | sum of `score_breakdown.concept_scores` |

- **Excluding No Finding:** H1 never negates No Finding, so an exclude of it is dropped with a warning (Review Focus 1).
- **Concepts outside the 12 training concepts (H2):** the concept is still scored, and a warning names it (Review Focus 2).
- **UNCERTAIN includes:** they trigger the same warning as mode `image`.
- **Disabled modes:** a missing manifest, collection or scores leaves the API `ready` and disables only that mode. Search in it returns `[]` with the warning `image mode unavailable: <reason>`.
- **Unchanged modes:** `vector`, `hybrid` and `image` give the same hits with or without the image modes (`test_vector_hybrid_and_image_hits_do_not_depend_on_the_image_modes`).
- **New response fields:** `SearchResult.projection` (AP/PA) and `ScoreBreakdown.concept_scores` (null outside `image_decomposed`).
- **`GET /api/v1/studies/{study_id}`:** the route uses a `:path` parameter, so both `patient00001/study1` and `patient00001%2Fstudy1` work. It returns the 14 resolved labels with their sources. `report` (`impression`, `findings`, `search_text`) is read per request from the canonical `reports.parquet`; it is `null` when that file is missing. Unknown ids return 404.
- **CORS:** `CORS_ORIGINS`, default `localhost:3000` and `127.0.0.1:3000`. The web UI calls the API from the browser.

## Building what the modes need

```bash
uv run cxr index --role image --build-id 20261023_chexpert25k_cxrclip_v1     # second collection + its manifest
uv run cxr build-image-scores --encoder cxrclip --corpus-build-id 20261023_chexpert25k_cxrclip_v1 \
    --pool-build-id 20261023_pool88k_cxrclip_v1                             # -> data/indexes/image_scores
uv run cxr validate-indexes                                                 # 8 checks + 6 image:* + image_scores
```

How `build-image-scores` calibrates each concept c:
- It computes the cosine between the corpus image and `render([c], [])` (L2-normalised), then fits one-variable logistic regression on pool-val.
- Pool-val is the patient split of `models.yaml:image_probe`, the same split that chose the probe's λ.
- The result is W[c] = a·t_c and b[c] = b. `probe.json` records `source: decomposed-<encoder>`, `prompts` and `calibration` (a, b, n_pos, n_neg, strict AUROC on pool-val).
- A concept with fewer than 10 positives or negatives on pool-val keeps a = 1, b = 0, and is listed under `uncalibrated`.

## Evaluation

`cxr eval-retrieval --modes … image_joint image_decomposed` refuses to run when a mode it was asked for is disabled, and names the command that fixes it. Per query it adds `hnrr@10` (as in `template_bench`; only when the gold has both an include and an exclude). For each new mode the report gains `mode_comparison`, comparing it with `image` and `vector` on **the queries `vector` answered**, as `image_vs_vector` (spec §5.1): one set for every baseline, and a new mode that returns nothing there scores 0 and loses. It gives `ndcg_binary@10` wins / ties / losses and means, mean `hnrr@10` where both sides have one, and `vector_unanswered` reported apart. The report also gains `image_modes`: the image build, encoder, revision and collection of `image_joint`, and the `probe_id`/`source`/`build_id` of the H6 scores of `image_decomposed`, so a result names the checkpoint it measured.

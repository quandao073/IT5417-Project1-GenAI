# WP1 — template benchmark (`feat/lair-bench`)

Files: `src/evaluation/template_bench.py`, `tests/unit/test_template_bench.py`,
`tests/unit/test_cli_image_bench.py`, the `eval-image-bench` block in `src/cli/main.py`
(imports `template_bench as tb`, `load_probe`; functions `_encoder_by_name`, `_bench_gallery`,
`_cmd_eval_image_bench`, the `ib` parser block; nothing else in `main.py` touched).

## Verified (worktree, torch 2.4.1+cpu)

- Baseline on the integration head: `uv run pytest -q` 922 passed; ruff clean.
- After WP1: **945 passed**, 11 deselected (23 new: 17 bench, 6 CLI); `uv run ruff check src tests`
  clean; `uv run cxr eval-image-bench --help` renders.
- Relevance equals `label_filter.filter_studies_by_labels` cell for cell on the toy gallery (both
  policies), with the matrix built by `lf.build_label_matrix` from a long frame as the reference.
- Toy ranking [s1, s3, s2] for "pulmonary edema and no pleural effusion": p@1 1.0, p@3 2/3,
  hit@3 1.0, hnrr@3 1/3, exclude_violation@10 1/3; six-study evaluate: default 4 relevant,
  strict 2 → `min_relevant=3` skips the strict cell, `skipped_small_relevant == 1`.
- Bootstrap: same seed → identical report; every cell with n > 0 has ci95[0] <= mean <= ci95[1];
  an oracle scorer against a random one gives `paired_delta` mean > 0 with ci95[0] > 0.
- Query sets: R10 10/45/90 (145), S12 12/55/110 (177), paraphrase 135 (over the paper's ten
  labels), multi 30 triple + 30 pair_negation (over the twelve training concepts).
- Galleries from fake shards: corpus rows follow `studies.parquet`, vectors are those of
  `selected_image_id`, projections joined from `images.parquet`; the 5-row sample is seed-fixed and
  a sub-sequence of the corpus order; pool-val rows are exactly `probe.patient_split(...)` True.
  With `models.yaml:image_probe` (0.1, 20260101) 2 of the 12 fake pool patients fall on the
  validation side.
- CLI on the 20-study fake corpus (S12 + multi, FakeEncoder, n_boot 200): report merged into an
  existing JSON next to its other sections, stdout ASCII; 438 of 474 (query, policy) pairs skipped
  because the toy labels make almost every query < 5 relevant. Guards tested: encoder revision !=
  `shards.json:revision` → `ENCODER MISMATCH`, exit 1, no report; joint without `--encoder`;
  decomposed without `--scores-dir`; pool-val without `--pool-build-id`; unknown pool build;
  `corpus5k` on a corpus smaller than 5000.

## Decisions that are not spelled out in the spec or plan

- `joint_scorer(encoder, vectors)` as in spec 5.2. The plan's Task 7 adds `concepts_cfg`; it has
  no use (`q.text` is already the H1 rendering), so it was dropped. **Callers written against the
  plan must pass two arguments.**
- `query_metrics(ranked_rows, relevant, hnrr, violation, *, ks)`: one more mask than the plan's
  three positional arguments, because exclude_violation@10 (any exclude PRESENT) and HNRR (every
  include and every exclude PRESENT) are different masks. `violation_mask(statuses, query)` is new.
- `evaluate` takes `queries` as a mapping set name → queries (what `query_sets` returns), and
  `per_query` is nested by set, because template ids repeat across sets (`single-000` exists in
  both R10 and S12). Each per_query row is `{"kind", "text", "policies": {name: metrics | None}}`;
  None marks a skipped (query, policy).
- `paired_delta(a, b, *, set_name, kinds, policy, metric, n_boot, seed)`: `set_name` and
  `policy` added to the plan's signature since a report holds several of each. Delta is b − a,
  paired by query id; queries skipped or None in either report are dropped.
- Policy keys in the JSON are the names `default` / `strict` (`POLICIES`, `policy_name`), as the
  CLI flags; the tuples stay the Python-level argument.
- P@k divides by k; HNRR@k and exclude_violation@10 divide by the results returned (the Phase 5
  convention of `eval-retrieval`). The two only differ when the gallery has fewer than k rows.
- The bootstrap generator is re-seeded per (set, kind, policy), so a cell's CI does not depend on
  which other sets were evaluated (WP4 running S12 alone gets the same CIs as the full run).
- `corpus5k` rows are sorted after sampling: a sub-sequence of the corpus order, so ties break the
  same way relative to the full gallery.
- `--encoder` loads any `models.yaml` candidate via `load_encoder(name, candidate, "cpu")`, not
  the pinned `selected`; consistency is enforced instead by comparing the encoder's revision (joint)
  or `probe.json:encoder.revision` (decomposed) with `shards.json:revision` of the gallery shards.
  Consequence: `--scoring decomposed --scores-dir data/indexes/image_probe` scores the BiomedCLIP
  probe itself as a decomposed line on BiomedCLIP shards (its probe.json has `encoder.revision`).
- Report payload (`bench` section via `vb.merge_report`) has, beyond the brief: `probe`
  (`probe_id`, `source`, `build_id` of probe.json; None for joint), `shards_dir`; `build_id` is
  `shards.json:build_id` of the shards actually used (the pool build for `pool-val`).
- Not exposed as flags: `min_relevant` (5) and the seed (20260101), fixed by the roadmap.

## For other packages

- WP4 checkpoint selection: `tb.evaluate({"S12": tb.query_sets(cfg, train)["S12"]}, gallery,
  tb.joint_scorer(encoder, vectors), policies=(tb.DEFAULT_POLICY,), n_boot=<small>)`, then the
  mean of `report["sets"]["S12"][kind]["default"]["p@5"]["mean"]` over single / pair / negation.
  Build the gallery with `tb.pool_val_gallery(pool_dir, None, val_fraction=..., seed=...)` and
  pass the GPU-embedded vectors (same row order as `pool_images.parquet` filtered by the split).
- WP6: `tb.paired_delta(zero_shot_report["bench"], ft_report["bench"], set_name="S12",
  kinds=("pair",), policy="default", metric="p@5")` → K1 lower bound; `kinds=("negation",)` and
  `metric="hnrr@5"` → K3.

## Open questions

- Plan vs spec disagree on `joint_scorer`; the spec form was kept (see above). Confirm at merge.
- HNRR for `pair_negation` ("a and b and no c") requires a, b **and** c PRESENT. Intended?
- The CLI prints `encoder=None` for a decomposed run whose probe.json lacks `encoder.name` (the
  BiomedCLIP probe artifact). Harmless; WP5's `build-image-scores` writes `name` per H6.
- Real-data cost not measured here (no data in the worktree): S12 on the 25k corpus is a
  177 x 25,000 float32 score matrix (~18 MB) plus one `encode_texts` call per 64 queries.

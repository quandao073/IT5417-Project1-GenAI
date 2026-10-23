"""CLI wiring of `cxr eval-image-bench` (V1.1 WP1): defaults, the two scorings over the
corpus and pool-val galleries built from fake shards, the report it merges, its guards."""

from __future__ import annotations

import json

import numpy as np
import polars as pl
import pytest

from src.cli.main import build_parser, main
from src.data.labels import CHEXPERT_LABELS
from src.embeddings import probe
from tests.unit.fakes import FakeEncoder
from tests.unit.test_api_state import BUILD
from tests.unit.test_template_bench import POOL_BUILD, bench_world

REQUIRED = ["eval-image-bench", "--gallery", "corpus", "--scoring", "joint", "--out", "x.json"]


def test_eval_image_bench_defaults():
    a = build_parser().parse_args(REQUIRED)
    assert a.sets == ["R10", "S12", "paraphrase", "multi"]
    assert a.policies == ["default", "strict"]
    assert a.n_boot == 1000
    assert a.encoder is None and a.build_id is None and a.scores_dir is None
    assert a.pool_dir == "data/probe_pool" and a.pool_build_id is None
    assert a.embeddings == "data/indexes/embeddings" and a.canonical_dir == "data/canonical"
    assert build_parser().parse_args([*REQUIRED, "--sets", "S12", "--policies", "default"]
                                     ).sets == ["S12"]


def test_eval_image_bench_requires_gallery_scoring_and_out():
    for missing in ("--gallery", "--scoring", "--out"):
        argv = [a for i, a in enumerate(REQUIRED) if a != missing and REQUIRED[i - 1] != missing]
        with pytest.raises(SystemExit):
            build_parser().parse_args(argv)
    with pytest.raises(SystemExit):
        build_parser().parse_args([*REQUIRED, "--sets", "R99"])


def _corpus_argv(tmp_path, canonical, embeddings, **overrides):
    args = {"--gallery": "corpus", "--scoring": "joint", "--encoder": "fake", "--build-id": BUILD,
            "--embeddings": str(embeddings), "--canonical-dir": str(canonical),
            "--out": str(tmp_path / "report.json"), "--n-boot": "20"}
    args.update(overrides)
    argv = ["eval-image-bench"]
    for flag, value in args.items():
        if value is not None:
            argv += [flag, *value.split()] if flag in ("--sets", "--policies") else [flag, value]
    return argv


def test_eval_image_bench_writes_report_with_fake_shards(tmp_path, monkeypatch, capsys):
    canonical, embeddings, _ = bench_world(tmp_path)
    monkeypatch.setattr("src.cli.main._encoder_by_name", lambda name: FakeEncoder())
    out = tmp_path / "report.json"
    out.write_text('{"other": {"keep": 1}}')
    argv = _corpus_argv(tmp_path, canonical, embeddings, **{"--sets": "S12 R10",
                                                           "--policies": "default strict"})
    assert main(argv) == 0

    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["other"] == {"keep": 1} and "git_commit" in report
    bench = report["bench"]
    assert bench["scoring"] == "joint" and bench["build_id"] == BUILD
    assert bench["encoder"] == {"name": "fake", "model_id": "fake/model", "revision": "fakerev0",
                                "dim": 4}
    assert bench["scores_dir"] is None and "git_commit" in bench
    assert bench["gallery"]["name"] == "corpus" and bench["gallery"]["n"] == 20
    assert len(bench["gallery"]["study_ids_sha256"]) == 64
    assert bench["queries"] == {"S12": {"single": 12, "pair": 55, "negation": 110},
                                "R10": {"single": 10, "pair": 45, "negation": 90}}
    sets = bench["bench"]["sets"]
    assert list(sets) == ["S12", "R10"] and bench["bench"]["policies"] == ["default", "strict"]
    # "pulmonary edema" (10 relevant) and "pulmonary edema and no pleural effusion"
    # (10 default / 5 strict) survive min_relevant 5; most of the 177 do not
    assert sets["S12"]["single"]["default"]["p@5"]["n"] >= 1
    assert sets["S12"]["negation"]["strict"]["p@5"]["n"] >= 1
    assert bench["bench"]["skipped_small_relevant"] > 100
    per = bench["bench"]["per_query"]["S12"]
    edema = next(r for r in per.values() if r["text"] == "pulmonary edema")
    assert edema["policies"]["default"]["n_relevant"] == 10
    negation = next(r for r in per.values()
                    if r["text"] == "pulmonary edema and no pleural effusion")
    assert negation["policies"]["default"]["n_relevant"] == 10
    assert negation["policies"]["strict"]["n_relevant"] == 5
    text = capsys.readouterr().out
    text.encode("ascii")
    assert "S12" in text and "skipped" in text and str(out) in text


def test_eval_image_bench_refuses_an_encoder_that_did_not_embed_the_shards(tmp_path, monkeypatch,
                                                                            capsys):
    canonical, embeddings, _ = bench_world(tmp_path)

    class Other(FakeEncoder):
        revision = "otherrev"

    monkeypatch.setattr("src.cli.main._encoder_by_name", lambda name: Other())
    assert main(_corpus_argv(tmp_path, canonical, embeddings)) == 1
    assert "ENCODER MISMATCH" in capsys.readouterr().err
    assert not (tmp_path / "report.json").exists()


def test_eval_image_bench_joint_needs_an_encoder_and_corpus5k_needs_the_corpus(
        tmp_path, monkeypatch, capsys):
    canonical, embeddings, _ = bench_world(tmp_path)
    assert main(_corpus_argv(tmp_path, canonical, embeddings, **{"--encoder": None})) == 1
    assert "--encoder" in capsys.readouterr().err
    monkeypatch.setattr("src.cli.main._encoder_by_name", lambda name: FakeEncoder())
    assert main(_corpus_argv(tmp_path, canonical, embeddings, **{"--gallery": "corpus5k"})) == 1
    assert "5000" in capsys.readouterr().err
    assert not (tmp_path / "report.json").exists()


def test_eval_image_bench_decomposed_on_pool_val(tmp_path, capsys):
    _, _, pool_dir = bench_world(tmp_path)
    scores_dir = tmp_path / "scores"
    rng = np.random.default_rng(0)
    meta = {"probe_id": "d1", "build_id": BUILD, "source": "decomposed-zeroshot",
            "encoder": {"name": "fake", "model_id": "fake/model", "revision": "fakerev0", "dim": 4}}
    empty = pl.DataFrame({"study_id": [], "image_id": [],
                          **{c: pl.Series([], dtype=pl.Float32) for c in CHEXPERT_LABELS}})
    probe.save_probe(scores_dir, probe.Probe(rng.normal(size=(14, 4)).astype(np.float32),
                                             np.zeros(14, np.float32), tuple(CHEXPERT_LABELS)),
                     meta, empty)
    out = tmp_path / "bench.json"
    argv = ["eval-image-bench", "--gallery", "pool-val", "--scoring", "decomposed",
            "--scores-dir", str(scores_dir), "--pool-dir", str(pool_dir),
            "--pool-build-id", POOL_BUILD, "--sets", "R10", "--policies", "default",
            "--n-boot", "10", "--out", str(out)]
    assert main(argv) == 0
    bench = json.loads(out.read_text(encoding="utf-8"))["bench"]
    assert bench["scoring"] == "decomposed" and bench["scores_dir"] == str(scores_dir)
    assert bench["encoder"] == meta["encoder"] and bench["probe"]["source"] == "decomposed-zeroshot"
    assert bench["build_id"] == POOL_BUILD
    # models.yaml:image_probe puts 2 of the 12 fake patients on the validation side
    assert bench["gallery"]["name"] == "pool-val" and bench["gallery"]["n"] == 2
    assert bench["queries"] == {"R10": {"single": 10, "pair": 45, "negation": 90}}
    assert bench["bench"]["skipped_small_relevant"] == 145      # 2 studies < min_relevant 5
    capsys.readouterr().out.encode("ascii")

    assert main([*argv[:-2], "--out", str(out), "--pool-build-id", "nope"]) == 1
    assert "nope" in capsys.readouterr().err
    assert main([a for a in argv if a not in ("--scores-dir", str(scores_dir))]) == 1
    assert "--scores-dir" in capsys.readouterr().err
    assert main([a for a in argv if a not in ("--pool-build-id", POOL_BUILD)]) == 1
    assert "--pool-build-id" in capsys.readouterr().err

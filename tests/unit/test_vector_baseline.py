"""Latency bench and full-corpus zero-shot scoring, on fakes."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from qdrant_client import QdrantClient

from src.common.config import load_config
from src.data.labels import CHEXPERT_LABELS
from src.embeddings import embed, index
from src.evaluation import vector_baseline as vb
from tests.unit.fakes import FakeEncoder
from tests.unit.test_embed import images_frame

BUILD = "20261011_chexpert25k_fake_v1"


def test_bench_queries_are_30_english_prompts():
    template = load_config("models")["image_encoder"]["text_prompt_template"]
    queries = vb.bench_queries(load_config("concepts"), template)
    assert len(queries) == 30 and len(set(queries)) == 30
    assert queries[0] == "chest x-ray showing no finding"
    assert " and " in queries[14]


def test_percentiles():
    p = vb.percentiles([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    assert p["p50"] == pytest.approx(5.5) and p["p95"] == pytest.approx(9.55) and p["n"] == 10


@pytest.fixture
def indexed(tmp_path):
    # 40 images over 20 studies: every concept gets ~20 positives and ~20 strict
    # negatives in the label fixture below, so all 14 are eligible (MIN_SAMPLES=10).
    images = images_frame(40)
    out = tmp_path / BUILD
    embed.run_embed(images, FakeEncoder(), Path("root"), out, BUILD, shard_size=15, batch_size=8,
                    library_versions={}, git_commit=None)
    client = QdrantClient(":memory:")
    manifest = index.run_index(client, out, images, "cxr_images", recreate=False,
                               config_hash="c", git_commit=None, dataset={}, library_versions={})
    return FakeEncoder(), client, manifest["collection"], out, images


def test_bench_warm_drops_the_first_repeat_and_reports_phases(indexed):
    encoder, client, collection, _, _ = indexed
    out = vb.bench_warm(encoder, client, collection, ["a", "b"], repeats=3)
    assert out["queries"] == 2 and out["repeats"] == 3
    assert out["total_ms"]["n"] == 4  # 2 queries x (3 - 1 warm repeats)
    assert set(out) >= {"encode_ms", "qdrant_ms", "total_ms"}
    assert out["total_ms"]["p50"] >= out["encode_ms"]["p50"]


def test_score_corpus_uses_every_shard_and_the_resolved_labels(indexed):
    encoder, _, _, out, images = indexed
    rows = []
    for i, study in enumerate(images["study_id"].unique(maintain_order=True)):
        for concept in CHEXPERT_LABELS:
            status = "PRESENT" if (i + len(concept)) % 2 else "ABSENT"
            rows.append((study, concept, status, "CHEXPERT_V1", "v"))
    labels = pl.DataFrame(rows, schema=["study_id", "concept", "status", "source",
                                        "source_version"], orient="row")
    prompts = [f"x {c}" for c in CHEXPERT_LABELS]
    report = vb.score_corpus(out, encoder, labels, images, prompts, ["CHEXPERT_V1"], n_boot=5)
    assert report["n_images"] == 40 and report["n_studies"] == 20
    assert len(report["concepts"]) == 14
    assert report["eligible_concepts"] == list(CHEXPERT_LABELS)
    assert report["macro_auroc_strict"] is not None and len(report["ci95"]) == 2
    assert all(0 <= v["prevalence"] <= 1 for v in report["concepts"].values())
    edema = report["label_counts"]["Edema"]
    assert edema["pos"] + edema["neg_strict"] == 40


def test_score_corpus_without_eligible_concepts_has_no_ci(indexed):
    encoder, _, _, out, images = indexed
    labels = pl.DataFrame({"study_id": [], "concept": [], "status": [], "source": [],
                           "source_version": []}, schema={"study_id": pl.Utf8, "concept": pl.Utf8,
                           "status": pl.Utf8, "source": pl.Utf8, "source_version": pl.Utf8})
    report = vb.score_corpus(out, encoder, labels, images, ["x"] * 14, ["CHEXPERT_V1"], n_boot=5)
    assert report["eligible_concepts"] == [] and report["macro_auroc_strict"] is None
    assert report["ci95"] is None


class SignedEncoder(FakeEncoder):
    """Image vectors carry the label rule, so the right alignment gives AUROC 1.0.

    Component j of an image is +1 when its study is PRESENT for concept j and -1
    when ABSENT; the prompt for concept j is the unit vector e_j.
    """

    def __init__(self):
        super().__init__(dim=len(CHEXPERT_LABELS))

    def encode_images(self, paths):
        rows = []
        for p in paths:
            study = int(re.search(r"patient(\d{5})", Path(p).as_posix()).group(1))
            rows.append([1.0 if _present(study, c) else -1.0 for c in CHEXPERT_LABELS])
        return np.asarray(rows, dtype=np.float32) / np.sqrt(len(CHEXPERT_LABELS))

    def encode_texts(self, texts):
        return np.eye(len(CHEXPERT_LABELS), dtype=np.float32)


def _present(study: int, concept: str) -> bool:
    return (study + len(concept)) % 2 == 1


def test_score_corpus_aligns_scores_with_labels_on_shuffled_images(tmp_path):
    images = images_frame(40).sample(fraction=1.0, shuffle=True, seed=3)
    assert images["image_id"].to_list() != sorted(images["image_id"].to_list())
    encoder = SignedEncoder()
    out = tmp_path / BUILD
    embed.run_embed(images, encoder, Path("root"), out, BUILD, shard_size=15, batch_size=8,
                    library_versions={}, git_commit=None)
    rows = [(study, c, "PRESENT" if _present(int(study[7:12]), c) else "ABSENT",
             "CHEXPERT_V1", "v")
            for study in images["study_id"].unique() for c in CHEXPERT_LABELS]
    labels = pl.DataFrame(rows, schema=["study_id", "concept", "status", "source",
                                        "source_version"], orient="row")
    prompts = [f"x {c}" for c in CHEXPERT_LABELS]
    report = vb.score_corpus(out, encoder, labels, images, prompts, ["CHEXPERT_V1"], n_boot=5)
    assert report["eligible_concepts"] == list(CHEXPERT_LABELS)
    for concept, metrics in report["concepts"].items():
        assert metrics["auroc_strict"] == 1.0, concept
    assert report["macro_auroc_strict"] == 1.0


def test_merge_report_creates_then_merges_sections(tmp_path):
    path = tmp_path / "metrics" / "vector_baseline.json"
    vb.merge_report(path, "latency", {"a": 1}, git_commit="c1", library_versions={"x": "1"})
    assert json.loads(path.read_text(encoding="utf-8"))["latency"] == {"a": 1}
    vb.merge_report(path, "quality", {"b": 2}, git_commit="c2", library_versions={"x": "2"})
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["latency"] == {"a": 1} and report["quality"] == {"b": 2}
    assert report["git_commit"] == "c2" and report["library_versions"] == {"x": "2"}
    vb.merge_report(path, "latency", {"a": 9}, git_commit="c3", library_versions={})
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["latency"] == {"a": 9} and report["quality"] == {"b": 2}
    assert not path.with_suffix(".json.part").exists()


def test_merge_report_refuses_a_corrupt_file_and_leaves_it_alone(tmp_path):
    path = tmp_path / "vector_baseline.json"
    path.write_text("{ not json", encoding="utf-8")
    with pytest.raises(ValueError, match="is not valid JSON; move it aside and re-run"):
        vb.merge_report(path, "latency", {}, git_commit=None, library_versions={})
    with pytest.raises(ValueError, match="is not valid JSON"):
        vb.check_report_writable(path)
    assert path.read_text(encoding="utf-8") == "{ not json"
    vb.check_report_writable(tmp_path / "missing.json")  # a missing file is fine

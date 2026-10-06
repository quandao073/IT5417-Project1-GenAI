"""Phase 2 spike: encoder helpers, sample selection, label matrix, runner."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from PIL import Image

from src.common.config import load_config
from src.embeddings import base, spike


class TestHelpers:
    def test_l2_normalize_gives_unit_rows(self):
        out = base.l2_normalize(np.array([[3.0, 4.0], [0.0, 2.0]]))
        assert out.dtype == np.float32
        assert np.allclose(np.linalg.norm(out, axis=1), 1.0)

    def test_l2_normalize_rejects_a_zero_row(self):
        with pytest.raises(ValueError, match="zero"):
            base.l2_normalize(np.array([[0.0, 0.0]]))

    def test_load_rgb_converts_grayscale(self, tmp_path):
        path = tmp_path / "g.jpg"
        Image.new("L", (8, 6), 128).save(path)
        assert base.load_rgb(path).mode == "RGB"

    def test_unknown_encoder_name_is_rejected(self):
        with pytest.raises(ValueError, match="nope"):
            base.load_encoder("nope", {})


class TestModelsConfig:
    def test_candidates_are_the_three_spike_models(self):
        cfg = load_config("models")["image_encoder"]
        assert set(cfg["candidates"]) == {"biomedclip", "chexzero", "xrayclip"}
        for c in cfg["candidates"].values():
            assert {"model_id", "framework", "revision", "license"} <= set(c)

    def test_embedding_dim_is_never_hardcoded(self):
        """D6: the encoder reports its dimension; config must not."""
        assert load_config("models")["image_encoder"]["embedding_dim"] is None

    def test_selected_pin_matches_its_candidate(self):
        cfg = load_config("models")["image_encoder"]
        if cfg["selected"] is None:
            pytest.skip("no model selected yet")
        assert cfg["selected"] in cfg["candidates"]
        assert cfg["revision"] == cfg["candidates"][cfg["selected"]]["revision"]

    def test_prompt_template_has_one_slot(self):
        template = load_config("models")["image_encoder"]["text_prompt_template"]
        assert template.count("{}") == 1


def write_canonical(root, splits):
    """Minimal canonical tables: study i has rank i and split splits[i]."""
    n = len(splits)
    ids = [f"patient{i:05d}/study1" for i in range(n)]
    pl.DataFrame({
        "study_id": ids, "stratum": ["x"] * n,
        "rank_in_manifest": list(range(n)), "selection_seed": [1] * n,
    }).write_parquet(root / "sample_manifest.parquet")
    pl.DataFrame({
        "study_id": ids, "selected_image_id": [f"img{i}" for i in range(n)],
    }).write_parquet(root / "studies.parquet")
    pl.DataFrame({
        "image_id": [f"img{i}" for i in reversed(range(n))],
        "local_image_path": [f"data/{i}.jpg" for i in reversed(range(n))],
        "dataset_split": list(reversed(splits)),
    }).write_parquet(root / "images.parquet")
    return ids


class TestSelectSample:
    def test_takes_the_rank_window_in_rank_order(self, tmp_path):
        ids = write_canonical(tmp_path, ["valid", "valid", "train", "train", "train", "train"])
        out = spike.select_sample(tmp_path, offset=2, n=3)
        assert out["study_id"].to_list() == ids[2:5]
        assert out["image_id"].to_list() == ["img2", "img3", "img4"]
        assert out["local_image_path"].to_list() == ["data/2.jpg", "data/3.jpg", "data/4.jpg"]

    def test_refuses_radiologist_labelled_studies(self, tmp_path):
        """They are the test set's only human ground truth; picking a model on
        them would be tuning on test (configs/evaluation.yaml)."""
        write_canonical(tmp_path, ["valid", "valid", "train", "train"])
        with pytest.raises(ValueError, match="valid"):
            spike.select_sample(tmp_path, offset=1, n=2)

    def test_refuses_a_short_window(self, tmp_path):
        write_canonical(tmp_path, ["train"] * 4)
        with pytest.raises(ValueError, match="3"):
            spike.select_sample(tmp_path, offset=2, n=3)


class TestStatusMatrix:
    def test_resolves_by_priority_and_fills_silence(self):
        long = pl.DataFrame({
            "study_id": ["s1", "s1", "s2"],
            "concept": ["Edema", "Edema", "Edema"],
            "status": ["NOT_MENTIONED", "ABSENT", "PRESENT"],
            "source": ["CHEXPERT_V1", "CHEXBERT_IMPRESSION", "CHEXPERT_V1"],
            "source_version": ["v"] * 3,
        })
        m = spike.status_matrix(
            long, ["s2", "s1", "s3"], ["CHEXPERT_V1", "CHEXBERT_IMPRESSION"],
            concepts=["Edema", "Fracture"],
        )
        assert m.tolist() == [
            ["PRESENT", "NOT_MENTIONED"],
            ["ABSENT", "NOT_MENTIONED"],
            ["NOT_MENTIONED", "NOT_MENTIONED"],
        ]


class TestBuildPrompts:
    def test_uses_the_first_english_form_in_label_order(self):
        cfg = {"concepts": {"Edema": {"en": ["pulmonary edema", "edema"]},
                            "Fracture": {"en": ["fracture"]}}}
        assert spike.build_prompts(cfg, "x-ray of {}", ["Fracture", "Edema"]) == [
            "x-ray of fracture", "x-ray of pulmonary edema",
        ]

    def test_real_config_gives_fourteen_prompts(self):
        prompts = spike.build_prompts(
            load_config("concepts"), load_config("models")["image_encoder"]["text_prompt_template"]
        )
        assert len(prompts) == 14 and all(p.startswith("chest x-ray showing ") for p in prompts)


class FakeEncoder:
    """Image vectors encode their own labels, so AUROC is ~1 by construction."""

    name = "fake"
    model_id = "fake/model"
    revision = "abc123"
    license = "mit"
    preprocessing = {"size": 1}
    weights_bytes = 10
    device = "cpu"

    def __init__(self, vectors: dict[str, np.ndarray], broken: frozenset[str] = frozenset()):
        self._vectors = vectors
        self._broken = broken

    @property
    def dim(self):
        return next(iter(self._vectors.values())).shape[0]

    def encode_images(self, paths):
        if any(str(p) in self._broken for p in paths):
            raise OSError("truncated jpeg")
        return base.l2_normalize(np.stack([self._vectors[str(p)] for p in paths]))

    def encode_texts(self, texts):
        return np.eye(self.dim, dtype=np.float32)[: len(texts)]


CONCEPTS = ["Edema", "Fracture"]
PROMPTS = ["chest x-ray showing edema", "chest x-ray showing fracture"]


def _fixture(n=40):
    rng = np.random.default_rng(3)
    statuses = np.empty((n, 2), dtype=object)
    statuses[:, 0] = ["PRESENT"] * 20 + ["ABSENT"] * 20
    statuses[:, 1] = ["PRESENT", "ABSENT"] * (n // 2)
    paths = [f"img{i}.jpg" for i in range(n)]
    # The constant third component keeps "absent everywhere" from normalizing
    # onto the same direction as "present everywhere".
    vectors = {
        p: np.append((statuses[i] == "PRESENT").astype(float) + rng.uniform(0, 0.05, 2), 1.0)
        for i, p in enumerate(paths)
    }
    return statuses, paths, vectors


CANDIDATES = {"fake": {"model_id": "fake/model", "revision": "abc123", "license": "mit"}}


class TestRunner:
    def test_embeds_caches_and_scores(self, tmp_path):
        statuses, paths, vectors = _fixture()
        meta = spike.embed_candidate(
            FakeEncoder(vectors), paths, [Path(p) for p in paths], PROMPTS,
            batch_size=16, cache_dir=tmp_path, load_seconds=1.5,
        )
        assert meta["dim_image"] == meta["dim_text"] == 3
        assert meta["errors"] == []
        assert (tmp_path / "fake.npz").is_file()

        report = spike.score_candidates(
            tmp_path, CANDIDATES, ["fake"], paths, PROMPTS, statuses, CONCEPTS, n_boot=20
        )
        model = report["models"]["fake"]
        assert model["macro_auroc_strict"] > 0.9
        assert model["concepts"]["Edema"]["auroc_strict"] == 1.0
        assert report["eligible_concepts"] == CONCEPTS
        assert report["gate"]["winner"] == "fake"
        assert report["bootstrap"]["seed"] == 20260101
        assert model["forecast_25k_hours"] == pytest.approx(
            meta["costs"]["s_per_image"] * spike.CORPUS_SIZE / 3600
        )
        assert model["vectors_25k_bytes"] == spike.CORPUS_SIZE * 3 * 4

    def test_a_broken_image_is_recorded_and_fails_the_gate(self, tmp_path):
        statuses, paths, vectors = _fixture()
        meta = spike.embed_candidate(
            FakeEncoder(vectors, broken=frozenset({"img5.jpg"})), paths,
            [Path(p) for p in paths], PROMPTS, batch_size=16, cache_dir=tmp_path,
            load_seconds=0.0,
        )
        assert [e["path"] for e in meta["errors"]] == ["img5.jpg"]
        assert not (tmp_path / "fake.npz").exists()

        report = spike.score_candidates(
            tmp_path, CANDIDATES, ["fake"], paths, PROMPTS, statuses, CONCEPTS, n_boot=5
        )
        assert report["models"]["fake"]["macro_auroc_strict"] is None
        assert report["gate"]["winner"] is None

    @pytest.mark.parametrize(
        "field, value",
        [("revision", "other"), ("model_id", "other/model")],
    )
    def test_a_cache_from_another_revision_is_refused(self, tmp_path, field, value):
        statuses, paths, vectors = _fixture()
        spike.embed_candidate(
            FakeEncoder(vectors), paths, [Path(p) for p in paths], PROMPTS,
            batch_size=16, cache_dir=tmp_path, load_seconds=0.0,
        )
        stale = {"fake": CANDIDATES["fake"] | {field: value}}
        with pytest.raises(ValueError, match="stale"):
            spike.score_candidates(tmp_path, stale, ["fake"], paths, PROMPTS, statuses, CONCEPTS)

    def test_mismatched_prompts_or_statuses_are_refused(self, tmp_path):
        statuses, paths, vectors = _fixture()
        spike.embed_candidate(
            FakeEncoder(vectors), paths, [Path(p) for p in paths], PROMPTS,
            batch_size=16, cache_dir=tmp_path, load_seconds=0.0,
        )
        with pytest.raises(ValueError, match="prompts"):
            spike.score_candidates(
                tmp_path, CANDIDATES, ["fake"], paths, PROMPTS[:-1], statuses, CONCEPTS
            )
        with pytest.raises(ValueError, match="statuses"):
            spike.score_candidates(
                tmp_path, CANDIDATES, ["fake"], paths, PROMPTS, statuses[:-1], CONCEPTS
            )

    def test_a_cache_for_other_images_or_prompts_is_refused(self, tmp_path):
        statuses, paths, vectors = _fixture()
        spike.embed_candidate(
            FakeEncoder(vectors), paths, [Path(p) for p in paths], PROMPTS,
            batch_size=16, cache_dir=tmp_path, load_seconds=0.0,
        )
        with pytest.raises(ValueError, match="stale"):
            spike.score_candidates(
                tmp_path, CANDIDATES, ["fake"], paths[::-1], PROMPTS, statuses, CONCEPTS
            )
        with pytest.raises(ValueError, match="stale"):
            spike.score_candidates(
                tmp_path, CANDIDATES, ["fake"], paths, PROMPTS[::-1], statuses, CONCEPTS
            )

    def test_cache_metadata_is_plain_json(self, tmp_path):
        statuses, paths, vectors = _fixture()
        spike.embed_candidate(
            FakeEncoder(vectors), paths, [Path(p) for p in paths], PROMPTS,
            batch_size=16, cache_dir=tmp_path, load_seconds=0.0,
        )
        meta = json.loads((tmp_path / "fake.json").read_text(encoding="utf-8"))
        assert meta["image_ids"] == paths and meta["prompts"] == PROMPTS
        assert {"s_per_image", "text_ms_per_query", "peak_rss_mb", "weights_bytes",
                "load_seconds", "torch_threads"} <= set(meta["costs"])

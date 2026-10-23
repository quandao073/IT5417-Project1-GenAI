import json

import numpy as np
import polars as pl
import pytest

from src.data.labels import CHEXPERT_LABELS
from src.embeddings import probe as pr
from src.query.schema import Constraint
from src.retrieval import image_concepts as ic

IDS = ["s1", "s2", "s3"]


def _frame(ids, **columns):
    """A corpus_scores frame with every concept column; unnamed concepts are zero."""
    data = {"study_id": list(ids), "image_id": [f"i-{s}" for s in ids]}
    for concept in CHEXPERT_LABELS:
        data[concept] = pl.Series(columns.get(concept, [0.0] * len(ids)), dtype=pl.Float32)
    return pl.DataFrame(data)


def _scores(**columns):
    return ic.build_image_scores(_frame(IDS, **columns), IDS)


def _present(concept):
    return Constraint(concept=concept, status="PRESENT")


def test_build_image_scores_places_rows_by_study_id():
    frame = _frame(["s3", "s1", "s2"], Edema=[3.0, 1.0, 2.0])
    scores = ic.build_image_scores(frame, IDS)
    assert scores.study_ids.tolist() == IDS
    assert scores.row_of == {"s1": 0, "s2": 1, "s3": 2}
    assert scores.logits.dtype == np.float32
    assert scores.logits.shape == (3, len(CHEXPERT_LABELS))
    assert scores.logits[:, scores.col_of["Edema"]].tolist() == [1.0, 2.0, 3.0]


def test_build_image_scores_rejects_missing_and_extra_studies():
    with pytest.raises(ic.ImageProbeError, match="1 corpus studies have no image score"):
        ic.build_image_scores(_frame(["s1", "s2"]), IDS)
    with pytest.raises(ic.ImageProbeError, match="2 scored studies are not in the corpus"):
        ic.build_image_scores(_frame(["s1", "s2", "s3", "x", "y"]), IDS)
    with pytest.raises(ic.ImageProbeError, match="corpus_scores is missing column 'Edema'"):
        ic.build_image_scores(_frame(IDS).drop("Edema"), IDS)


def test_include_raises_score_with_logit_and_exclude_lowers_it():
    scores = _scores(Edema=[-2.0, 0.0, 2.0])
    inc = ic.image_score(scores, [_present("Edema")], [])
    assert inc[0] < inc[1] < inc[2]
    exc = ic.image_score(scores, [], ["Edema"])
    assert exc[0] > exc[1] > exc[2]
    both = ic.image_score(scores, [_present("Edema")], ["Edema"])
    np.testing.assert_allclose(both, inc + exc)
    assert inc.dtype == np.float64 and inc.shape == (3,)


def test_uncertain_include_is_scored_like_present():
    scores = _scores(Edema=[-1.0, 0.5, 2.0])
    present = ic.image_score(scores, [Constraint(concept="Edema", status="PRESENT")], [])
    uncertain = ic.image_score(scores, [Constraint(concept="Edema", status="UNCERTAIN")], [])
    np.testing.assert_array_equal(present, uncertain)


def test_image_score_is_finite_at_extreme_logits(recwarn):
    extreme = _scores(Edema=[-50.0, 0.0, 50.0])
    mild = _scores(Edema=[-5.0, 0.0, 5.0])
    for include, exclude in (([_present("Edema")], []), ([], ["Edema"])):
        big = ic.image_score(extreme, include, exclude)
        small = ic.image_score(mild, include, exclude)
        assert np.isfinite(big).all()
        assert np.argsort(big).tolist() == np.argsort(small).tolist()
    assert not recwarn.list


def test_image_score_needs_a_constraint():
    with pytest.raises(ValueError, match="at least one constraint"):
        ic.image_score(_scores(), [], [])


def test_rank_studies_orders_and_restricts():
    scores = _scores(Edema=[1.0, 3.0, 3.0])
    include = [_present("Edema")]
    whole = ic.rank_studies(scores, include, [], allowed=None, limit=10)
    assert [s for s, _ in whole] == ["s2", "s3", "s1"]  # tie on s2/s3 goes to study_id
    assert all(type(v) is float for _, v in whole)
    assert whole[0][1] == whole[1][1] > whole[2][1]
    assert [s for s, _ in ic.rank_studies(scores, include, [], allowed=["s1", "s3"], limit=10)] == [
        "s3", "s1"]
    assert ic.rank_studies(scores, include, [], allowed=[], limit=10) == []
    assert len(ic.rank_studies(scores, include, [], allowed=None, limit=2)) == 2
    with pytest.raises(KeyError):
        ic.rank_studies(scores, include, [], allowed=["nope"], limit=10)


def _write_artifact(tmp_path, **meta_overrides):
    meta = {"probe_id": "p1", "build_id": "b1",
            "encoder": {"model_id": "m", "revision": "r1", "dim": 4}}
    meta.update(meta_overrides)
    path = tmp_path / "image_probe"
    probe = pr.Probe(
        W=np.zeros((len(CHEXPERT_LABELS), 4), dtype=np.float32),
        b=np.zeros(len(CHEXPERT_LABELS), dtype=np.float32),
        concepts=tuple(CHEXPERT_LABELS),
    )
    pr.save_probe(path, probe, meta, _frame(["s2", "s3", "s1"], Edema=[2.0, 3.0, 1.0]))
    return path


def test_open_image_probe_checks_build_id_and_revision(tmp_path):
    path = _write_artifact(tmp_path)
    scores, meta = ic.open_image_probe(path, build_id="b1", revision="r1", study_ids=IDS)
    assert meta["probe_id"] == "p1"
    assert scores.logits[:, scores.col_of["Edema"]].tolist() == [1.0, 2.0, 3.0]

    with pytest.raises(ic.ImageProbeError, match="trained for build b1, manifest says b2"):
        ic.open_image_probe(path, build_id="b2", revision="r1", study_ids=IDS)
    with pytest.raises(ic.ImageProbeError, match=r"r1.*r2"):
        ic.open_image_probe(path, build_id="b1", revision="r2", study_ids=IDS)
    with pytest.raises(ic.ImageProbeError, match="1 corpus studies have no image score"):
        ic.open_image_probe(path, build_id="b1", revision="r1", study_ids=[*IDS, "s4"])

    (path / pr.SCORES_FILE).unlink()
    with pytest.raises(ic.ImageProbeError, match="not found; run cxr train-probe"):
        ic.open_image_probe(path, build_id="b1", revision="r1", study_ids=IDS)


def test_validate_image_probe_returns_a_check_not_an_exception(tmp_path):
    missing = ic.validate_image_probe(
        tmp_path / "image_probe", build_id="b1", revision="r1", study_ids=IDS)
    assert missing.name == "image_probe" and not missing.ok
    assert "image_probe" in missing.detail

    path = _write_artifact(tmp_path)
    good = ic.validate_image_probe(path, build_id="b1", revision="r1", study_ids=IDS)
    assert good.ok and good.detail == "3 studies scored, build b1"
    bad = ic.validate_image_probe(path, build_id="zzz", revision="r1", study_ids=IDS)
    assert not bad.ok and bad.detail.startswith("ImageProbeError: ")


@pytest.mark.parametrize("name", [pr.PROBE_FILE, pr.SCORES_FILE, pr.META_FILE])
def test_an_unreadable_artifact_file_is_an_image_probe_error(tmp_path, name):
    path = _write_artifact(tmp_path)
    (path / name).write_bytes(b"not a valid file")
    with pytest.raises(ic.ImageProbeError, match=name):
        ic.open_image_probe(path, build_id="b1", revision="r1", study_ids=IDS)
    check = ic.validate_image_probe(path, build_id="b1", revision="r1", study_ids=IDS)
    assert not check.ok and name in check.detail


def test_meta_without_build_id_or_revision_is_an_image_probe_error(tmp_path):
    path = _write_artifact(tmp_path)
    meta_path = path / pr.META_FILE
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    del meta["build_id"]
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ic.ImageProbeError, match="build_id"):
        ic.open_image_probe(path, build_id="b1", revision="r1", study_ids=IDS)
    meta_path.write_text(json.dumps({**meta, "build_id": "b1", "encoder": {}}), encoding="utf-8")
    with pytest.raises(ic.ImageProbeError, match="revision"):
        ic.open_image_probe(path, build_id="b1", revision="r1", study_ids=IDS)

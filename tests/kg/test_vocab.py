"""`cxr kg-pool-vocab` (TIP V2-2) on invented reports, RadGraph records and MRCONSO names."""

from __future__ import annotations

import json

import polars as pl
import pytest

from src.cli.main import main
from src.common.config import load_config
from src.kg import vocab as vb

CFG = load_config("kg")["vocab"]
NAMES = {"loculated effusion", "lower lobe", "airspace consolidation", "lung", "effusion",
         "opacity", "ga", "mas"}


def names(cands):
    return set(cands) & NAMES


def entity(tokens, start, label="Observation::definitely present", modifies=None):
    return {"tokens": tokens, "label": label, "start_ix": start, "end_ix": start,
            "relations": [["modify", modifies]] if modifies else []}


def record(text, entities):
    return {"0": {"text": text, "entities": {str(i): e for i, e in enumerate(entities)},
                  "data_source": "test", "data_split": "inference"}}


ANAT = "Anatomy::definitely present"
IMPRESSIONS = {
    "train/patient00001/study1/view1_frontal.jpg": (
        "Small loculated effusions. Left lower lobe clear.",
        [entity("Small", 0, modifies="2"), entity("loculated", 1, modifies="2"),
         entity("effusions", 2), entity("Left", 3, ANAT, modifies="5"),
         entity("lower", 4, ANAT, modifies="5"), entity("lobe", 5, ANAT)]),
    "train/patient00002/study1/view1_frontal.jpg": (
        "Stable patchy airspace consolidation, lower opacity.",
        [entity("Stable", 0), entity("patchy", 1, modifies="3"),
         entity("airspace", 2, modifies="3"), entity("consolidation", 3),
         entity("lower", 4, modifies="5"),
         entity("opacity", 5, "Observation::definitely absent")]),
    # Same impression text as patient00002: both studies get its mentions.
    "train/patient00003/study1/view1_frontal.jpg": None,
}
FINDINGS = {"train/patient00001/study1/view1_frontal.jpg": (
    "Lungs and gas.", [entity("Lungs", 0, ANAT), entity("gas", 2, "Observation::uncertain")])}


@pytest.fixture
def world(tmp_path):
    paths = list(IMPRESSIONS)
    pool = pl.DataFrame({"image_id": ["i1", "i2", "i3"], "study_id": ["s1", "s2", "s3"],
                         "patient_id": ["p1", "p2", "p3"], "path_to_image": paths,
                         "projection": ["AP"] * 3})
    impressions = {p: (IMPRESSIONS[p] or IMPRESSIONS[paths[1]])[0] for p in paths}
    plus = pl.DataFrame({
        "path_to_image": paths + ["train/patient09999/study1/view1_frontal.jpg"],
        "section_impression": [impressions[p] for p in paths] + ["Not in the pool."],
        "section_findings": [FINDINGS.get(p, ("",))[0] for p in paths] + [""]})
    csv = tmp_path / "plus.csv"
    plus.write_csv(csv)
    rg = tmp_path / "radgraph"
    rg.mkdir()
    # RadGraph text is tokenised differently and its records are in reverse order (T29).
    imp = [record(t.replace(".", " .").replace(",", " ,"), ents)
           for t, ents in (IMPRESSIONS[p] for p in paths[:2])][::-1]
    fnd = [record(FINDINGS[paths[0]][0].replace(".", " ."), FINDINGS[paths[0]][1])]
    (rg / "section_impression.json").write_text(json.dumps(imp), encoding="utf-8")
    (rg / "section_findings.json").write_text(json.dumps(fnd), encoding="utf-8")
    return pool, csv, rg


def mentions_of(world):
    pool, csv, rg = world
    found, stats = vb.read_mentions(vb.pool_sections(pool, csv), rg)
    return vb.apply_rules(found, CFG, names), stats


def test_singular_follows_the_approved_rule():
    known = {"lung", "opacity", "effusion", "ga", "mas", "atelectasi"}
    assert [vb.singular(w, known) for w in ("gas", "atelectasis", "lungs", "opacities",
                                            "effusions", "mass", "status", "foos")] == [
        "gas", "atelectasis", "lung", "opacity", "effusion", "mass", "status", "foos"]


def test_drop_sets_keep_position_words_for_anatomy():
    drops = vb.drop_sets(CFG)
    assert {"small", "stable", "lower", "left"} <= drops["Observation"]
    assert "left" in drops["Anatomy"] and "lower" not in drops["Anatomy"]


def test_mentions_join_by_text_not_by_position_and_reach_every_study(world):
    found, stats = mentions_of(world)
    assert stats["impression"] == {"texts": 3, "matched": 3}
    assert stats["findings"] == {"texts": 1, "matched": 1}
    airspace = found.filter(pl.col("joined") == "airspace consolidation")
    assert set(airspace["study_id"]) == {"s2", "s3"}


def test_observation_drops_its_lists_and_keeps_a_modifier_only_when_mrconso_has_the_phrase(world):
    found, _ = mentions_of(world)
    s1 = found.filter(pl.col("study_id") == "s1").sort("kind", "joined")
    observation = s1.filter(pl.col("kind") == "Observation")["joined"].to_list()
    assert observation == ["gas", "loculated effusion"]
    assert s1.filter(pl.col("kind") == "Anatomy")["joined"].to_list() == ["lower lobe", "lung"]
    s2 = found.filter(pl.col("study_id") == "s2", ~pl.col("dropped_head"))
    assert sorted(s2["joined"]) == ["airspace consolidation", "opacity"]   # "lower" dropped
    assert s2.filter(pl.col("joined") == "opacity")["polarity"].to_list() == ["absent"]


def test_a_head_on_the_drop_lists_is_kept_out_of_the_vocabulary(world):
    found, _ = mentions_of(world)
    assert found.filter(pl.col("dropped_head"))["head"].to_list() == ["stable", "stable"]
    table = vb.vocab_table(found, "joined")
    assert "stable" not in table["string"].to_list()
    effusion = table.filter(pl.col("string") == "loculated effusion")
    assert effusion.select("mentions", "studies").row(0) == (1, 1)


def test_curve_counts_strings_needed_for_each_level():
    assert vb.curve([5, 3, 1, 1], [0.8, 0.9, 0.95, 0.99]) == {0.8: 2, 0.9: 3, 0.95: 4, 0.99: 4}


def test_cli_writes_the_vocabulary_and_reports_the_gate(world, tmp_path, monkeypatch, capsys):
    pool, csv, rg = world
    pool_dir = tmp_path / "pool"
    pool_dir.mkdir()
    pool.write_parquet(pool_dir / "pool_images.parquet")
    (pool_dir / "pool.json").write_text(json.dumps({"pool_sha256": "abc"}), encoding="utf-8")
    umls = tmp_path / "umls"
    umls.mkdir()
    pl.DataFrame({"STR": sorted(NAMES)}).write_parquet(umls / "mrconso.parquet")
    out = tmp_path / "vocab"
    assert main(["kg-pool-vocab", "--pool-dir", str(pool_dir), "--plus-csv", str(csv),
                 "--radgraph-dir", str(rg), "--umls-dir", str(umls), "--out", str(out)]) == 0
    report = json.loads((out / "coverage.json").read_text(encoding="utf-8"))
    assert report["pool_sha256"] == "abc" and report["gate"]["passed"] is True
    assert {p.name for p in out.iterdir()} >= {"mentions.parquet", "vocab_head.parquet",
                                               "vocab_joined.parquet", "coverage.json"}
    assert "gate" in capsys.readouterr().out

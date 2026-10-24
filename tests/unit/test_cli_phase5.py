"""CLI wiring for build-report-index and eval-retrieval (one search path)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import polars as pl
import pytest

from src.cli.main import build_parser, main
from src.common.config import load_config
from src.embeddings import index
from src.query.parser import QueryParser
from src.query.rule_parser import RuleParser
from src.retrieval.report_index import open_report_index, read_meta
from tests.unit.test_label_filter import labels_frame
from tests.unit.test_parser_benchmark import row, write
from tests.unit.test_report_index import ROWS, reports_frame


def _canonical(tmp_path):
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    reports_frame(ROWS).write_parquet(canonical / "reports.parquet")
    return canonical


def test_build_report_index_writes_the_file_and_prints_ascii(tmp_path, capsys):
    canonical = _canonical(tmp_path)
    out = tmp_path / "idx" / "report_search.sqlite"
    assert main(["build-report-index", "--canonical-dir", str(canonical),
                 "--out", str(out)]) == 0
    assert out.is_file() and read_meta(open_report_index(out))["studies"] == "4"
    text = capsys.readouterr().out
    text.encode("ascii")
    assert "rows=4 sha=" in text


def test_eval_retrieval_defaults():
    a = build_parser().parse_args(["eval-retrieval"])
    assert (a.split, a.parsers, a.policies, a.out) == (
        "heldout", ["gold", "rule", "auto"], "negation_both",
        "artifacts/metrics/retrieval_eval.json")


@pytest.mark.parametrize("argv", [["eval-retrieval", "--modes", "hybrid"],
                                  ["eval-retrieval", "--grid"], ["eval-retrieval", "--ablation"],
                                  ["bench-vector", "--mode", "hybrid"]])
def test_the_flags_of_the_removed_modes_are_gone(argv):
    with pytest.raises(SystemExit):
        build_parser().parse_args(argv)


METRICS = {"ndcg_binary@10": 0.4, "ndcg_graded@10": 0.6, "p@10": 0.5, "mrr": 0.7,
           "exclude_violation@10": None, "returned@10": 1.0, "n": 2}
LATENCY = {"p50": 12.0, "p95": 30.0, "n": 1}


def canned(parsers):
    return {"n": 2, "skipped_no_gold": 0, "policies": ["exclude_present/EXCLUDE"],
            "parsers": {p: {"fallback_rate": None if p == "gold" else 0.25, "metrics": METRICS,
                            "by_group": {}, "by_language": {}, "by_policy": {},
                            "latency_ms": LATENCY, "per_query": {"en-001": {"p@10": 0.5}}}
                        for p in parsers}}


@pytest.fixture
def world(tmp_path, monkeypatch):
    """A fake ready state without labels, a rule-only parser, a tiny benchmark and a canonical
    dir whose labels are the answer key."""
    parser = QueryParser(RuleParser(load_config("concepts")), template="x {}")
    status = {"value": "disabled"}
    monkeypatch.setattr("src.api.dependencies.build_query_parser",
                        lambda settings: (parser, status["value"]))
    state = SimpleNamespace(
        ready=True, not_ready_reason=None, report_status="ok", report_reason=None,
        report_meta={"studies": "1"}, images={"i1": SimpleNamespace(projection="PA")},
        manifest={"build_id": "b1", "encoder": "fake"}, parser=parser)
    monkeypatch.setattr("src.api.dependencies.build_state", lambda settings: state)
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    labels_frame([("s1", "Edema", "PRESENT", "CHEXBERT_IMPRESSION")]).write_parquet(
        canonical / "labels.parquet")
    pl.DataFrame({"study_id": ["s1"], "selected_image_id": ["i1"]}).write_parquet(
        canonical / "studies.parquet")
    bench = write(tmp_path, [row(id="en-001", split="dev"), row(id="en-002")])
    return SimpleNamespace(state=state, status=status, canonical=canonical, bench=bench,
                           out=tmp_path / "retrieval.json")


def run(world, *extra):
    return main(["eval-retrieval", "--benchmark", str(world.bench),
                 "--canonical-dir", str(world.canonical), "--out", str(world.out), *extra])


def test_eval_retrieval_exits_1_when_report_index_is_disabled(world, capsys):
    world.state.report_status, world.state.report_reason = "disabled", "no file"
    assert run(world, "--split", "dev", "--parsers", "gold") == 1
    err = capsys.readouterr().err
    assert "report index disabled: no file" in err and "build-report-index" in err
    assert not world.out.exists()


def test_eval_retrieval_exits_1_when_state_is_not_ready(world, capsys):
    world.state.ready, world.state.not_ready_reason = False, "qdrant down"
    assert run(world, "--split", "dev", "--parsers", "gold") == 1
    assert "qdrant down" in capsys.readouterr().err


def test_eval_retrieval_exits_1_when_auto_needs_an_unreachable_llm(world, monkeypatch):
    world.status["value"] = "unreachable"

    def boom(settings):
        raise AssertionError("build_state must not run")

    monkeypatch.setattr("src.api.dependencies.build_state", boom)
    with pytest.raises(SystemExit) as exc:
        run(world, "--split", "dev", "--parsers", "gold", "auto")
    assert exc.value.code == 1 and not world.out.exists()


def test_eval_retrieval_writes_the_eval_section_and_prints_ascii(world, monkeypatch, capsys):
    seen = {}

    def fake_eval(rows, **kw):
        seen.update(kw, ids=[r.id for r in rows])
        return canned(kw["parsers"])

    monkeypatch.setattr("src.evaluation.retrieval_eval.evaluate_retrieval", fake_eval)
    assert run(world, "--split", "dev", "--parsers", "gold", "rule", "--verbose") == 0
    data = json.loads(world.out.read_text(encoding="utf-8"))["eval"]
    assert data["benchmark"]["split"] == "dev" and len(data["benchmark"]["sha256"]) == 64
    assert data["build_id"] == "b1" and data["encoder"] == "fake"
    assert data["report_index"] == {"studies": "1"}
    assert {"llm", "verdict", "fusion"}.isdisjoint(data)
    assert seen["ids"] == ["en-001"] and seen["parsers"] == ["gold", "rule"]
    assert [tuple(p) for p in seen["policies"]] == [
        ("exclude_present", "EXCLUDE"), ("explicit_absence_required", "EXCLUDE")]
    # the answer key comes from labels.parquet, never from the state
    assert list(seen["matrix"].study_ids) == ["s1"] and seen["projection_of"] == {"s1": "PA"}
    out = capsys.readouterr().out
    out.encode("ascii")
    assert ("gold  n=2    ndcg_bin=0.400 (graded 0.600) p@10=0.500 mrr=0.700 viol=n/a "
            "ret=1.000 p50=12ms p95=30ms") in out
    assert "fallback=0.250" in out and out.count('{"id": "en-001", "p@10": 0.5}') == 2


def test_eval_retrieval_default_policy_only_and_llm_section(world, monkeypatch):
    world.status["value"] = "ok"
    seen = {}

    def fake_eval(rows, **kw):
        seen.update(kw)
        return canned(kw["parsers"])

    monkeypatch.setattr("src.evaluation.retrieval_eval.evaluate_retrieval", fake_eval)
    assert run(world, "--split", "dev", "--parsers", "gold", "auto",
               "--policies", "default") == 0
    assert [tuple(p) for p in seen["policies"]] == [("exclude_present", "EXCLUDE")]
    data = json.loads(world.out.read_text(encoding="utf-8"))["eval"]
    assert data["llm"]["status"] == "ok" and "prompt_sha256" in data["llm"]


def test_eval_retrieval_builds_the_state_from_the_canonical_dir_flag(world, monkeypatch):
    seen = {}

    def fake_build_state(settings):
        seen["settings"] = settings
        return world.state

    monkeypatch.setattr("src.api.dependencies.build_state", fake_build_state)
    monkeypatch.setattr("src.evaluation.retrieval_eval.evaluate_retrieval",
                        lambda rows, **kw: canned(kw["parsers"]))
    assert run(world, "--split", "dev", "--parsers", "gold") == 0
    assert seen["settings"].canonical_dir == world.canonical


def test_eval_retrieval_scores_the_build_of_the_encoder_flag(world, monkeypatch, tmp_path):
    """Two encoders' builds side by side; --encoder picks which one the state serves."""
    indexes = tmp_path / "indexes"
    for build_id, encoder in (("b_fake", "fake"), ("b_other", "other")):
        index.write_manifest(index.manifest_path(indexes, build_id),
                             {"build_id": build_id, "encoder": encoder})
    monkeypatch.setenv("INDEXES_DIR", str(indexes))
    seen = {}

    def fake_build_state(settings):
        seen["settings"] = settings
        return world.state

    monkeypatch.setattr("src.api.dependencies.build_state", fake_build_state)
    monkeypatch.setattr("src.evaluation.retrieval_eval.evaluate_retrieval",
                        lambda rows, **kw: canned(kw["parsers"]))
    assert run(world, "--split", "dev", "--parsers", "gold", "--encoder", "other") == 0
    assert seen["settings"].build_id == "b_other"


def test_eval_retrieval_refuses_a_build_id_and_an_encoder_that_disagree(world, monkeypatch,
                                                                         tmp_path, capsys):
    indexes = tmp_path / "indexes"
    index.write_manifest(index.manifest_path(indexes, "b_fake"),
                         {"build_id": "b_fake", "encoder": "fake"})
    monkeypatch.setenv("INDEXES_DIR", str(indexes))
    assert run(world, "--split", "dev", "--parsers", "gold",
               "--build-id", "b_fake", "--encoder", "other") == 1
    assert "build b_fake was embedded by 'fake', not 'other'" in capsys.readouterr().err
    assert not world.out.exists()


def test_eval_retrieval_refuses_an_encoder_without_a_build(world, monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("INDEXES_DIR", str(tmp_path / "empty"))
    assert run(world, "--split", "dev", "--parsers", "gold", "--encoder", "other") == 1
    assert "0 registered builds of encoder 'other'" in capsys.readouterr().err
    assert not world.out.exists()

"""CLI wiring for Phase 5: build-report-index and eval-retrieval."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from src.cli.main import build_parser, main
from src.common.config import load_config
from src.query.parser import QueryParser
from src.query.rule_parser import RuleParser
from src.retrieval.hybrid import FusionParams
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
    assert (a.split, a.parsers, a.modes, a.policies, a.grid, a.out) == (
        "heldout", ["gold", "rule", "auto"], ["label", "vector", "hybrid"], "negation_both",
        False, "artifacts/metrics/retrieval_eval.json")


def test_bench_vector_mode_option():
    assert build_parser().parse_args(["bench-vector", "--mode", "hybrid"]).mode == "hybrid"
    assert build_parser().parse_args(["bench-vector"]).mode == "vector"


METRICS = {"p@10": 0.5, "ndcg@10": 0.6, "mrr": 0.7, "exclude_violation@10": None,
           "returned@10": 1.0, "n": 2}
LATENCY = {"p50": 12.0, "p95": 30.0, "n": 1}
PASSING = {"single_p10": {"passed": True}, "passed": True}
FUSION = {"rrf_k": 60, "leg_weights": {"vector": 1.0, "report": 1.0},
          "vector_top_k": 100, "report_top_k": 100, "image_top_k": 100}


def canned(parsers, modes):
    entry = {"metrics": METRICS, "by_group": {}, "by_language": {}, "by_policy": {},
             "latency_ms": LATENCY, "per_query": {"en-001": {"p@10": 0.5}}}
    return {"n": 2, "skipped_no_gold": 0, "policies": ["exclude_present/EXCLUDE"],
            "parsers": {p: {"fallback_rate": None if p == "gold" else 0.25,
                            "modes": dict.fromkeys(modes, entry)} for p in parsers}}


@pytest.fixture
def world(tmp_path, monkeypatch):
    """A fake ready state, a rule-only parser, a tiny benchmark and canonical dir."""
    parser = QueryParser(RuleParser(load_config("concepts")), template="x {}")
    status = {"value": "disabled"}
    monkeypatch.setattr("src.api.dependencies.build_query_parser",
                        lambda settings: (parser, status["value"]))
    state = SimpleNamespace(
        ready=True, not_ready_reason=None, report_status="ok", report_reason=None,
        report_meta={"studies": "1"}, image_status="ok", image_reason=None,
        labels=SimpleNamespace(study_ids=["s1"]),
        images={"i1": SimpleNamespace(projection="PA")}, study_image={"s1": "i1"},
        manifest={"build_id": "b1"}, parser=parser,
        fusion=FusionParams(60, {"vector": 1.0, "report": 1.0}, 100, 100))
    monkeypatch.setattr("src.api.dependencies.build_state", lambda settings: state)
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    labels_frame([("s1", "Edema", "PRESENT", "CHEXBERT_IMPRESSION")]).write_parquet(
        canonical / "labels.parquet")
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


def test_eval_retrieval_refuses_grid_on_heldout(world, capsys):
    assert run(world, "--split", "heldout", "--grid") == 1
    assert "grid runs on dev only" in capsys.readouterr().err
    assert not world.out.exists()


def test_eval_retrieval_writes_the_eval_section_and_prints_ascii(world, monkeypatch, capsys):
    seen = {}

    def fake_eval(rows, **kw):
        seen.update(kw, ids=[r.id for r in rows])
        return canned(kw["parsers"], kw["modes"])

    monkeypatch.setattr("src.evaluation.retrieval_eval.evaluate_retrieval", fake_eval)
    monkeypatch.setattr("src.evaluation.retrieval_eval.verdict", lambda report, t: PASSING)
    assert run(world, "--split", "dev", "--parsers", "gold", "rule", "--verbose") == 0
    data = json.loads(world.out.read_text(encoding="utf-8"))["eval"]
    assert data["benchmark"]["split"] == "dev" and len(data["benchmark"]["sha256"]) == 64
    assert data["build_id"] == "b1" and data["report_index"] == {"studies": "1"}
    assert data["fusion"] == FUSION
    assert "llm" not in data and data["verdict"] == PASSING
    assert seen["ids"] == ["en-001"] and seen["parsers"] == ["gold", "rule"]
    assert seen["modes"] == ["label", "vector", "hybrid"]
    assert [tuple(p) for p in seen["policies"]] == [
        ("exclude_present", "EXCLUDE"), ("explicit_absence_required", "EXCLUDE")]
    assert seen["projection_of"] == {"s1": "PA"} and seen["matrix"] is world.state.labels
    out = capsys.readouterr().out
    out.encode("ascii")
    assert ("gold  hybrid  n=2    p@10=0.500 ndcg=0.600 mrr=0.700 viol=n/a ret=1.000 "
            "p50=12ms p95=30ms") in out
    assert "fallback=0.250" in out and out.count('{"id": "en-001", "p@10": 0.5}') == 6
    assert "verdict: PASS" in out


def test_eval_retrieval_default_policy_only_and_llm_section(world, monkeypatch, capsys):
    world.status["value"] = "ok"
    seen = {}

    def fake_eval(rows, **kw):
        seen.update(kw)
        return canned(kw["parsers"], kw["modes"])

    monkeypatch.setattr("src.evaluation.retrieval_eval.evaluate_retrieval", fake_eval)
    assert run(world, "--split", "dev", "--parsers", "gold", "auto", "--modes", "label",
               "--policies", "default") == 0
    assert [tuple(p) for p in seen["policies"]] == [("exclude_present", "EXCLUDE")]
    data = json.loads(world.out.read_text(encoding="utf-8"))["eval"]
    assert data["llm"]["status"] == "ok" and "prompt_sha256" in data["llm"]
    assert data["verdict"] is None
    assert "verdict: n/a (needs gold + vector + hybrid)" in capsys.readouterr().out


def test_eval_retrieval_exit_code_follows_a_failing_verdict(world, monkeypatch):
    monkeypatch.setattr("src.evaluation.retrieval_eval.evaluate_retrieval",
                        lambda rows, **kw: canned(kw["parsers"], kw["modes"]))
    monkeypatch.setattr("src.evaluation.retrieval_eval.verdict",
                        lambda report, t: {"passed": False})
    assert run(world, "--split", "dev", "--parsers", "gold") == 1


def test_eval_retrieval_builds_the_state_from_the_canonical_dir_flag(world, monkeypatch):
    seen = {}

    def fake_build_state(settings):
        seen["settings"] = settings
        return world.state

    monkeypatch.setattr("src.api.dependencies.build_state", fake_build_state)
    monkeypatch.setattr("src.evaluation.retrieval_eval.evaluate_retrieval",
                        lambda rows, **kw: canned(kw["parsers"], kw["modes"]))
    monkeypatch.setattr("src.evaluation.retrieval_eval.verdict", lambda report, t: PASSING)
    assert run(world, "--split", "dev", "--parsers", "gold") == 0
    assert seen["settings"].canonical_dir == world.canonical


def test_eval_retrieval_grid_writes_the_grid_section(world, monkeypatch, capsys):
    # the state runs the chosen fusion; the grid section records it next to the plan default
    world.state.fusion = FusionParams(10, {"vector": 0.5, "report": 1.0}, 100, 100)
    config = {"rrf_k": 60, "leg_weights": {"vector": 1.0, "report": 1.0, "image": 0.5},
              "ndcg@10": 0.5, "p@10": 0.4}
    grid = {"configs": [config] * 9, "chosen": config, "baseline_ndcg": 0.5}
    seen = {}

    def fake_grid(rows, **kw):
        seen.update(kw)
        return grid

    monkeypatch.setattr("src.evaluation.retrieval_eval.grid_search", fake_grid)
    assert run(world, "--split", "dev", "--grid") == 0
    assert seen["baseline"] is world.state.fusion
    data = json.loads(world.out.read_text(encoding="utf-8"))
    assert "eval" not in data and data["grid"]["chosen"] == config
    assert data["grid"]["build_id"] == "b1" and data["grid"]["benchmark"]["split"] == "dev"
    assert data["grid"]["fusion_default"] == FUSION
    assert data["grid"]["fusion_current"] == {**FUSION, "rrf_k": 10,
                                              "leg_weights": {"vector": 0.5, "report": 1.0}}
    out = capsys.readouterr().out
    out.encode("ascii")
    assert out.count("rrf_k=60 ") == 9 and out.count("image=0.5 ") == 9
    assert "chosen: " in out and "baseline_ndcg=0.500" in out


def test_eval_retrieval_ablation_writes_its_section_and_prints_one_line_per_config(
        world, monkeypatch, capsys):
    metrics = {"ndcg@10": 0.4, "p@10": 0.3, "exclude_violation@10": None, "n": 1}
    result = {"n": 1, "policy": "exclude_present/EXCLUDE",
              "configs": {"label": metrics, "image": metrics}, "per_query": {}}
    seen = {}

    def fake_ablation(rows, **kw):
        seen.update(kw)
        return result

    monkeypatch.setattr("src.evaluation.retrieval_eval.ablation", fake_ablation)
    assert run(world, "--split", "dev", "--ablation") == 0
    assert seen["fusion"] is world.state.fusion
    data = json.loads(world.out.read_text(encoding="utf-8"))
    assert "eval" not in data and data["ablation"]["configs"]["image"] == metrics
    assert data["ablation"]["build_id"] == "b1"
    out = capsys.readouterr().out
    out.encode("ascii")
    assert "label          ndcg=0.400 p@10=0.300 viol=n/a" in out


def test_eval_retrieval_refuses_ablation_on_heldout(world, capsys):
    assert run(world, "--split", "heldout", "--ablation") == 1
    assert "ablation runs on dev only" in capsys.readouterr().err
    assert not world.out.exists()


def test_eval_retrieval_prints_image_vs_vector_after_the_verdict(world, monkeypatch, capsys):
    monkeypatch.setattr("src.evaluation.retrieval_eval.evaluate_retrieval",
                        lambda rows, **kw: canned(kw["parsers"], kw["modes"]))
    monkeypatch.setattr("src.evaluation.retrieval_eval.verdict", lambda report, t: PASSING)
    comparison = {"n_common": 2, "ndcg@10": {"image": 0.6, "vector": 0.5}, "wins": 1,
                  "ties": 1, "losses": 0, "s2_passed": True}
    monkeypatch.setattr("src.evaluation.retrieval_eval.image_vs_vector",
                        lambda report: comparison)
    assert run(world, "--split", "dev", "--parsers", "gold", "--modes", "vector", "image",
               "hybrid") == 0
    saved = json.loads(world.out.read_text(encoding="utf-8"))["eval"]
    assert saved["image_vs_vector"] == comparison
    assert ("image_vs_vector: n=2 ndcg image=0.600 vector=0.500 wins/ties/losses=1/1/0 s2=PASS"
            in capsys.readouterr().out)
    # without the image mode there is nothing to compare, and no line
    assert run(world, "--split", "dev", "--parsers", "gold", "--modes", "vector", "hybrid") == 0
    assert "image_vs_vector" not in capsys.readouterr().out
    assert "image_vs_vector" not in json.loads(world.out.read_text(encoding="utf-8"))["eval"]


@pytest.mark.parametrize("extra", [("--modes", "label", "vector", "image"), ("--grid",),
                                   ("--ablation",)])
def test_eval_retrieval_refuses_image_dependent_runs_without_the_probe(
        world, monkeypatch, capsys, extra):
    world.state.image_status, world.state.image_reason = "disabled", "probe artifact missing"

    def boom(*a, **kw):
        raise AssertionError("nothing may be evaluated")

    for name in ("evaluate_retrieval", "grid_search", "ablation"):
        monkeypatch.setattr(f"src.evaluation.retrieval_eval.{name}", boom)
    assert run(world, "--split", "dev", "--parsers", "gold", *extra) == 1
    err = capsys.readouterr().err
    assert "image probe disabled: probe artifact missing" in err and "train-probe" in err
    assert not world.out.exists()


def test_eval_retrieval_without_image_still_runs_when_the_probe_is_disabled(world, monkeypatch):
    world.state.image_status = "disabled"
    monkeypatch.setattr("src.evaluation.retrieval_eval.evaluate_retrieval",
                        lambda rows, **kw: canned(kw["parsers"], kw["modes"]))
    monkeypatch.setattr("src.evaluation.retrieval_eval.verdict", lambda report, t: PASSING)
    assert run(world, "--split", "dev", "--parsers", "gold", "--modes", "label", "vector",
               "hybrid") == 0
    assert world.out.exists()

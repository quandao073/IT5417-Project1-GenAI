"""The Phase 4 commands end to end, on a rule-only or scripted parser; no network."""

from __future__ import annotations

import json

import polars as pl
import pytest

from src.cli.main import build_parser, main
from src.common.config import load_config
from src.query import llm_parser as llm
from src.query.parser import QueryParser
from src.query.rule_parser import RuleParser
from tests.unit.fakes import FakeOllama
from tests.unit.test_label_filter import labels_frame
from tests.unit.test_parser_benchmark import row, write


def present(concept):
    return [{"concept": concept, "status": "PRESENT"}]


# Golds the rule parser reproduces exactly.
EXACT = [
    row(id="en-001", split="dev", query="cardiomegaly"),
    row(id="en-002", split="dev", group="multi", query="edema with no pleural effusion",
        include=present("Edema"), exclude=["Pleural Effusion"]),
    row(id="en-003", group="uncertainty", query="possible pneumonia",
        include=[{"concept": "Pneumonia", "status": "UNCERTAIN"}]),
    row(id="en-004", group="view", query="PA view with edema", include=present("Edema"),
        views=["PA"]),
    row(id="vi-001", split="dev", language="vi", query="tim to"),
    row(id="vi-002", language="vi", query="tràn khí màng phổi",
        include=present("Pneumothorax")),
]
# Out-of-lexicon paraphrases: no surface form in concepts.yaml, so the rule parser misses all.
PARAPHRASES = [
    row(id="en-001", query="heart looks too large"),
    row(id="en-002", query="fluid in the lung tissue", include=present("Edema")),
    row(id="en-003", query="air around the lung", include=present("Pneumothorax")),
    row(id="en-004", query="broken rib", include=present("Fracture")),
    row(id="en-005", query="lung infection", include=present("Pneumonia")),
    row(id="en-006", query="water around the lungs", include=present("Pleural Effusion")),
]


def use_parser(monkeypatch, status, llm_client=None) -> QueryParser:
    """_build_parser_or_exit imports build_query_parser at call time, so patch the source."""
    parser = QueryParser(RuleParser(load_config("concepts")), template="x {}",
                         llm_client=llm_client, system_prompt="SYS")
    monkeypatch.setattr("src.api.dependencies.build_query_parser",
                        lambda settings: (parser, status))
    return parser


def test_parse_defaults():
    args = build_parser().parse_args(["parse", "tim to"])
    assert args.text == "tim to" and args.parser == "auto"


def test_eval_parser_defaults():
    args = build_parser().parse_args(["eval-parser"])
    assert args.benchmark == "benchmarks/parser_queries_v1.jsonl" and args.split == "heldout"
    assert args.parsers == ["rule", "llm", "auto"]
    assert args.out == "artifacts/metrics/parser_eval.json"
    assert args.label_filter is False and args.canonical_dir == "data/canonical"
    assert args.verbose is False


def test_eval_parser_accepts_a_subset_of_parsers():
    args = build_parser().parse_args(["eval-parser", "--parsers", "rule", "--split", "all"])
    assert args.parsers == ["rule"] and args.split == "all"


def test_eval_parser_exits_1_when_llm_requested_but_unavailable(tmp_path, monkeypatch):
    use_parser(monkeypatch, "unreachable")
    out = tmp_path / "eval.json"
    with pytest.raises(SystemExit) as exc:
        main(["eval-parser", "--benchmark", str(write(tmp_path, EXACT)), "--split", "all",
              "--parsers", "llm", "--out", str(out)])
    assert exc.value.code == 1 and not out.exists()


@pytest.mark.parametrize("rows, code", [(EXACT, 0), (PARAPHRASES, 1)],
                         ids=["rule-exact", "out-of-lexicon"])
def test_eval_parser_rule_only_exit_code_follows_the_verdict(tmp_path, monkeypatch, rows, code):
    use_parser(monkeypatch, "disabled")
    out = tmp_path / "eval.json"
    assert main(["eval-parser", "--benchmark", str(write(tmp_path, rows)), "--split", "all",
                 "--parsers", "rule", "--out", str(out)]) == code
    report = json.loads(out.read_text(encoding="utf-8"))["eval"]
    assert report["judged_parser"] == "rule"
    assert report["verdict"]["passed"] is (code == 0)


def test_parse_prints_ascii_json(capsys, monkeypatch):
    use_parser(monkeypatch, "disabled")
    assert main(["parse", "tìm tim to", "--parser", "rule"]) == 0
    printed = capsys.readouterr().out
    payload = json.loads(printed.encode("ascii").decode("ascii"))
    assert payload["include"] == [{"concept": "Cardiomegaly", "status": "PRESENT"}]
    assert payload["parser"] == "rule" and payload["llm_status"] == "disabled"


def test_eval_parser_warms_up_the_llm_and_survives_a_failed_warm_up(tmp_path, monkeypatch):
    # The first call after a server start can exceed the timeout (WORKLOG T16).
    reply = json.dumps({"findings": "", "include": present("Cardiomegaly"), "exclude": [],
                        "views": [], "language": "en"})
    fake = FakeOllama("ok", [llm.LLMError("ReadTimeout"), reply])
    use_parser(monkeypatch, "ok", llm_client=fake)
    out = tmp_path / "eval.json"
    assert main(["eval-parser", "--benchmark", str(write(tmp_path, EXACT[:1])), "--split", "all",
                 "--parsers", "llm", "--out", str(out)]) == 0
    assert [call[-1]["content"] for call in fake.calls] == ["chest x-ray", "cardiomegaly"]
    metrics = json.loads(out.read_text(encoding="utf-8"))["eval"]["parsers"]["llm"]["metrics"]
    assert metrics["schema_valid_rate"] == 1.0


def test_label_filter_section_names_its_benchmark(tmp_path, monkeypatch):
    use_parser(monkeypatch, "disabled")
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    pl.DataFrame({"study_id": ["s1", "s2"]}).write_parquet(canonical / "studies.parquet")
    labels_frame([("s1", "Cardiomegaly", "PRESENT", "CHEXPERT_V1")]).write_parquet(
        canonical / "labels.parquet")
    out = tmp_path / "eval.json"
    main(["eval-parser", "--benchmark", str(write(tmp_path, EXACT)), "--split", "all",
          "--parsers", "rule", "--out", str(out), "--label-filter",
          "--canonical-dir", str(canonical)])
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["label_filter"]["benchmark"] == report["eval"]["benchmark"]
    assert report["label_filter"]["per_query"]["en-001"]["exclude_present/EXCLUDE"] == 1

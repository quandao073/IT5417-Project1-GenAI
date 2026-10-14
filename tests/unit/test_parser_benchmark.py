"""The benchmark file is data the thesis argues from: keep it structurally honest."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from src.common.config import load_config
from src.data.labels import CHEXPERT_LABELS
from src.evaluation import parser_benchmark as pb
from src.query import llm_parser as llm
from src.query.rule_parser import normalize
from src.query.schema import SearchQuery

BENCH = Path("benchmarks/parser_queries_v1.jsonl")


def row(id="en-001", split="heldout", language="en", group="single", query="cardiomegaly",
        include=({"concept": "Cardiomegaly", "status": "PRESENT"},), exclude=(), views=()):
    return {"id": id, "split": split, "language": language, "group": group, "query": query,
            "gold": {"include": list(include), "exclude": list(exclude), "views": list(views)}}


def write(tmp_path, rows) -> Path:
    path = tmp_path / "b.jsonl"
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                    encoding="utf-8")
    return path


class TestDeriveGroups:
    @pytest.mark.parametrize("include, exclude, views, expected", [
        ([("Edema", "PRESENT")], [], [], ("single",)),
        ([("Edema", "PRESENT"), ("Cardiomegaly", "PRESENT")], [], [], ("multi",)),
        ([("Edema", "PRESENT")], ["Pleural Effusion"], [], ("multi", "negation")),
        ([], ["Pleural Effusion"], [], ("negation",)),
        ([("Pneumonia", "UNCERTAIN")], [], [], ("single", "uncertainty")),
        ([("Edema", "PRESENT")], [], ["PA"], ("single", "view")),
        ([], [], [], ("other",)),
        ([], [], ["AP"], ("view", "other")),
    ])
    def test_groups_follow_the_gold(self, include, exclude, views, expected):
        gold = SearchQuery(language="en",
                           include=[{"concept": c, "status": s} for c, s in include],
                           exclude=exclude, views=views)
        assert pb.derive_groups(gold) == expected


class TestLoader:
    def test_reads_rows_and_validates_gold(self, tmp_path):
        rows = pb.load_benchmark(write(tmp_path, [row()]))
        assert rows[0].id == "en-001" and rows[0].gold.include[0].concept == "Cardiomegaly"
        assert rows[0].groups == ("single",) and rows[0].gold.language == "en"

    def test_primary_group_must_be_one_of_the_derived_groups(self, tmp_path):
        with pytest.raises(ValueError, match="en-001.*group"):
            pb.load_benchmark(write(tmp_path, [row(group="negation")]))

    @pytest.mark.parametrize("field, value", [("split", "test"), ("language", "fr"),
                                              ("group", "misc")])
    def test_enumerations_are_checked(self, tmp_path, field, value):
        with pytest.raises(ValueError, match="en-001"):
            pb.load_benchmark(write(tmp_path, [row(**{field: value})]))

    def test_unknown_concept_in_gold_is_rejected(self, tmp_path):
        bad = row(include=[{"concept": "Cardiomegalyy", "status": "PRESENT"}])
        with pytest.raises(ValueError, match="en-001"):
            pb.load_benchmark(write(tmp_path, [bad]))

    def test_duplicate_ids_are_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="duplicate"):
            pb.load_benchmark(write(tmp_path, [row(), row(query="other words")]))

    @pytest.mark.parametrize("text, message", [('{"id": "en-001",', "line 2: not JSON"),
                                               ("[1, 2]", "line 2: not a JSON object")])
    def test_a_malformed_line_names_its_line(self, tmp_path, text, message):
        path = tmp_path / "b.jsonl"
        path.write_text(json.dumps(row()) + "\n" + text + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            pb.load_benchmark(path)

    @pytest.mark.parametrize("key", pb.REQUIRED_KEYS)
    def test_a_missing_key_is_named(self, tmp_path, key):
        bad = row()
        del bad[key]
        rid = "line 1" if key == "id" else "en-001"
        with pytest.raises(ValueError, match=f"{rid}: missing key '{key}'"):
            pb.load_benchmark(write(tmp_path, [bad]))

    def test_an_unknown_gold_key_is_rejected(self, tmp_path):
        bad = row()
        bad["gold"]["includes"] = bad["gold"].pop("include")
        with pytest.raises(ValueError, match=r"en-001: unknown gold key\(s\) \['includes'\]"):
            pb.load_benchmark(write(tmp_path, [bad]))


@pytest.fixture(scope="module")
def rows():
    return pb.load_benchmark(BENCH)


class TestShippedFile:
    def test_two_hundred_rows_half_per_language(self, rows):
        assert len(rows) == 200
        assert Counter(r.language for r in rows) == {"en": 100, "vi": 100}

    def test_ids_are_sequential_per_language(self, rows):
        for lang in ("en", "vi"):
            ids = [r.id for r in rows if r.language == lang]
            assert ids == [f"{lang}-{i:03d}" for i in range(1, 101)]

    def test_primary_groups_follow_the_quota(self, rows):
        for lang in ("en", "vi"):
            got = Counter(r.group for r in rows if r.language == lang)
            assert got == pb.PRIMARY_QUOTA, (lang, got)

    def test_primary_group_follows_the_pattern(self, rows):
        for r in rows:
            i = int(r.id.split("-")[1])
            assert r.group == pb.PATTERN[(i - 1) % 20], r.id

    def test_every_concept_appears_at_least_four_times_per_language(self, rows):
        for lang in ("en", "vi"):
            got = Counter(c for r in rows if r.language == lang
                          for c in r.gold.constrained_concepts())
            thin = {c: got[c] for c in CHEXPERT_LABELS if got[c] < 4}
            assert not thin, (lang, thin)

    def test_dev_split_follows_its_quota_and_the_rest_is_heldout(self, rows):
        for lang in ("en", "vi"):
            dev = Counter(r.group for r in rows if r.language == lang and r.split == "dev")
            assert dev == pb.DEV_QUOTA, (lang, dev)
        assert Counter(r.split for r in rows) == {"dev": 40, "heldout": 160}

    def test_queries_are_unique_after_normalisation(self, rows):
        texts = [normalize(r.query) for r in rows]
        assert len(set(texts)) == len(texts)

    def test_language_matches_the_text(self, rows):
        for r in rows:
            vi = any(ch in "ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ"
                     for ch in normalize(r.query))
            assert vi == (r.language == "vi"), r.id

    def test_few_shot_examples_are_not_heldout_queries(self, rows):
        prompt = llm.load_prompt(load_config("models")["llm"]["prompt_file"],
                                 load_config("concepts"))
        examples = {normalize(e) for e in llm.prompt_examples(prompt)}
        heldout = {normalize(r.query) for r in rows if r.split == "heldout"}
        assert not examples & heldout

    def test_queries_are_short_requests_not_reports(self, rows):
        for r in rows:
            assert 3 <= len(r.query) <= 160, r.id
            assert "\n" not in r.query, r.id

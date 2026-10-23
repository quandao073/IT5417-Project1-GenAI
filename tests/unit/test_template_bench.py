"""Known-answer tests for the template benchmark (V1.1 WP1, spec 5.2): relevance under the
two absence policies, HNRR, the per-query metrics, skipping, bootstrap and the query sets.
A six-study toy gallery; every scorer is a fake."""

from __future__ import annotations

import numpy as np
import pytest

from src.common.config import load_config
from src.data.labels import CHEXPERT_LABELS
from src.evaluation import template_bench as tb
from src.retrieval import label_filter as lf
from src.retrieval import templates as tp
from tests.unit.test_label_filter import labels_frame

CFG = load_config("concepts")
TRAIN = load_config("training")
PRIORITY = ("CHEXPERT_V1", "CHEXBERT_IMPRESSION", "CHEXBERT_FINDINGS", "CHEXBERT_REPORT")
COL = {c: j for j, c in enumerate(CHEXPERT_LABELS)}

# "pulmonary edema and no pleural effusion": s1, s2, s4, s5 relevant under default
# (NOT_MENTIONED passes the exclude), only s1, s2 under strict (ABSENT required);
# s3 has both PRESENT (the HNRR offender); s6's UNCERTAIN effusion fails both policies.
STUDIES = ["s1", "s2", "s3", "s4", "s5", "s6"]
EFFUSION = ["ABSENT", "ABSENT", "PRESENT", "NOT_MENTIONED", "NOT_MENTIONED", "UNCERTAIN"]


def toy_statuses() -> np.ndarray:
    statuses = np.full((6, len(CHEXPERT_LABELS)), "NOT_MENTIONED", dtype=object)
    statuses[:, COL["Edema"]] = "PRESENT"
    statuses[:, COL["Pleural Effusion"]] = EFFUSION
    statuses[5, COL["Cardiomegaly"]] = "PRESENT"
    return statuses


def toy_gallery() -> tb.Gallery:
    return tb.Gallery(study_ids=np.asarray(STUDIES, dtype=object), statuses=toy_statuses(),
                      vectors=None, projections=None)


def query(include, exclude=(), kind=None) -> tp.TemplateQuery:
    kind = kind or ("negation" if exclude else "single")
    (q,) = [q for q in tp.generate(list(include) + list(exclude), [kind], CFG)
            if set(q.include) == set(include) and set(q.exclude) == set(exclude)]
    return q


def fixed_scorer(rows: dict[str, list[float]]):
    """Scores by query text; a query not in `rows` gets zeros (ties -> gallery order)."""
    def scorer(queries):
        return np.array([rows.get(q.text, [0.0] * 6) for q in queries], dtype=np.float64)
    return scorer


EDEMA = query(["Edema"])
NEGATION = query(["Edema"], ["Pleural Effusion"])


class TestRelevance:
    def test_relevant_mask_matches_label_filter_truth_table(self):
        statuses = toy_statuses()
        default = tb.relevant_mask(statuses, NEGATION, tb.DEFAULT_POLICY)
        strict = tb.relevant_mask(statuses, NEGATION, tb.STRICT_POLICY)
        assert default.tolist() == [True, True, False, True, True, False]
        assert strict.tolist() == [True, True, False, False, False, False]
        assert tb.relevant_mask(statuses, EDEMA, tb.STRICT_POLICY).all()
        # The same truth table as the label filter built from a long frame.
        rows = [(s, c, statuses[i, COL[c]], "CHEXPERT_V1") for i, s in enumerate(STUDIES)
                for c in ("Edema", "Pleural Effusion", "Cardiomegaly")]
        matrix = lf.build_label_matrix(labels_frame(rows), STUDIES, PRIORITY, seed=1)
        for policy in (tb.DEFAULT_POLICY, tb.STRICT_POLICY):
            want = lf.filter_studies_by_labels(
                matrix, NEGATION.gold.include, NEGATION.gold.exclude,
                absence_policy=policy[0], uncertainty_policy=policy[1])
            assert tb.relevant_mask(statuses, NEGATION, policy).tolist() == want.tolist()

    def test_label_matrix_from_statuses_codes_every_status(self):
        matrix = tb.label_matrix_from_statuses(np.asarray(STUDIES, dtype=object), toy_statuses())
        col = matrix.status[:, matrix.col_of["Pleural Effusion"]]
        assert col.tolist() == [lf.STATUS_CODE[s] for s in EFFUSION]
        assert matrix.status.dtype == np.int8 and matrix.row_of["s3"] == 2
        bad = toy_statuses()
        bad[0, 0] = "MAYBE"
        with pytest.raises(ValueError, match="MAYBE"):
            tb.label_matrix_from_statuses(np.asarray(STUDIES, dtype=object), bad)

    def test_hnrr_mask_marks_both_present(self):
        assert tb.hnrr_mask(toy_statuses(), NEGATION).tolist() == [
            False, False, True, False, False, False]

    def test_violation_mask_marks_any_exclude_present(self):
        statuses = toy_statuses()
        statuses[0, COL["Edema"]] = "ABSENT"          # s1 no longer has the include
        assert tb.violation_mask(statuses, NEGATION).tolist() == [
            False, False, True, False, False, False]
        assert tb.hnrr_mask(statuses, NEGATION).tolist() == [
            False, False, True, False, False, False]


class TestQueryMetrics:
    def test_query_metrics_values(self):
        statuses = toy_statuses()
        relevant = tb.relevant_mask(statuses, NEGATION, tb.DEFAULT_POLICY)
        hnrr, violation = tb.hnrr_mask(statuses, NEGATION), tb.violation_mask(statuses, NEGATION)
        ranked = np.array([0, 2, 1])                      # s1, s3, s2
        m = tb.query_metrics(ranked, relevant, hnrr, violation, ks=(1, 3, 10))
        assert m["p@1"] == 1.0 and m["p@3"] == pytest.approx(2 / 3)
        assert m["hit@1"] == 1.0 and m["hit@3"] == 1.0
        assert m["hnrr@1"] == 0.0 and m["hnrr@3"] == pytest.approx(1 / 3)
        assert m["exclude_violation@10"] == pytest.approx(1 / 3)
        assert m["n_relevant"] == 4

    def test_query_metrics_without_exclude_are_none(self):
        relevant = np.array([True, False, False, False, False, False])
        m = tb.query_metrics(np.array([1, 2, 0]), relevant, None, None, ks=(1, 3))
        assert m["p@1"] == 0.0 and m["p@3"] == pytest.approx(1 / 3)
        assert m["hit@1"] == 0.0 and m["hit@3"] == 1.0
        assert m["hnrr@1"] is None and m["hnrr@3"] is None
        assert m["exclude_violation@10"] is None


class TestEvaluate:
    def test_evaluate_skips_small_relevant_and_counts(self):
        scorer = fixed_scorer({NEGATION.text: [6, 5, 4, 3, 2, 1], EDEMA.text: [1, 2, 3, 4, 5, 6]})
        report = tb.evaluate({"toy": [EDEMA, NEGATION]}, toy_gallery(), scorer,
                             policies=(tb.DEFAULT_POLICY, tb.STRICT_POLICY), ks=(1, 3),
                             min_relevant=3, n_boot=10)
        assert report["skipped_small_relevant"] == 1
        assert report["policies"] == ["default", "strict"]
        sets = report["sets"]["toy"]
        assert sets["single"]["default"]["p@3"]["n"] == 1
        assert sets["single"]["strict"]["p@3"]["n"] == 1
        assert sets["negation"]["default"]["p@3"]["n"] == 1
        assert sets["negation"]["strict"]["p@3"] == {"mean": None, "ci95": None, "n": 0}
        # ranked s1, s2, s3 for the negation: p@3 2/3, hnrr@3 1/3, violation 1/3 (s3 of six)
        neg = sets["negation"]["default"]
        assert neg["p@3"]["mean"] == pytest.approx(2 / 3)
        assert neg["hnrr@3"]["mean"] == pytest.approx(1 / 3)
        assert neg["exclude_violation@10"]["mean"] == pytest.approx(1 / 6)
        assert sets["single"]["default"]["hnrr@3"] == {"mean": None, "ci95": None, "n": 0}
        per = report["per_query"]["toy"]
        assert per[NEGATION.id]["policies"]["strict"] is None
        assert per[NEGATION.id]["policies"]["default"]["n_relevant"] == 4
        assert per[EDEMA.id]["kind"] == "single" and per[EDEMA.id]["text"] == "pulmonary edema"

    def test_evaluate_rejects_a_scorer_of_the_wrong_shape(self):
        with pytest.raises(ValueError, match="shape"):
            tb.evaluate({"toy": [EDEMA]}, toy_gallery(), lambda qs: np.zeros((2, 6)),
                        policies=(tb.DEFAULT_POLICY,), n_boot=5)

    def test_evaluate_breaks_ties_by_gallery_row(self):
        report = tb.evaluate({"toy": [NEGATION]}, toy_gallery(), fixed_scorer({}),
                             policies=(tb.DEFAULT_POLICY,), ks=(3,), min_relevant=1, n_boot=5)
        # all scores equal -> s1, s2, s3 first -> p@3 2/3
        assert report["sets"]["toy"]["negation"]["default"]["p@3"]["mean"] == pytest.approx(2 / 3)


def random_world(n: int = 40, seed: int = 7):
    rng = np.random.default_rng(seed)
    states = np.array(["PRESENT", "ABSENT", "UNCERTAIN", "NOT_MENTIONED"], dtype=object)
    statuses = states[rng.integers(0, 4, size=(n, len(CHEXPERT_LABELS)))].astype(object)
    gallery = tb.Gallery(np.array([f"g{i}" for i in range(n)], dtype=object), statuses,
                         vectors=None, projections=None)
    queries = tp.generate(["Edema", "Pleural Effusion", "Cardiomegaly", "Pneumothorax"],
                          ["single", "pair", "negation"], CFG)
    return gallery, queries


def random_scorer(seed: int):
    def scorer(queries):
        return np.random.default_rng(seed).normal(size=(len(queries), 40))
    return scorer


def test_evaluate_bootstrap_ci_brackets_mean():
    gallery, queries = random_world()
    kwargs = dict(policies=(tb.DEFAULT_POLICY,), min_relevant=1, n_boot=200, seed=3)
    a = tb.evaluate({"rnd": queries}, gallery, random_scorer(1), **kwargs)
    b = tb.evaluate({"rnd": queries}, gallery, random_scorer(1), **kwargs)
    assert a == b
    seen = 0
    for kind, per_policy in a["sets"]["rnd"].items():
        for metric, cell in per_policy["default"].items():
            if cell["n"] == 0:
                assert cell["mean"] is None and cell["ci95"] is None
                continue
            lo, hi = cell["ci95"]
            assert lo <= cell["mean"] <= hi, (kind, metric, cell)
            seen += 1
    assert seen > 10
    assert tb.evaluate({"rnd": queries}, gallery, random_scorer(1), **{**kwargs, "seed": 4}) != a


def test_paired_delta_sign():
    gallery, queries = random_world()

    def oracle(qs):
        return np.stack([tb.relevant_mask(gallery.statuses, q, tb.DEFAULT_POLICY).astype(float)
                         for q in qs])

    kwargs = dict(policies=(tb.DEFAULT_POLICY,), min_relevant=1, n_boot=200)
    worse = tb.evaluate({"rnd": queries}, gallery, random_scorer(1), **kwargs)
    better = tb.evaluate({"rnd": queries}, gallery, oracle, **kwargs)
    delta = tb.paired_delta(worse, better, set_name="rnd", kinds=("pair", "negation"),
                            policy="default", metric="p@5", n_boot=200)
    assert delta["n"] > 10 and delta["mean"] > 0 and delta["ci95"][0] > 0
    assert delta["b"] > delta["a"]
    assert delta["mean"] == pytest.approx(delta["b"] - delta["a"])
    back = tb.paired_delta(better, worse, set_name="rnd", kinds=("pair", "negation"),
                           policy=tb.DEFAULT_POLICY, metric="p@5", n_boot=200)
    assert back["mean"] == pytest.approx(-delta["mean"])
    empty = tb.paired_delta(worse, better, set_name="rnd", kinds=("single",), policy="default",
                            metric="hnrr@5", n_boot=20)
    assert empty == {"mean": None, "ci95": None, "n": 0, "a": None, "b": None}


def test_query_sets_sizes():
    sets = tb.query_sets(CFG, TRAIN)
    assert list(sets) == ["R10", "S12", "paraphrase", "multi"]
    counts = {name: {k: n for k in tp.KINDS if (n := sum(q.kind == k for q in qs))}
              for name, qs in sets.items()}
    assert counts["R10"] == {"single": 10, "pair": 45, "negation": 90}
    assert counts["S12"] == {"single": 12, "pair": 55, "negation": 110}
    assert counts["paraphrase"] == {"paraphrase": 135}
    assert counts["multi"] == {"triple": 30, "pair_negation": 30}
    assert {c for q in sets["R10"] for c in q.include} == set(TRAIN["paper_concepts"])
    assert {c for q in sets["paraphrase"] for c in q.include} == set(TRAIN["paper_concepts"])
    assert "No Finding" in {c for q in sets["S12"] for c in q.include}
    assert tb.query_sets(CFG, TRAIN)["multi"][0].text == sets["multi"][0].text
    assert tb.study_ids_sha256(["b", "a"]) == tb.study_ids_sha256(np.array(["a", "b"]))
    assert len(tb.study_ids_sha256(["a"])) == 64

"""score_images on the in-RAM matrix (src/kg/scoring.py)."""

from __future__ import annotations

import numpy as np
import pytest

from src.kg import scoring
from src.kg.expand import ABSENT, NOISY_OR, PRESENT, SINGLE, WEIGHTED, Column, Intent

COLS = ["A", "B", "C"]
Z = np.array([[2.0, -1.0, 0.5, -3.0], [-0.5, 1.5, 0.0, 2.5], [0.2, -2.0, 1.0, 0.1]])


def matrix(views=("AP", "PA", "AP", "AP")):
    return scoring.ScoreMatrix(COLS, Z, [f"s{i}" for i in range(4)], [f"i{i}" for i in range(4)],
                               np.array(views, dtype="U2"))


def intent(polarity, combine, weights):
    cols = [Column(cui=c, column=c, weight=w, quality=None, weak=False, borderline=False)
            for c, w in weights]
    return Intent("t", "x", "x", polarity, "test", combine, cols)


def sigma(z):
    return 1 / (1 + np.exp(-z.astype(np.float64)))


@pytest.mark.parametrize("polarity", [PRESENT, ABSENT])
def test_each_kind_of_intent_matches_the_formula(polarity):
    z = {c: Z[j] for j, c in enumerate(COLS)}
    m = matrix()
    sign = 1 if polarity == PRESENT else -1
    # single
    got = m.contribution(intent(polarity, SINGLE, [("A", 1.0)]))
    want = np.log(sigma(sign * z["A"]))
    assert np.allclose(got, want, atol=1e-5)
    # noisy-OR over B and C: p = 1 - prod(1 - sigma)
    p = 1 - (1 - sigma(z["B"])) * (1 - sigma(z["C"]))
    got = m.contribution(intent(polarity, NOISY_OR, [("B", 1.0), ("C", 1.0)]))
    want = np.log(p) if polarity == PRESENT else np.log(1 - p)
    assert np.allclose(got, want, atol=1e-4)
    # weighted: present sum w log sigma(z); absent log(1 - sum w sigma(z))
    w = [("A", 0.75), ("C", 0.25)]
    got = m.contribution(intent(polarity, WEIGHTED, w))
    if polarity == PRESENT:
        want = 0.75 * np.log(sigma(z["A"])) + 0.25 * np.log(sigma(z["C"]))
    else:
        want = np.log(1 - (0.75 * sigma(z["A"]) + 0.25 * sigma(z["C"])))
    assert np.allclose(got, want, atol=1e-4)


def test_the_score_is_the_sum_of_intents_and_an_empty_intent_adds_nothing():
    m = matrix()
    a, b = intent(PRESENT, SINGLE, [("A", 1.0)]), intent(ABSENT, SINGLE, [("B", 1.0)])
    skipped = intent(ABSENT, SINGLE, [])
    assert np.allclose(m.score([a, b, skipped]), m.contribution(a) + m.contribution(b))
    assert m.contribution(skipped) is None


def test_top_k_filters_views_and_orders_by_score_then_row():
    m = matrix()
    scores = np.array([0.1, 0.9, 0.9, 0.5], dtype=np.float32)
    assert m.top_k(scores, 3).tolist() == [1, 2, 3]            # the tie at 0.9 goes by row
    assert m.top_k(scores, 3, ("AP",)).tolist() == [2, 3, 0]   # row 1 is PA
    assert m.top_k(scores, 5, ("PA",)).tolist() == [1]          # fewer rows than k
    assert m.top_k(scores, 2, ("XX",)).tolist() == []


def test_probabilities_are_the_numbers_the_explanation_may_quote():
    m = matrix()
    wide = intent(PRESENT, NOISY_OR, [("B", 1.0), ("C", 1.0)])
    got = m.probabilities(np.array([0, 3]), [wide])[0][0]
    p_b, p_c = sigma(Z[1][0]), sigma(Z[2][0])
    assert got["intent"] == pytest.approx(1 - (1 - p_b) * (1 - p_c))
    assert got["columns"] == {"B": pytest.approx(p_b), "C": pytest.approx(p_c)}


def test_the_benchmark_reports_cold_and_warm_times_for_both_mixes():
    out = scoring.benchmark(2000, [f"c{i}" for i in range(12)], repeats=3)
    assert out["rows"] == 2000 and out["columns"] == 12
    assert out["typical"]["columns_touched"] == 8 and out["heavy"]["columns_touched"] == 12
    assert all(out[m]["cold_ms"] > 0 and out[m]["warm_p50_ms"] > 0 for m in ("typical", "heavy"))


def test_preload_fills_every_column_and_changes_no_score():
    m, fresh = matrix(), matrix()
    intents = [scoring._intent("a", "PRESENT", scoring.NOISY_OR, [(COLS[0], 1.0), (COLS[1], 1.0)]),
               scoring._intent("b", "ABSENT", scoring.SINGLE, [(COLS[2], 1.0)])]
    expected = fresh.score(intents)
    assert m.preload() >= 0 and len(m._lsp) == len(m._lsn) == len(COLS)
    assert (m.score(intents) == expected).all()

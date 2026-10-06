"""Zero-shot metrics for the model gate.

The negative set is where this project differs from the usual CheXpert
convention: NOT_MENTIONED is never an ABSENT, so it never enters the strict
negatives. That rule gets its own test rather than being left implicit.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.evaluation import zero_shot as zs


def _mask(*flags):
    return np.array(flags, dtype=bool)


class TestAuroc:
    def test_perfect_separation_is_one(self):
        scores = np.array([0.9, 0.8, 0.1, 0.2])
        assert zs.auroc(scores, _mask(1, 1, 0, 0), _mask(0, 0, 1, 1)) == 1.0

    def test_reversed_separation_is_zero(self):
        scores = np.array([0.1, 0.2, 0.9, 0.8])
        assert zs.auroc(scores, _mask(1, 1, 0, 0), _mask(0, 0, 1, 1)) == 0.0

    def test_all_ties_is_one_half(self):
        scores = np.full(4, 0.3)
        assert zs.auroc(scores, _mask(1, 1, 0, 0), _mask(0, 0, 1, 1)) == 0.5

    def test_matches_a_hand_computed_case(self):
        # pos {0.35, 0.8} vs neg {0.1, 0.4}: 3 of 4 pairs ordered correctly
        scores = np.array([0.1, 0.4, 0.35, 0.8])
        assert zs.auroc(scores, _mask(0, 0, 1, 1), _mask(1, 1, 0, 0)) == pytest.approx(0.75)

    def test_a_tie_between_classes_counts_one_half(self):
        # pos {0.5} vs neg {0.5, 0.2}: (0.5 + 1) / 2
        scores = np.array([0.5, 0.5, 0.2])
        assert zs.auroc(scores, _mask(1, 0, 0), _mask(0, 1, 1)) == pytest.approx(0.75)

    def test_rows_in_neither_mask_are_ignored(self):
        scores = np.array([0.9, 0.1, 100.0])
        assert zs.auroc(scores, _mask(1, 0, 0), _mask(0, 1, 0)) == 1.0

    def test_a_missing_class_is_nan(self):
        assert math.isnan(zs.auroc(np.array([0.1, 0.2]), _mask(1, 1), _mask(0, 0)))


class TestLabelMasks:
    def test_strict_negatives_never_include_not_mentioned(self):
        statuses = np.array(["PRESENT", "ABSENT", "NOT_MENTIONED", "UNCERTAIN"], dtype=object)
        pos, strict, lenient = zs.label_masks(statuses)
        assert pos.tolist() == [True, False, False, False]
        assert strict.tolist() == [False, True, False, False]
        assert lenient.tolist() == [False, True, True, False]

    def test_uncertain_is_in_no_mask(self):
        pos, strict, lenient = zs.label_masks(np.array(["UNCERTAIN"], dtype=object))
        assert not (pos[0] or strict[0] or lenient[0])


def _statuses(n_pos, n_abs, n_nm, n_unc=0):
    return np.array(
        ["PRESENT"] * n_pos + ["ABSENT"] * n_abs + ["NOT_MENTIONED"] * n_nm
        + ["UNCERTAIN"] * n_unc,
        dtype=object,
    )


class TestConceptMetrics:
    def test_counts_and_perfect_scores(self):
        statuses = _statuses(12, 10, 5, 3)
        scores = np.where(statuses == "PRESENT", 1.0, 0.0)
        m = zs.concept_metrics(scores, statuses)
        assert (m["n_pos"], m["n_neg_strict"], m["n_neg_lenient"]) == (12, 10, 15)
        assert m["auroc_strict"] == 1.0
        assert m["auroc_lenient"] == 1.0
        assert m["p_at_10"] == 1.0
        assert m["prevalence"] == pytest.approx(12 / 30)

    def test_too_few_strict_negatives_gives_null(self):
        statuses = _statuses(12, 9, 20)
        m = zs.concept_metrics(np.linspace(0, 1, len(statuses)), statuses)
        assert m["auroc_strict"] is None
        assert m["auroc_lenient"] is not None

    def test_too_few_positives_gives_null_for_both(self):
        statuses = _statuses(9, 20, 20)
        m = zs.concept_metrics(np.linspace(0, 1, len(statuses)), statuses)
        assert m["auroc_strict"] is None and m["auroc_lenient"] is None

    def test_p_at_10_uses_the_ten_highest_scores(self):
        statuses = _statuses(5, 10, 15)
        scores = np.zeros(len(statuses))
        scores[:5] = 1.0          # the five positives
        scores[5:10] = 0.9        # five absents fill the rest of the top ten
        assert zs.concept_metrics(scores, statuses)["p_at_10"] == 0.5

    def test_cosine_summary_per_group(self):
        statuses = _statuses(10, 10, 0)
        scores = np.where(statuses == "PRESENT", 0.3, 0.1)
        cos = zs.concept_metrics(scores, statuses)["cosine"]
        assert cos["present"]["mean"] == pytest.approx(0.3)
        assert cos["absent"]["p95"] == pytest.approx(0.1)

    def test_empty_group_has_null_cosine_summary(self):
        statuses = _statuses(10, 0, 10)
        assert zs.concept_metrics(np.zeros(20), statuses)["cosine"]["absent"] is None


class TestEligibleAndMacro:
    def test_eligibility_depends_on_labels_only(self):
        statuses = np.stack(
            [_statuses(10, 10, 0), _statuses(10, 9, 1), _statuses(9, 11, 0)], axis=1
        )
        assert zs.eligible_concepts(statuses) == [0]

    def test_macro_is_the_mean(self):
        assert zs.macro([0.6, 0.8]) == pytest.approx(0.7)

    def test_macro_of_nothing_is_none(self):
        assert zs.macro([]) is None


class TestBootstrap:
    def _data(self):
        rng = np.random.default_rng(0)
        statuses = np.stack([_statuses(30, 30, 40), _statuses(25, 35, 40)], axis=1)
        good = (statuses == "PRESENT").astype(float) + rng.normal(0, 0.5, statuses.shape)
        noise = rng.normal(0, 1, statuses.shape)
        return statuses, {"good": good, "noise": noise}

    def test_is_deterministic_for_a_seed(self):
        statuses, scores = self._data()
        a = zs.bootstrap_macro(scores, statuses, [0, 1], n_boot=50, seed=7)
        b = zs.bootstrap_macro(scores, statuses, [0, 1], n_boot=50, seed=7)
        assert np.array_equal(a["good"], b["good"])

    def test_returns_one_replicate_per_draw_per_model(self):
        statuses, scores = self._data()
        reps = zs.bootstrap_macro(scores, statuses, [0, 1], n_boot=40, seed=1)
        assert set(reps) == {"good", "noise"}
        assert reps["good"].shape == (40,)
        assert reps["good"].mean() > reps["noise"].mean()

    def test_ci95_brackets_the_middle(self):
        lo, hi = zs.ci95(np.arange(1001, dtype=float))
        assert lo == pytest.approx(25.0) and hi == pytest.approx(975.0)


def _result(macro_auc=0.7, s=0.1, lic="mit", **over):
    base = {
        "device": "cpu", "errors": 0, "dim_image": 512, "dim_text": 512,
        "macro_auroc_strict": macro_auc, "forecast_25k_hours": s * 25000 / 3600,
        "s_per_image": s, "license": lic,
    }
    return base | over


class TestGate:
    def test_highest_macro_wins_when_the_gap_is_clear(self):
        winner, _ = zs.gate({"a": _result(0.70), "b": _result(0.75, s=0.5)})
        assert winner == "b"

    def test_a_near_tie_goes_to_the_cheaper_model(self):
        winner, reasons = zs.gate({"a": _result(0.750, s=0.5), "b": _result(0.735, s=0.1)})
        assert winner == "b"
        assert any("cheaper" in r for r in reasons)

    def test_a_near_tie_at_equal_cost_goes_to_the_freer_license(self):
        winner, _ = zs.gate({
            "a": _result(0.750, s=0.100, lic="cc-by-nc-4.0"),
            "b": _result(0.740, s=0.105, lic="mit"),
        })
        assert winner == "b"

    def test_a_full_tie_keeps_the_higher_macro(self):
        winner, _ = zs.gate({"a": _result(0.750, s=0.1), "b": _result(0.745, s=0.1)})
        assert winner == "a"

    def test_a_gap_of_exactly_the_margin_is_not_a_tie(self):
        assert 0.75 - 0.73 >= 0.02
        winner, reasons = zs.gate({"a": _result(0.75, s=0.5), "b": _result(0.73, s=0.1)})
        assert winner == "a"
        assert not any("within" in r for r in reasons)

    def test_a_gap_just_under_the_margin_is_a_tie_and_prints_precisely(self):
        winner, reasons = zs.gate({"a": _result(0.7499, s=0.5), "b": _result(0.73, s=0.1)})
        assert winner == "b"
        assert any("within 0.0199 AUROC" in r for r in reasons)

    def test_only_the_top_two_take_part_in_the_tie_break(self):
        winner, _ = zs.gate({
            "a": _result(0.80, s=0.5),
            "b": _result(0.79, s=0.4),
            "c": _result(0.785, s=0.01),
        })
        assert winner == "b"

    @pytest.mark.parametrize(
        "override, fragment",
        [
            ({"device": "cuda"}, "cpu"),
            ({"errors": 3}, "error"),
            ({"dim_text": 768}, "dim"),
            ({"macro_auroc_strict": 0.55}, "macro"),
            ({"macro_auroc_strict": None}, "macro"),
            ({"forecast_25k_hours": 8.5}, "25k"),
        ],
    )
    def test_each_hard_gate_rejects(self, override, fragment):
        winner, reasons = zs.gate({"a": _result(0.9, **override), "b": _result(0.6)})
        assert winner == "b"
        assert any(r.startswith("a: rejected") and fragment in r for r in reasons)

    def test_nobody_passing_returns_none(self):
        winner, reasons = zs.gate({"a": _result(0.5), "b": _result(0.9, errors=1)})
        assert winner is None
        assert reasons[-1] == "no candidate passed the hard gates"

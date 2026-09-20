"""Phase 1 gate: corpus decision from the measured join rate. See plan section 2.3."""

import pytest

from cxr_retrieval.data.audit import CorpusMode, decide_corpus_mode


def test_join_rate_at_or_above_95_percent_unlocks_full_corpus():
    assert decide_corpus_mode(0.97) is CorpusMode.FULL_FRONTAL


def test_join_rate_in_the_middle_band_falls_back_to_sampled_corpus():
    assert decide_corpus_mode(0.82) is CorpusMode.SAMPLED_25K


def test_join_rate_below_70_percent_aborts_instead_of_degrading():
    assert decide_corpus_mode(0.55) is CorpusMode.ABORT


@pytest.mark.parametrize(
    ("rate", "expected"),
    [
        (0.95, CorpusMode.FULL_FRONTAL),
        (0.9499, CorpusMode.SAMPLED_25K),
        (0.70, CorpusMode.SAMPLED_25K),
        (0.6999, CorpusMode.ABORT),
    ],
)
def test_thresholds_are_inclusive_at_their_lower_bound(rate, expected):
    assert decide_corpus_mode(rate) is expected


def test_rate_outside_zero_to_one_is_rejected():
    with pytest.raises(ValueError):
        decide_corpus_mode(1.5)


class TestImageKey:
    """Normalized key: patient/study/view, without prefix or file extension."""

    def test_extracts_key_from_chexpert_small_style_path(self):
        from cxr_retrieval.data.audit import image_key

        path = "CheXpert-v1.0-small/train/patient00001/study1/view1_frontal.jpg"
        assert image_key(path) == "patient00001/study1/view1_frontal"

    def test_same_key_despite_different_prefix_and_extension(self):
        from cxr_retrieval.data.audit import image_key

        small = "CheXpert-v1.0-small/train/patient00001/study1/view1_frontal.jpg"
        plus = "PNG/train/patient00001/study1/view1_frontal.png"
        assert image_key(small) == image_key(plus)

    def test_windows_separators_are_normalized(self):
        from cxr_retrieval.data.audit import image_key

        path = r"data\raw\CheXpert-v1.0-small\valid\patient64541\study1\view1_frontal.jpg"
        assert image_key(path) == "patient64541/study1/view1_frontal"

    def test_returns_none_when_path_has_no_patient_study_view_triple(self):
        from cxr_retrieval.data.audit import image_key

        assert image_key("labels/findings_fixed.json") is None


class TestEvaluateJoin:
    """Measure the join across normalization strategies, best one first."""

    SMALL = [
        "CheXpert-v1.0-small/train/patient00001/study1/view1_frontal.jpg",
        "CheXpert-v1.0-small/train/patient00002/study1/view1_frontal.jpg",
    ]

    def test_identical_paths_join_perfectly_on_the_exact_strategy(self):
        from cxr_retrieval.data.audit import evaluate_join

        best = evaluate_join(self.SMALL, self.SMALL)[0]
        assert best.strategy == "exact"
        assert best.join_rate == 1.0

    def test_different_prefix_and_extension_only_join_via_image_key(self):
        from cxr_retrieval.data.audit import evaluate_join

        plus = [
            "PNG/train/patient00001/study1/view1_frontal.png",
            "PNG/train/patient00002/study1/view1_frontal.png",
        ]
        reports = {r.strategy: r for r in evaluate_join(plus, self.SMALL)}
        assert reports["exact"].join_rate == 0.0
        assert reports["image_key"].join_rate == 1.0

    def test_best_strategy_is_returned_first(self):
        from cxr_retrieval.data.audit import evaluate_join

        plus = ["PNG/train/patient00001/study1/view1_frontal.png"]
        reports = evaluate_join(plus, self.SMALL)
        assert reports[0].strategy == "image_key"
        assert reports == sorted(reports, key=lambda r: r.join_rate, reverse=True)

    def test_join_rate_is_the_share_of_plus_rows_that_found_an_image(self):
        from cxr_retrieval.data.audit import evaluate_join

        plus = [
            "PNG/train/patient00001/study1/view1_frontal.png",
            "PNG/train/patient99999/study1/view1_frontal.png",
        ]
        best = evaluate_join(plus, self.SMALL)[0]
        assert best.plus_total == 2
        assert best.matched == 1
        assert best.join_rate == 0.5

    def test_unmatched_plus_paths_are_sampled_for_missing_analysis(self):
        from cxr_retrieval.data.audit import evaluate_join

        plus = ["PNG/train/patient99999/study1/view1_frontal.png"]
        best = evaluate_join(plus, self.SMALL)[0]
        assert "patient99999" in best.unmatched_samples[0]

    def test_sample_of_unmatched_paths_is_capped(self):
        from cxr_retrieval.data.audit import evaluate_join

        plus = [f"PNG/train/patient9{i:04d}/study1/view1_frontal.png" for i in range(50)]
        best = evaluate_join(plus, self.SMALL, sample_size=5)
        assert len(best[0].unmatched_samples) == 5

    def test_empty_plus_side_is_rejected_rather_than_dividing_by_zero(self):
        from cxr_retrieval.data.audit import evaluate_join

        with pytest.raises(ValueError):
            evaluate_join([], self.SMALL)

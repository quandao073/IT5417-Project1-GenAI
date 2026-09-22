"""Stratified, interleaved corpus sampling.

The manifest orders every eligible study once. `--limit` then takes a prefix, so
the ordering has to keep the label distribution stable at *every* prefix length,
not just at the full length.
"""

from __future__ import annotations

import polars as pl
import pytest

from src.data import sample


def resolved(rows):
    """A resolved label frame: (study_id, concept, status)."""
    return pl.DataFrame(
        rows,
        schema={"study_id": pl.Utf8, "concept": pl.Utf8, "status": pl.Utf8},
        orient="row",
    )


def strata(pairs):
    return pl.DataFrame(
        pairs, schema={"study_id": pl.Utf8, "stratum": pl.Utf8}, orient="row"
    )


class TestStratumPriority:
    def test_orders_concepts_from_rarest_to_commonest(self):
        df = resolved(
            [
                ("s1", "Edema", "PRESENT"),
                ("s2", "Edema", "PRESENT"),
                ("s3", "Edema", "PRESENT"),
                ("s1", "Fracture", "PRESENT"),
                ("s2", "Cardiomegaly", "PRESENT"),
                ("s3", "Cardiomegaly", "PRESENT"),
            ]
        )
        assert sample.stratum_priority(df) == ("Fracture", "Cardiomegaly", "Edema")

    def test_counts_only_present(self):
        df = resolved(
            [
                ("s1", "Edema", "PRESENT"),
                ("s2", "Fracture", "ABSENT"),
                ("s3", "Fracture", "UNCERTAIN"),
                ("s4", "Fracture", "NOT_MENTIONED"),
            ]
        )
        assert sample.stratum_priority(df) == ("Edema",)

    def test_is_deterministic_when_counts_tie(self):
        df = resolved([("s1", "Edema", "PRESENT"), ("s2", "Fracture", "PRESENT")])
        assert sample.stratum_priority(df) == sample.stratum_priority(df)


class TestAssignStrata:
    PRIORITY = ("Fracture", "Cardiomegaly", "Edema")

    def test_a_single_present_label_is_the_stratum(self):
        df = resolved([("s1", "Edema", "PRESENT")])
        assert sample.assign_strata(df, self.PRIORITY).row(0) == ("s1", "Edema")

    def test_the_rarest_present_label_wins(self):
        df = resolved(
            [
                ("s1", "Edema", "PRESENT"),
                ("s1", "Cardiomegaly", "PRESENT"),
                ("s1", "Fracture", "PRESENT"),
            ]
        )
        assert sample.assign_strata(df, self.PRIORITY).row(0) == ("s1", "Fracture")

    def test_a_study_with_no_present_label_falls_in_none(self):
        df = resolved(
            [
                ("s1", "Edema", "ABSENT"),
                ("s1", "Fracture", "UNCERTAIN"),
                ("s1", "Cardiomegaly", "NOT_MENTIONED"),
            ]
        )
        assert sample.assign_strata(df, self.PRIORITY).row(0) == ("s1", "none")

    def test_every_study_gets_exactly_one_stratum(self):
        df = resolved(
            [
                ("s1", "Edema", "PRESENT"),
                ("s1", "Cardiomegaly", "PRESENT"),
                ("s2", "Edema", "ABSENT"),
            ]
        )
        out = sample.assign_strata(df, self.PRIORITY)
        assert out.height == 2
        assert out.get_column("study_id").n_unique() == 2

    def test_a_concept_outside_the_priority_is_an_error(self):
        df = resolved([("s1", "Pneumonia", "PRESENT")])
        with pytest.raises(ValueError, match="Pneumonia"):
            sample.assign_strata(df, self.PRIORITY)


def pool(counts):
    """A stratum frame with `counts` = {stratum: how many studies}."""
    rows = [
        (f"{name}-{i:05d}", name) for name, n in counts.items() for i in range(n)
    ]
    return strata(rows)


class TestBuildManifest:
    def test_matches_the_sample_manifest_contract(self):
        from src.data import contracts

        out = sample.build_manifest(pool({"a": 5, "b": 5}), seed=20260101)
        contracts.validate(out, "sample_manifest")

    def test_orders_every_study_exactly_once(self):
        out = sample.build_manifest(pool({"a": 7, "b": 3}), seed=1)
        assert out.height == 10
        assert sorted(out.get_column("rank_in_manifest").to_list()) == list(range(10))

    def test_is_deterministic_under_the_same_seed(self):
        first = sample.build_manifest(pool({"a": 20, "b": 10}), seed=7)
        second = sample.build_manifest(pool({"a": 20, "b": 10}), seed=7)
        assert first.rows() == second.rows()

    def test_a_different_seed_reorders(self):
        first = sample.build_manifest(pool({"a": 20, "b": 10}), seed=7)
        second = sample.build_manifest(pool({"a": 20, "b": 10}), seed=8)
        assert first.rows() != second.rows()

    def test_records_the_seed_it_used(self):
        out = sample.build_manifest(pool({"a": 3}), seed=20260101)
        assert out.get_column("selection_seed").unique().to_list() == [20260101]

    def test_pinned_studies_take_the_head(self):
        out = sample.build_manifest(
            pool({"a": 50, "b": 50}), seed=1, pinned=["a-00000", "b-00000"]
        )
        head = out.sort("rank_in_manifest").head(2).get_column("study_id").to_list()
        assert set(head) == {"a-00000", "b-00000"}

    def test_a_pinned_study_outside_the_pool_is_an_error(self):
        with pytest.raises(ValueError, match="ghost"):
            sample.build_manifest(pool({"a": 3}), seed=1, pinned=["ghost"])

    @pytest.mark.parametrize("prefix", [100, 300, 600, 1000])
    def test_every_prefix_keeps_the_pool_proportions(self, prefix):
        """This is what lets --limit pick a corpus size without re-sampling."""
        counts = {"common": 600, "mid": 300, "rare": 100}
        out = sample.build_manifest(pool(counts), seed=20260101).sort("rank_in_manifest")

        seen = out.head(prefix).get_column("stratum").value_counts()
        got = dict(seen.iter_rows())
        for name, n in counts.items():
            expected = n / 1000 * prefix
            assert abs(got.get(name, 0) - expected) <= 0.03 * prefix, (
                f"{name} at prefix {prefix}: got {got.get(name, 0)}, expected ~{expected:.0f}"
            )

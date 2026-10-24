"""The one search path: parsed constraints -> template -> text encoder -> cosine -> evidence."""

from __future__ import annotations

import pytest

from src.common.config import load_config
from src.query.schema import Constraint, SearchQuery
from src.retrieval import pipeline
from tests.unit.test_api_state import _state
from tests.unit.test_api_state import world as _world_fixture

world = _world_fixture


def q(include=(), exclude=(), semantic="", views=()) -> SearchQuery:
    return SearchQuery(semantic_query=semantic, views=list(views), exclude=list(exclude),
                       include=[Constraint(concept=c, status=s) for c, s in include],
                       language="en")


@pytest.fixture(scope="module")
def concepts():
    return load_config("concepts")


def text_of(query, concepts):
    warnings: list[str] = []
    return pipeline.query_text(query, concepts, warnings), warnings


class TestQueryText:
    def test_constraints_become_the_template(self, concepts):
        assert text_of(q([("Edema", "PRESENT")], ["Pleural Effusion"]), concepts) == (
            "pulmonary edema and no pleural effusion", [])

    def test_no_concept_falls_back_to_the_free_text(self, concepts):
        assert text_of(q(semantic="hiatal hernia"), concepts) == (
            "hiatal hernia", [pipeline.NO_CONCEPT])

    def test_no_concept_and_no_free_text_searches_nothing(self, concepts):
        assert text_of(q(), concepts) == (None, [pipeline.NOTHING_TO_SEARCH])

    def test_only_negations_warn_that_the_model_tends_to_return_the_finding(self, concepts):
        text, warnings = text_of(q(exclude=["Pneumothorax", "Pleural Effusion"]), concepts)
        assert text == "no pneumothorax and no pleural effusion"
        assert warnings == [pipeline.ONLY_NEGATIONS.format("Pneumothorax, Pleural Effusion")]

    def test_a_negated_no_finding_is_dropped(self, concepts):
        assert text_of(q([("Edema", "PRESENT")], ["No Finding"]), concepts) == (
            "pulmonary edema", [pipeline.NO_FINDING_DROPPED])

    def test_a_query_left_without_a_concept_falls_back_to_the_free_text(self, concepts):
        assert text_of(q(exclude=["No Finding"], semantic="not a normal film"), concepts) == (
            "not a normal film", [pipeline.NO_FINDING_DROPPED, pipeline.NO_CONCEPT])

    def test_uncertain_is_searched_as_present(self, concepts):
        assert text_of(q([("Pneumonia", "UNCERTAIN")]), concepts) == (
            "pneumonia", [pipeline.UNCERTAIN_AS_PRESENT])

    def test_three_constraints_keep_every_one_in_canonical_order(self, concepts):
        text, warnings = text_of(
            q([("Edema", "PRESENT"), ("Cardiomegaly", "PRESENT")], ["Pleural Effusion"]),
            concepts)
        assert text == "cardiomegaly and pulmonary edema and no pleural effusion"
        assert warnings == [pipeline.MORE_THAN_TWO.format(3)]

    def test_two_constraints_need_no_warning(self, concepts):
        assert text_of(q([("Edema", "PRESENT"), ("Cardiomegaly", "PRESENT")]), concepts)[1] == []

    def test_no_finding_reads_normal_chest(self, concepts):
        assert text_of(q([("No Finding", "PRESENT")]), concepts) == ("normal chest", [])


class Spy:
    """Wraps the state's Qdrant client and records the query vectors it is asked for."""

    def __init__(self, client):
        self.client, self.calls = client, 0

    def __getattr__(self, name):
        return getattr(self.client, name)

    def query_points(self, *args, **kwargs):
        self.calls += 1
        return self.client.query_points(*args, **kwargs)


class TestRunSearch:
    def test_ranks_by_the_template_and_reports_what_the_encoder_read(self, world):
        state = _state(world)
        out = pipeline.run_search(state, q([("Edema", "PRESENT")], ["Pleural Effusion"]),
                                  top_k=3, views=[])
        assert out.text == "pulmonary edema and no pleural effusion"
        assert [h.rank for h in out.hits] == [1, 2, 3]
        assert all(isinstance(h.score, float) for h in out.hits)
        assert out.warnings == []

    def test_nothing_to_search_never_reaches_qdrant(self, world):
        state = _state(world)
        state.client = spy = Spy(state.client)
        out = pipeline.run_search(state, q(), top_k=3, views=[])
        assert out.hits == [] and out.text is None and spy.calls == 0
        assert pipeline.NOTHING_TO_SEARCH in out.warnings

    def test_evidence_is_quoted_after_ranking_and_never_moves_a_hit(self, world, tmp_path):
        query = q([("Edema", "PRESENT")], ["Pleural Effusion"])
        quoted = pipeline.run_search(_state(world), query, top_k=8, views=[])
        bare = pipeline.run_search(
            _state(world, settings=world[0].model_copy(
                update={"fts_db_path": tmp_path / "missing.sqlite"})),
            query, top_k=8, views=[])
        assert [h.study_id for h in quoted.hits] == [h.study_id for h in bare.hits]
        assert [h.score for h in quoted.hits] == [h.score for h in bare.hits]
        by_study = {h.study_id: [(e.concept, e.polarity) for e in h.evidence]
                    for h in quoted.hits}
        assert by_study["patient00000/study1"] == [("Edema", "affirmed"),
                                                   ("Pleural Effusion", "negated")]
        assert all(h.evidence == [] for h in bare.hits)
        assert bare.warnings == [pipeline.NO_EVIDENCE]

    def test_the_view_facet_filters_in_qdrant(self, world):
        state = _state(world)
        out = pipeline.run_search(state, q([("Edema", "PRESENT")]), top_k=50, views=["PA"])
        assert {state.images[h.image_id].projection for h in out.hits} == {"PA"}

    def test_a_dead_qdrant_is_a_search_backend_error(self, world):
        state = _state(world)

        class Dying:
            def query_points(self, *a, **k):
                raise ConnectionError("qdrant went away")

        state.client = Dying()
        with pytest.raises(pipeline.VectorLegError, match="qdrant went away"):
            pipeline.run_search(state, q([("Edema", "PRESENT")]), top_k=3, views=[])

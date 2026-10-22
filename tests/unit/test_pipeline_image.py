"""Mode `image` and the image leg of hybrid, against the four-study world."""

from __future__ import annotations

import polars as pl
import pytest

from src.data.labels import CHEXPERT_LABELS
from src.query.schema import Constraint, SearchQuery
from src.retrieval import pipeline
from src.retrieval.hybrid import FusionParams
from src.retrieval.image_concepts import build_image_scores
from tests.unit.test_api_state import _state
from tests.unit.test_api_state import world as _world_fixture

world = _world_fixture

P0, P1, P2, P3 = (f"patient0000{i}/study1" for i in range(4))   # views: PA, AP, AP, PA

IMAGE_ONLY = FusionParams(10, {"vector": 0.0, "report": 0.0, "image": 1.0}, 100, 100, 100)
ALL_LEGS = FusionParams(10, {"vector": 0.5, "report": 1.0, "image": 1.0}, 100, 100, 100)


def make_state(world, logits=None, *, probe=True):
    """The world's state with a probe whose logits are zero except those in `logits`."""
    state = _state(world)
    if probe:
        ids = state.labels.study_ids.tolist()
        columns = {c: [0.0] * len(ids) for c in CHEXPERT_LABELS}
        for (study, concept), z in (logits or {}).items():
            columns[concept][ids.index(study)] = z
        state.image_scores = build_image_scores(pl.DataFrame({"study_id": ids, **columns}), ids)
        state.image_status = "ok"
    else:   # the world already ships a probe; take it away
        state.image_scores, state.image_status = None, "disabled"
    return state


def edema_no_effusion(status="PRESENT"):
    return SearchQuery(semantic_query="edema without pleural effusion", language="en",
                       include=[Constraint(concept="Edema", status=status)],
                       exclude=["Pleural Effusion"])


def run(state, query, mode, *, views=(), uncertainty="EXCLUDE", fusion=None, top_k=10):
    return pipeline.run_search(state, query, mode, top_k=top_k, views=views,
                               absence_policy="exclude_present", uncertainty_policy=uncertainty,
                               fusion=fusion)


LOGITS = {(P0, "Edema"): 4.0, (P1, "Edema"): 4.0, (P2, "Edema"): 4.0, (P1, "Pleural Effusion"): 4.0}


def test_image_mode_puts_the_excluded_finding_last(world):
    out = run(make_state(world, LOGITS), edema_no_effusion(), "image")
    assert [h.study_id for h in out.hits] == [P0, P2, P3, P1]
    assert [h.rank for h in out.hits] == [1, 2, 3, 4]
    assert [h.image_rank for h in out.hits] == [1, 2, 3, 4]
    assert out.total_matches is None and out.warnings == []
    best, worst = out.hits[0], out.hits[-1]
    assert best.score > worst.score and best.vector_rank is None and best.rrf_score is None
    assert best.image_id == world_image(world, P0)
    # informational flag from the labels, never a filter: P1 has the effusion
    assert best.constraints_satisfied is True and worst.constraints_satisfied is False
    assert [label["concept"] for label in best.labels] == ["Edema", "Pleural Effusion"]


def world_image(world, study):
    return _state(world).study_image[study]


@pytest.mark.parametrize("mode", ["image", "hybrid"])
def test_uncertain_include_warns(world, mode):
    state = make_state(world, LOGITS)
    warning = "image score treats UNCERTAIN as PRESENT"
    out = run(state, edema_no_effusion("UNCERTAIN"), mode, uncertainty="INCLUDE",
              fusion=IMAGE_ONLY)
    assert warning in out.warnings
    assert warning not in run(state, edema_no_effusion(), mode, uncertainty="INCLUDE",
                              fusion=IMAGE_ONLY).warnings


def test_image_mode_without_a_concept_falls_back_to_cosine_in_english(world):
    state = make_state(world)
    query = SearchQuery(semantic_query="chest x-ray", language="en")
    out = run(state, query, "image")
    assert out.warnings == ["no concept; ranked by text-image cosine"]
    vector = run(state, query, "vector")
    assert [h.image_id for h in out.hits] == [h.image_id for h in vector.hits] and out.hits
    assert out.hits[0].constraints_satisfied is None and out.total_matches is None


def test_image_mode_without_a_concept_in_vietnamese_returns_nothing(world):
    out = run(make_state(world), SearchQuery(semantic_query="", language="vi"), "image")
    assert out.hits == [] and out.warnings == ["semantic query is empty; vector leg skipped"]


def test_image_mode_without_the_probe_is_empty_with_a_warning_not_an_error(world):
    state = make_state(world, probe=False)
    assert state.image_status == "disabled"
    out = run(state, edema_no_effusion(), "image")
    assert out.hits == [] and out.warnings == ["image probe unavailable"]


def test_image_mode_applies_the_view_filter(world):
    state = make_state(world, LOGITS)
    out = run(state, edema_no_effusion(), "image", views=["PA"])
    assert [h.study_id for h in out.hits] == [P0, P3]
    assert [h.image_rank for h in out.hits] == [1, 2]
    assert [h.study_id for h in run(state, edema_no_effusion(), "image", views=["AP"]).hits] == [
        P2, P1]


def test_image_mode_honours_top_k(world):
    out = run(make_state(world, LOGITS), edema_no_effusion(), "image", top_k=2)
    assert [h.study_id for h in out.hits] == [P0, P2]


def test_hybrid_with_the_probe_reports_image_rank_and_without_it_warns(world):
    logits = {(P0, "Edema"): 2.0, (P2, "Edema"): 6.0}
    query = edema_no_effusion()
    with_probe = run(make_state(world, logits), query, "hybrid", uncertainty="INCLUDE",
                     fusion=IMAGE_ONLY)
    # P0 and P2 pass the label filter; the image leg alone ranks P2 (stronger edema) first
    assert [(h.study_id, h.image_rank) for h in with_probe.hits] == [(P2, 1), (P0, 2)]
    assert all(h.vector_rank is None and h.report_rank is None for h in with_probe.hits)
    assert with_probe.total_matches == 2 and with_probe.warnings == []

    without = run(make_state(world, probe=False), query, "hybrid", uncertainty="INCLUDE",
                  fusion=ALL_LEGS)
    assert "image probe unavailable; hybrid ran without the image leg" in without.warnings
    assert without.hits and all(h.image_rank is None for h in without.hits)


def test_hybrid_with_the_default_image_weight_stays_silent_and_unranked(world):
    # the shipped weight is 0.0: the probe is neither called nor mentioned, with or without it
    for probe in (True, False):
        out = run(make_state(world, LOGITS, probe=probe), edema_no_effusion(), "hybrid")
        assert [h.study_id for h in out.hits] == [P0]
        assert out.hits[0].image_rank is None and out.warnings == []


def test_hybrid_without_constraints_has_no_image_leg(world):
    state = make_state(world, LOGITS)
    query = SearchQuery(semantic_query="chest x-ray", language="en")
    out = run(state, query, "hybrid", fusion=ALL_LEGS)
    assert out.hits and all(h.image_rank is None for h in out.hits)
    assert not any("image" in w for w in out.warnings)

"""Layer 4 with vs without knowledge: blind order, outcome sign, tally."""

from src.evaluation import kg_explain_pairs as kp


def test_the_order_is_seeded_per_question_and_mixes():
    assert kp.swapped("q1", 7) == kp.swapped("q1", 7)
    assert {kp.swapped(f"q{i}", 7) for i in range(40)} == {True, False}


def test_the_outcome_follows_the_side_of_the_explanation_with_knowledge():
    answer = {"faithfulness": {"winner": "B", "reason": ""},
              "usefulness": {"winner": "tie", "reason": ""}}
    assert kp.outcome(answer, "faithfulness", knowledge_is_b=True) == 1
    assert kp.outcome(answer, "faithfulness", knowledge_is_b=False) == -1
    assert kp.outcome(answer, "usefulness", knowledge_is_b=True) == 0


def test_tally_counts_and_bounds_the_mean():
    t = kp.tally([1, 1, 0, -1], n_boot=500, seed=1)
    assert (t["wins"], t["losses"], t["ties"], t["n"]) == (2, 1, 1, 4)
    assert t["mean"] == 0.25 and t["ci95"][0] <= 0.25 <= t["ci95"][1]
    assert t["win_share"] == 0.625


def test_render_shows_what_the_user_reads():
    text = kp.render({"summary": "S.", "images": [{"rank": 1, "line": "L."}],
                      "knowledge": "K [K1]."})
    assert text == "S.\n#1: L.\n\nK [K1]."

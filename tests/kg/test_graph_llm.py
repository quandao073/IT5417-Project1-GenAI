import json

import pytest

from src.kg import graph_llm, llm_select


def test_the_alias_check_names_every_proposed_alias_exactly_once():
    rows = [{"cui": "C1", "alias": "xơ phổi"}, {"cui": "C1", "alias": "xơ hóa phổi"},
            {"cui": "C2", "alias": "tim to"}]
    items = graph_llm.alias_check_items(rows, {"C1": ("Lung scarring", "T047")}, {"C1": ["a"]})
    assert [it["string"] for it in items] == ["C1", "C2"]
    assert items[0]["input"]["proposed_vietnamese_names"] == ["xơ hóa phổi", "xơ phổi"]
    assert "reason" not in items[0]["input"]           # independent of the proposer
    good = {"verdicts": [{"alias": "Xơ phổi", "verdict": "agree", "reason": "r"},
                         {"alias": "xơ hóa phổi", "verdict": "disagree", "reason": "r"}]}
    assert len(graph_llm.parse_alias_check(json.dumps(good), items[0])["verdicts"]) == 2
    with pytest.raises(llm_select.LLMSelectError):
        graph_llm.parse_alias_check(json.dumps({"verdicts": good["verdicts"][:1]}), items[0])

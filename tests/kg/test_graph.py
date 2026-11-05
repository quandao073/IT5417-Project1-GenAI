"""V2-4 graph tables (D57, D58) on invented UMLS rows, links and mentions."""

from __future__ import annotations

import polars as pl

from src.common.config import load_config
from src.kg import graph, graph_llm

LCFG = load_config("kg")["linker"]
GCFG = {**load_config("kg")["graph"], "anatomy_depth": 1, "finding_down": 1, "finding_up": 2,
        "stop_up": ["C0000090"], "min_n": 2}


def conso(cui, text):
    return {"CUI": cui, "STR": text, "SAB": "MSH", "TS": "P", "ISPREF": "Y", "STT": "PF"}


STY = {"C0000001": "T047", "C0000002": "T047", "C0000003": "T047", "C0000090": "T047",
       "C0000004": "T047", "C0000010": "T023", "C0000011": "T023", "C0000012": "T023",
       "C0000013": "T023", "C0000020": "T074", "C0000021": "T074", "C0025080": "T074"}
NAMES = {"C0000001": "Pneumonia", "C0000002": "Lobar pneumonia", "C0000003": "Lung disease",
         "C0000090": "Disease", "C0000004": "Lung infection", "C0000010": "Lung",
         "C0000011": "Lower lobe", "C0000012": "Left lower lobe", "C0000013": "Thorax",
         "C0000020": "Chest tube", "C0000021": "Bone screw", "C0025080": "Medical device"}
REL = [("C0000003", "CHD", "C0000001", "MSH", None), ("C0000090", "CHD", "C0000003", "MSH", None),
       ("C0000001", "CHD", "C0000002", "MSH", None), ("C0000004", "CHD", "C0000002", "MSH", None),
       ("C0000010", "RO", "C0000011", "FMA", "regional_part_of"),
       ("C0000011", "CHD", "C0000012", "FMA", "isa"),
       ("C0000013", "RO", "C0000010", "FMA", "regional_part_of")]


def umls(tmp_path):
    out = tmp_path / "umls"
    out.mkdir()
    pl.DataFrame([conso(c, n) for c, n in NAMES.items()]).write_parquet(out / "mrconso.parquet")
    pl.DataFrame([{"CUI": c, "TUI": t, "STY": t, "STN": "A1.2" if t == "T023" else "B2.2"}
                  for c, t in STY.items()]).write_parquet(out / "mrsty.parquet")
    pl.DataFrame(REL, schema=["CUI1", "REL", "CUI2", "SAB", "RELA"], orient="row").write_parquet(
        out / "mrrel.parquet")
    return out


def domain():
    return pl.DataFrame([(c, "Anatomy" if t == "T023" else "Observation") for c, t in STY.items()],
                        schema=["CUI", "kind"], orient="row")


def test_growth_stays_in_depth_stops_at_the_stop_list_and_drops_sided(tmp_path):
    core = pl.DataFrame({"cui": ["C0000001", "C0000010"], "kind": ["Observation", "Anatomy"]})
    grown = graph.expand(core, umls(tmp_path), LCFG, GCFG, domain())
    got = {c: k for c, k in grown.filter(~pl.col("in_core")).select("cui", "kind").iter_rows()}
    # down 1 (lobar pneumonia), up 2 but "Disease" is a stop; anatomy 1 step each way, the
    # sided lobe two steps down is out of reach and would be dropped anyway
    assert got == {"C0000002": "Observation", "C0000003": "Observation",
                   "C0000011": "Anatomy", "C0000013": "Anatomy"}


def test_devices_follow_the_labeler_phrases_inside_their_text(tmp_path):
    links = pl.DataFrame({"kind": ["Observation"] * 2, "string": ["left chest tube", "screw"],
                          "cui": ["C0000020", "C0000021"]})
    nodes = pl.DataFrame({"cui": ["C0000020", "C0000021", "C0025080"], "in_core": [True] * 3})
    groups = graph.device_groups(links, nodes, umls(tmp_path), LCFG, ["chest tube", "tube"],
                                 "C0025080", [])
    assert groups == {"support_devices": ["C0000020"], "other": ["C0000021"]}   # parent left out


def test_weights_count_reports_with_a_present_source_and_pairs_win(tmp_path):
    mentions = pl.DataFrame(
        [(f"s{i}", "impression", "1", "Anatomy", None, "lower lobe", [], [], False)
         for i in range(3)]
        + [(f"s{i}", "impression", "2", "Observation", "present", "consolidation", ["1"], ["3"],
            False) for i in range(3)]
        + [(f"s{i}", "impression", "3", "Observation", "uncertain", "pneumonia", [], [], False)
           for i in range(3)]
        + [("s9", "impression", "2", "Observation", "absent", "consolidation", [], ["3"], False),
           ("s9", "impression", "3", "Observation", "present", "pneumonia", [], [], False)],
        schema=["study_id", "section", "entity", "kind", "polarity", "joined", "located_at",
                "suggests", "dropped_head"], orient="row")
    links = pl.DataFrame({"kind": ["Anatomy", "Observation", "Observation"],
                          "string": ["lower lobe", "consolidation", "pneumonia"],
                          "cui": ["C0000011", "C0000005", "C0000001"]})
    pairs = pl.DataFrame({"obs": ["consolidation"], "anat": ["lower lobe"], "cui": ["C0000006"]})
    mc = graph.mention_cuis(mentions, links, pairs)
    assert set(mc.filter(pl.col("entity") == "2")["cui"]) == {"C0000006", "C0000005"}  # pair, s9
    kept, every = graph.weighted(mc, "suggests", "Observation", graph.SUGGESTS, 2)
    # the uncertain target counts; the absent source in s9 does not
    assert kept.select("src", "dst", "n", "n_src", "p").rows() == [
        ("C0000006", "C0000001", 3, 3, 1.0)]
    at, _ = graph.weighted(mc, "located_at", "Anatomy", graph.TYPICALLY_AT, 4)
    assert at.is_empty()                                  # n = 3 is under the cut
    assert graph.n_pool(mc).filter(pl.col("cui") == "C0000001")["n_pool"].item() == 4


def test_chexpert_edges_and_scorable_reach():
    mapping = {"Pneumonia": {"id": "C0000001"}, "Lung Opacity": {"id": "C0000099"},
               "Consolidation": {"id": "C0000005"},
               "Support Devices": {"id": "PRJ:support_devices", "is_a": "C0025080",
                                   "children": ["C0000020"]}}
    cfg = {**GCFG, "chexpert_hierarchy": {"Pneumonia": "Consolidation",
                                          "Consolidation": "Lung Opacity"},
           "chexpert_children": {}, "generic_is_a": {"Lung Opacity": ["infiltrate", "haze"]},
           "devices_parent": "C0025080", "phrase_is_a": {}}
    links = pl.DataFrame({"kind": ["Observation"], "string": ["infiltrate"], "cui": ["C0000007"]})
    edges, missing = graph.chexpert_edges(mapping, cfg, links,
                                          {"support_devices": [], "other": ["C0000021"]})
    assert missing == ["haze"]
    assert set(edges.select("src", "dst").rows()) == {
        ("PRJ:support_devices", "C0025080"), ("C0000020", "PRJ:support_devices"),
        ("C0000001", "C0000005"), ("C0000005", "C0000099"), ("C0000007", "C0000099"),
        ("C0000021", "C0025080")}
    nodes = pl.DataFrame({"cui": ["C0000001", "C0000002", "C0000003", "C0000010", "C0000011",
                                  "C0000012"],
                          "scorable": [True, False, False, False, False, False]})
    more = pl.DataFrame({"src": ["C0000002", "C0000001", "C0000003", "C0000011", "C0000001"],
                         "rel": ["IS_A", "IS_A", "IS_A", "PART_OF", "TYPICALLY_AT"],
                         "dst": ["C0000001", "C0000003", "C0000090", "C0000010", "C0000011"],
                         "review": [None] * 5}, schema_overrides={"review": pl.String})
    reach = dict(graph.scorable_reach(nodes, more).iter_rows())
    # D59: below a reaching concept, not above it; a TYPICALLY_AT target and the whole it is
    # part of; nothing else
    assert reach == {"C0000001": True, "C0000002": True, "C0000003": False,
                     "C0000010": True, "C0000011": True, "C0000012": False}
    dropped = pl.DataFrame({"src": ["C0000001", "C0000001"], "rel": ["SUGGESTS"] * 2,
                            "dst": ["C0000003", "C0000012"], "review": [None, None]})
    review = pl.DataFrame({"src": ["C0000001"], "dst": ["C0000012"], "decision": ["drop"]})
    reviewed = graph.apply_review(dropped, review)
    assert reviewed["review"].to_list() == [None, "drop"]
    reach = dict(graph.scorable_reach(nodes, reviewed).iter_rows())
    assert reach["C0000003"] and not reach["C0000012"]     # a dropped SUGGESTS is not used


def test_alias_flags_and_suggest_items():
    flags = graph_llm.alias_flags(
        [("C1", "phù"), ("C1", "phù phổi"), ("C2", "phù phổi"), ("C3", "tim to")],
        blocked=["phù hợp"], existing_vi={"Cardiomegaly": ["tim to"]}, label_of={})
    assert flags == ["inside_blocked;single_syllable", "shared", "shared", "shared"]
    edges = pl.DataFrame({"src": ["A", "B"], "rel": ["SUGGESTS"] * 2, "dst": ["D", "D"],
                          "n": [5, 9], "n_src": [50, 90], "p": [0.1, 0.1]})
    items = graph_llm.suggest_items(edges, {"A": ("a", ""), "D": ("d", "")}, top=1)
    assert [i["string"] for i in items] == ["B>D"] and items[0]["input"]["source"]["name"] == "B"


def test_aliases_keep_only_reviewed_vietnamese_forms(tmp_path):
    nodes = pl.DataFrame({"cui": ["C0000001", "PRJ:support_devices"]})
    review = pl.DataFrame({"cui": ["C0000001", "C0000001"], "alias": ["viêm phổi", "phổi"],
                           "decision": ["keep", "drop"]})
    concepts = {"Pneumonia": {"en": ["pneumonia"], "vi": ["viêm phổi"]}}
    got = graph.aliases(nodes, umls(tmp_path), concepts, {"Pneumonia": {"id": "C0000001"}}, review,
                        {"vi": {"C0000001": ["phổi viêm"]}})
    assert ("phổi viêm", "vi", "manual") in set(got.select("text", "lang", "source").rows())
    got = got.filter(pl.col("source") != "manual")
    assert set(got.select("text", "lang", "source").rows()) == {
        ("Pneumonia", "en", "mrconso"), ("viêm phổi", "vi", "concepts_yaml")}  # llm form, same norm
    assert "phổi" not in got["text"].to_list()


def test_manual_edges_carry_the_source_and_equal_weights():
    cfg = {"manual_edges": {"is_a": [["C1", "C2"]], "suggests": {"C9": ["C3", "C4", "C5"]}}}
    edges = graph.manual_edges(cfg)
    assert edges["source"].unique().to_list() == ["manual"]
    assert edges.filter(pl.col("rel") == "IS_A").select("src", "dst").rows() == [("C1", "C2")]
    s = edges.filter(pl.col("rel") == "SUGGESTS")
    assert s["dst"].unique().to_list() == ["C9"] and s["p"].to_list() == [0.3333] * 3
    assert s["n"].null_count() == 3 and graph.manual_edges({}).height == 0

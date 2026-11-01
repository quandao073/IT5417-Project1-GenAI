"""`cxr kg-link` (TIP V2-3, D56) on invented MRCONSO, MRSTY, MRREL rows and a fake SapBERT."""

from __future__ import annotations

import json

import numpy as np
import polars as pl
import pytest

from src.cli.main import main
from src.common.config import default_config_dir, load_config
from src.kg import linker as lk
from src.kg import vocab
from tests.kg.test_vocab import world  # noqa: F401  (the pool, Plus CSV and RadGraph fixture)

REAL = load_config("kg")["linker"]
CFG = {**REAL, "domain": {"code_sabs": ["LNC"], "relas": REAL["domain"]["relas"],
                          "anatomy": {"positive": ["C1000020"], "negative": []},
                          "observation": {"positive": ["C1000021"], "negative": ["C1000022"]}},
       "candidates": {"top_k": 3, "floor": 0.1}}


def conso(cui, text, sab, preferred=False):
    return {"CUI": cui, "STR": text, "SAB": sab, "TS": "P" if preferred else "S",
            "ISPREF": "Y" if preferred else "N", "STT": "PF" if preferred else "VO"}


ROWS = [
    conso("C1000001", "Pleural effusion", "MSH", True), conso("C1000002", "Effusion", "NCI", True),
    conso("C1000003", "Pneumothorax", "MSH", True), conso("C1000003", "Collapsed lung", "MSH"),
    conso("C1000004", "Thickening", "NCI", True),
    conso("C1000005", "Pleural thickening", "MSH", True),
    conso("C1000006", "Pleura", "FMA", True), conso("C1000007", "Lung", "FMA", True),
    conso("C1000008", "Mass", "NCI", True), conso("C1000009", "Mass", "HPO"),
    conso("C1000009", "Lump", "MSH"), conso("C1000011", "Sternotomy", "MSH", True),
    conso("C1000012", "Left lower lobe", "FMA", True), conso("C1000013", "Lower lobe", "FMA", True),
    conso("C1000014", "Thickening pleura", "NCI", True),
    conso("C1000016", "Crowding", "NCI", True),              # below the negative root only
    conso("C1000017", "Hilum", "LNC", True),                 # named by a code system only
    conso("C1000020", "Thorax", "FMA", True), conso("C1000021", "Thoracic disease", "MSH", True),
    conso("C1000022", "Tooth disease", "MSH", True),
]
STY = {"C1000001": "T046", "C1000002": "T046", "C1000003": "T047", "C1000004": "T033",
       "C1000005": "T047", "C1000006": "T023", "C1000007": "T023", "C1000008": "T033",
       "C1000009": "T033", "C1000011": "T061", "C1000012": "T023", "C1000013": "T023",
       "C1000014": "T033", "C1000016": "T033", "C1000017": "T029", "C1000020": "T029",
       "C1000021": "T047", "C1000022": "T047"}


def rel(whole, rela, part, sab="FMA", kind="RO"):
    return {"CUI1": whole, "REL": kind, "CUI2": part, "SAB": sab, "RELA": rela}


REL = [rel("C1000012", "inverse_isa", "C1000013", kind="PAR"),
       rel("C1000013", "isa", "C1000012", kind="CHD"),
       rel("C1000020", "regional_part_of", "C1000007"), rel("C1000020", "regional_part_of",
                                                            "C1000006"),
       rel("C1000007", "constitutional_part_of", "C1000013"),
       rel("C1000020", "regional_part_of", "C1000017"),
       *[rel("C1000021", None, c, sab="MSH", kind="CHD") for c in
         ("C1000001", "C1000002", "C1000003", "C1000005", "C1000008", "C1000014")],
       rel("C1000008", "isa", "C1000009", sab="NCI", kind="CHD"),   # below a root, two steps down
       *[rel("C1000022", None, c, sab="MSH", kind="CHD") for c in ("C1000016", "C1000009")]]
OVERRIDES = {"Observation": {"effusion": "C1000001", "tip": "skip", "collapsed lung": "C9999999"},
             "Anatomy": {}}


@pytest.fixture
def umls(tmp_path):
    out = tmp_path / "umls"
    out.mkdir()
    pl.DataFrame(ROWS).write_parquet(out / "mrconso.parquet")
    pl.DataFrame([{"CUI": c, "TUI": t, "STY": t} for c, t in STY.items()]).write_parquet(
        out / "mrsty.parquet")
    pl.DataFrame(REL).write_parquet(out / "mrrel.parquet")
    return out


class FakeSap:
    """Dictionary names are one-hot; a query is a mix, so its nearest name is known."""

    revision, max_length = "fakerevision0000", 25
    NAMES = sorted({vocab.normalize(r["STR"]) for r in ROWS})
    MIXES = {"airspace process": {"effusion": 0.9, "lung": 0.2},
             "hilum": {"effusion": 0.9, "lung": 0.3}}   # Anatomy: "effusion" is not allowed

    def __init__(self, cfg=None):
        self.calls = 0

    def encode(self, texts):
        self.calls += 1
        out = np.zeros((len(texts), len(self.NAMES) + 1), dtype=np.float32)
        for row, text in enumerate(texts):
            mix = self.MIXES.get(text, {text: 1.0} if text in self.NAMES else {})
            for name, w in mix.items():
                out[row, self.NAMES.index(name)] = w
            out[row, -1] = 0.0 if mix else 1.0
        return out / np.linalg.norm(out, axis=1, keepdims=True)


def heads(*rows):
    return pl.DataFrame([(k, i + 1, s, 10, 5, "80") for i, (k, s) in enumerate(rows)],
                        schema={"kind": pl.String, "rank": pl.UInt32, "string": pl.String,
                                "mentions": pl.UInt32, "studies": pl.UInt32, "band": pl.String},
                        orient="row")


def link(umls, tmp_path, *rows, overrides=OVERRIDES):
    found, cands, outside = lk.link_heads(heads(*rows), umls, CFG, overrides,
                                          lk.domain(umls, CFG), FakeSap, tmp_path / "sapbert")
    by_string = {(k, s): r for k, s, r in zip(found["kind"], found["string"],
                                               found.iter_rows(named=True), strict=True)}
    return by_string, cands, outside


def candidate_cuis(cands, string):
    return cands.filter(pl.col("string") == string).sort("rank")["cui"].to_list()


def test_rules_fire_in_order(umls, tmp_path):
    got, _, _ = link(umls, tmp_path, *[("Observation", s) for s in
                                       ("normal", "tip", "effusion", "collapsed lung",
                                        "thickening", "pneumothorax")])
    methods = {s: (r["method"], r["rule_cui"], r["route"]) for (_, s), r in got.items()}
    assert methods == {
        "normal": ("normal", None, None), "tip": ("skip", None, None),
        "effusion": ("override", "C1000001", None),          # over the exact C1000002
        "collapsed lung": ("override", "C9999999", None),    # over the exact Pneumothorax CUI
        "thickening": ("descriptive", None, None),           # alone, though an exact name exists
        "pneumothorax": ("select", None, "exact")}           # the LLMs choose (D56)
    assert got[("Observation", "pneumothorax")]["argmax_cui"] == "C1000003"


def test_exact_candidates_come_first_ranked_by_preferred_name(umls, tmp_path):
    got, cands, outside = link(umls, tmp_path, ("Observation", "mass"),
                               ("Anatomy", "pneumothorax"), ("Observation", "sternotomy"))
    assert candidate_cuis(cands, "mass")[:2] == ["C1000008", "C1000009"]   # preferred first
    assert cands.filter(pl.col("string") == "mass", pl.col("rank") <= 2)["exact"].all()
    assert got[("Anatomy", "pneumothorax")]["route"] == "sapbert"   # T047 is no Anatomy type
    assert outside == {("Observation", "sternotomy")}     # a procedure (T061) only
    assert (got[("Observation", "sternotomy")]["method"],
            got[("Observation", "sternotomy")]["route"]) == ("out_of_scope", None)
    hand, _, _ = link(umls, tmp_path, ("Observation", "sternotomy"),
                      overrides={"Observation": {"sternotomy": "skip"}})
    assert hand[("Observation", "sternotomy")]["method"] == "skip"   # an override still wins


def test_the_domain_drops_code_system_names_and_cuis_below_a_negative_root(umls, tmp_path):
    dom = set(lk.domain(umls, CFG).iter_rows())
    assert {("C1000007", "Anatomy"), ("C1000012", "Anatomy"),       # thorax > lung > lobe > left
            ("C1000009", "Observation")} <= dom                     # below the positive root too
    assert ("C1000017", "Anatomy") not in dom                       # LNC names only
    assert ("C1000016", "Observation") not in dom                   # below the negative root only
    assert ("C1000022", "Observation") not in dom                   # the negative root itself
    assert ("C1000004", "Observation") in dom                       # below no root at all
    got, cands, outside = link(umls, tmp_path, ("Observation", "crowding"), ("Anatomy", "hilum"))
    # on the label table, so not out_of_scope; out of the domain, so no exact candidate
    assert got[("Observation", "crowding")]["route"] == "sapbert" and not outside
    assert "C1000016" not in candidate_cuis(cands, "crowding")
    assert candidate_cuis(cands, "hilum") == ["C1000007"]           # FakeSap's mix, not the LNC CUI
    sizes = lk.domain_sizes(umls, CFG, lk.domain(umls, CFG))
    assert sizes["Anatomy"] == {"label_table": 6, "domain": 5}


def test_adjective_goes_through_its_noun(umls, tmp_path):
    got, cands, _ = link(umls, tmp_path, ("Anatomy", "pleural"))
    pleural = got[("Anatomy", "pleural")]
    assert (pleural["route"], pleural["query"], pleural["argmax_cui"]) == (
        "adjective", "pleura", "C1000006")


def test_sapbert_gathers_the_nearest_names_allowed_for_the_kind(umls, tmp_path):
    got, cands, _ = link(umls, tmp_path, ("Observation", "airspace process"),
                         ("Anatomy", "hilum"))
    obs, anat = got[("Observation", "airspace process")], got[("Anatomy", "hilum")]
    assert (obs["route"], obs["argmax_cui"]) == ("sapbert", "C1000002")
    assert anat["argmax_cui"] == "C1000007"                         # "effusion" is no Anatomy
    assert 0 < anat["argmax_cosine"] < obs["argmax_cosine"] < 1
    assert not cands["exact"].any()
    cache = tmp_path / "sapbert" / FakeSap.revision[:12]
    assert (cache / "vectors.npy").is_file() and (cache / "meta.json").is_file()


def choice(decision, cui="", reason="r"):
    return {"decision": decision, "cui": cui, "reason": reason}


def decided(umls, tmp_path, rows, choices, adjudication=None):
    found, _, _ = lk.link_heads(heads(*rows), umls, CFG, OVERRIDES, lk.domain(umls, CFG),
                                FakeSap, tmp_path / "sapbert")
    final = lk.decide(found, choices, adjudication or {}, umls, CFG, lk.domain(umls, CFG))
    return {s: r for s, r in zip(final["string"], final.iter_rows(named=True), strict=True)}


def test_the_llms_must_agree_and_an_adjudication_wins(umls, tmp_path):
    rows = [("Observation", s) for s in ("mass", "pneumothorax", "airspace process", "nodule",
                                         "normal", "effusion")]
    a = {("Observation", "mass"): choice("cui", "C1000008"),
         ("Observation", "pneumothorax"): choice("cui", "C1000003"),
         ("Observation", "airspace process"): choice("skip"),
         ("Observation", "nodule"): choice("rejected", "C9999998")}
    b = {**a, ("Observation", "pneumothorax"): choice("none"),
         ("Observation", "nodule"): choice("rejected", "C9999998")}
    del b[("Observation", "mass")]
    got = decided(umls, tmp_path, rows, {"opus": a, "sonnet": b},
                  {("Observation", "effusion"): choice("cui", "C1000002", "adjudicated")})
    view = {s: (r["method"], r["decision"], r["cui"], r["agree"]) for s, r in got.items()}
    assert view == {
        "mass": ("pending", "pending", None, None),            # only one provider answered
        "pneumothorax": ("llm_disagree", "none", None, False),
        "airspace process": ("llm", "skip", None, True),
        "nodule": ("llm_disagree", "none", None, False),       # two rejections never agree
        "normal": ("normal", "normal", None, None),
        "effusion": ("adjudicated", "cui", "C1000002", None)}  # over the override
    assert got["pneumothorax"]["llm_opus"] == "cui C1000003: r"
    assert got["pneumothorax"]["llm_sonnet"] == "none: r"
    cov = lk.coverage(pl.DataFrame({"kind": ["Observation"] * 3, "mentions": [6, 3, 1],
                                    "decision": ["cui", "skip", "none"],
                                    "cui": ["C1", None, None]}))
    assert cov["all"] == {"decided": 0.9, "linked": 0.6}


def test_a_lateral_cui_moves_to_its_unlateralised_parent(umls, tmp_path):
    rows = [("Anatomy", "left lower lobe"), ("Anatomy", "lung")]
    both = {("Anatomy", "left lower lobe"): choice("cui", "C1000012"),
            ("Anatomy", "lung"): choice("cui", "C1000007")}
    got = decided(umls, tmp_path, rows, {"opus": both, "sonnet": both})
    lobe = got["left lower lobe"]
    assert (lobe["cui"], lobe["lateral_from"], lobe["lateral_unresolved"]) == (
        "C1000013", "C1000012", False)
    assert got["lung"]["lateral_from"] is None


def test_flags_point_at_what_needs_a_look():
    links = pl.DataFrame({
        "kind": ["Observation", "Anatomy", "Observation"], "string": ["a", "size", "c"],
        "cui": ["C0000001", "C0000002", None],
        "llm_opus": ["cui C0000001: r", "cui C0000002: r", "cui C0000003: r"],
        "lateral_unresolved": [False, False, True]})
    info = pl.DataFrame({"cui": ["C0000001", "C0000002", "C0000003"],
                         "name": ["Crowding of teeth", "Size", "Thing"],
                         "sabs": ["MSH", "CHV,MTH", "NCI"], "tuis": ["T033", "T081", "T023"]})
    got = lk.flags(links, info, CFG, skipped={"size"}, mismatched={("Observation", "a")})
    assert got == ["chexpert_mismatch;off_domain_word", "skip_listed;thin_sources;type_mismatch",
                   "lateral_unresolved;type_mismatch"]


def test_adjudication_rows_are_checked(tmp_path):
    path = tmp_path / "adjudication.csv"
    assert lk.load_adjudication(path) == {}
    path.write_text("string,kind,decision,cui,reason\nlung,Anatomy,cui,C0024109,ok\n"
                    "change,Observation,skip,,qualifier\n", encoding="utf-8")
    assert lk.load_adjudication(path) == {
        ("Anatomy", "lung"): {"decision": "cui", "cui": "C0024109", "reason": "ok"},
        ("Observation", "change"): {"decision": "skip", "cui": "", "reason": "qualifier"}}
    for bad in ("lung,Anatomy,cui,,x\n", "lung,Anatomy,skip,C0024109,x\n", "lung,Body,none,,x\n",
                "tube,Observation,cui,PRJ:tubes,x\n"):
        path.write_text("string,kind,decision,cui,reason\n" + bad, encoding="utf-8")
        with pytest.raises(lk.LinkerError):
            lk.load_adjudication(path, {"PRJ:support_devices"})
    path.write_text("string,kind,decision,cui,reason\ntube,Observation,cui,PRJ:support_devices,x\n",
                    encoding="utf-8")
    assert lk.load_adjudication(path, {"PRJ:support_devices"})[("Observation", "tube")][
        "cui"] == "PRJ:support_devices"


def test_adjudicated_cuis_are_checked_not_corrected(umls):
    adj = {("Anatomy", "lung"): {"decision": "cui", "cui": "C1000007", "reason": ""},
           ("Anatomy", "x"): {"decision": "cui", "cui": "C9999999", "reason": ""},
           ("Anatomy", "y"): {"decision": "cui", "cui": "C1000003", "reason": ""},
           ("Anatomy", "z"): {"decision": "cui", "cui": "C1000017", "reason": ""},
           ("Observation", "t"): {"decision": "cui", "cui": "PRJ:support_devices", "reason": ""}}
    assert lk.check_adjudication(adj, umls, CFG, lk.domain(umls, CFG)) == [
        ("Anatomy", "x", "C9999999", "not in MRCONSO"),
        ("Anatomy", "y", "C1000003", "no semantic type of the kind"),
        ("Anatomy", "z", "C1000017", "outside the domain")]


def mentions(*rows):
    """(study, kind, entity, joined, located_at) -> the columns located_pairs reads."""
    return pl.DataFrame([(s, "impression", k, e, j, False, loc) for s, k, e, j, loc in rows],
                        schema={"study_id": pl.String, "section": pl.String, "kind": pl.String,
                                "entity": pl.String, "joined": pl.String,
                                "dropped_head": pl.Boolean, "located_at": pl.List(pl.String)},
                        orient="row")


PAIRS = mentions(
    *[(f"s{i}", "Anatomy", "1", "pleural", []) for i in range(10)],
    *[(f"s{i}", "Observation", "2", "effusion", ["1"]) for i in range(10)],
    ("s10", "Anatomy", "1", "lung", []), ("s10", "Observation", "2", "effusion", ["1"]),
    ("s11", "Observation", "2", "effusion", []),
    ("s12", "Anatomy", "1", "pleura", []), ("s12", "Observation", "2", "thickening", ["1"]))


def test_pairs_are_cut_at_their_coverage_and_tried_in_both_orders(umls):
    occurrences = lk.located_pairs(PAIRS, ["effusion", "thickening"])
    top = lk.top_pairs(occurrences, {"coverage": 0.9, "max_pairs": 10})
    # 10 pleural + 1 lung: lung starts at 10/11 > 0.9 of effusion's located mentions, so it is cut
    assert top.select("obs", "anat", "n").rows() == [("effusion", "pleural", 10),
                                                     ("thickening", "pleura", 1)]
    pairs = lk.pair_links(top, umls, CFG, lk.domain(umls, CFG))
    assert pairs.select("phrase", "cui").rows() == [("pleural effusion", "C1000001"),
                                                    ("thickening pleura", "C1000014")]
    share = lk.pair_share(PAIRS, occurrences, pairs, ["effusion"])
    assert share == {"effusion": {"mentions": 12, "located": 11, "pair_linked": 10}}
    assert lk.top_pairs(occurrences, {"coverage": 1.0, "max_pairs": 1})["anat"].to_list() == [
        "pleural", "pleura"]


def test_mentions_carry_their_entity_and_the_anatomy_heads_they_are_located_at():
    def entity(tokens, start, kind="Observation", relations=()):
        return {"tokens": tokens, "label": f"{kind}::definitely present", "start_ix": start,
                "end_ix": start, "relations": [list(r) for r in relations]}
    entities = {
        "0": entity("Small", 0, relations=[("modify", "2")]),
        "1": entity("pleural", 1, "Anatomy"),
        "2": entity("effusion", 2, relations=[("located_at", "1")]),
        "3": entity("left", 3, "Anatomy", [("modify", "4")]),
        "4": entity("lobe", 4, "Anatomy"),
        "5": entity("opacity", 5, relations=[("located_at", "3")]),       # a modifier: its head
        "6": entity("patchy", 6, relations=[("modify", "7"), ("located_at", "4")]),
        "7": entity("consolidation", 7),                                  # through its modifier
        "8": entity("tube", 8, relations=[("located_at", "2")]),          # not an Anatomy
    }
    got = {tokens[head]: (eid, located) for _, _, tokens, head, eid, located
           in vocab._mentions(entities)}
    assert got["effusion"] == ("2", ["1"])
    assert got["opacity"] == ("5", ["4"]) and got["consolidation"] == ("7", ["4"])
    assert got["tube"] == ("8", []) and got["lobe"] == ("4", [])


def test_overrides_hold_strings_and_codes_only(tmp_path):
    real = lk.load_overrides(default_config_dir() / "kg" / "linker_overrides.yaml")
    assert real["Observation"]["collapsed lung"] == "C0004144"
    bad = tmp_path / "overrides.yaml"
    for text in ("observation: {effusion: pleural effusion}\n", "observation: {Effusion: skip}\n",
                 "findings: {}\n"):
        bad.write_text(text, encoding="utf-8")
        with pytest.raises(lk.LinkerError):
            lk.load_overrides(bad)


class FakeChooser:
    """Answers with the first candidate, or `none` without one; counts its calls."""

    calls = 0

    def __init__(self, name, cfg, key):
        self.name, self.model = name, cfg["model"]

    def ask(self, system, user, schema):
        type(self).calls += 1
        cands = json.loads(user)["candidates"]
        answer = ({"decision": "cui", "cui": cands[0]["cui"], "reason": "first"} if cands
                  else {"decision": "none", "cui": "", "reason": "no candidate"})
        return json.dumps(answer), {"input_tokens": 100, "output_tokens": 10}, self.model


def test_cli_links_asks_the_llms_and_samples(world, umls, tmp_path,  # noqa: F811
                                             monkeypatch, capsys):
    pool, csv, rg = world
    pool_dir = tmp_path / "pool"
    pool_dir.mkdir()
    pool.write_parquet(pool_dir / "pool_images.parquet")
    (pool_dir / "pool.json").write_text(json.dumps({"pool_sha256": "abc"}), encoding="utf-8")
    common = ["--pool-dir", str(pool_dir), "--plus-csv", str(csv), "--radgraph-dir", str(rg),
              "--umls-dir", str(umls)]
    main(["kg-pool-vocab", *common, "--out", str(tmp_path / "vocab")])
    monkeypatch.setattr("src.kg.sapbert.SapBert", FakeSap)
    real_domain = lk.domain
    monkeypatch.setattr(lk, "domain", lambda umls_dir, _cfg: real_domain(umls_dir, CFG))
    monkeypatch.setattr("src.cli.main._chooser", FakeChooser)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret")
    out = tmp_path / "link"
    link_args = ["kg-link", *common, "--vocab-dir", str(tmp_path / "vocab"),
                 "--sapbert-cache", str(tmp_path / "sapbert"), "--out", str(out)]
    assert main(link_args) == 0
    links = pl.read_parquet(out / "links.parquet")
    table = pl.read_parquet(tmp_path / "vocab" / "vocab_joined.parquet")
    assert set(links["string"]) <= set(table["string"]) and links["method"].null_count() == 0
    selected = links.filter(pl.col("route").is_not_null()).height
    assert links.filter(pl.col("method") == "pending").height == selected > 0
    assert len((out / "llm_inputs.jsonl").read_text(encoding="utf-8").splitlines()) == selected

    assert main(["kg-link-llm", "--model", "opus", "--limit", "1", "--out", str(out)]) == 0
    assert FakeChooser.calls == 1
    for model in ("opus", "sonnet"):
        assert main(["kg-link-llm", "--model", model, "--all", "--out", str(out)]) == 0
    assert FakeChooser.calls == 2 * selected                 # the cached answer is not asked again
    assert main(link_args) == 0
    links = pl.read_parquet(out / "links.parquet")
    assert links.filter(pl.col("route").is_not_null())["method"].unique().to_list() == ["llm"]
    review = pl.read_csv(out / "review.csv")
    assert review.height == links.height
    assert {"llm_opus", "llm_sonnet", "agree", "flags"} <= set(review.columns)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["pool_sha256"] == "abc" and manifest["llm"]["opus"]["answered"] == selected
    assert len(manifest["llm"]["prompt_sha256"]) == 64
    assert "Pleural" not in capsys.readouterr().out      # counts only, no UMLS names

    assert main(["kg-link-sample", "--umls-dir", str(umls), "--out", str(out)]) == 0
    sample = pl.read_csv(out / "sample.csv", infer_schema_length=0).fill_null("")
    assert 0 < sample.height <= selected and set(sample["link_grade"]) == {""}
    assert main(["kg-link-sample", "--umls-dir", str(umls), "--out", str(out)]) == 1   # kept
    graded = out / "graded.csv"
    sample.with_columns(pl.lit("correct").alias("link_grade")).write_csv(graded)
    assert main(["kg-link-sample", "--score", str(graded), "--out", str(out)]) == 0
    gate = json.loads((out / "gate.json").read_text(encoding="utf-8"))
    assert gate["link"]["precision"] == 1.0 and gate["precision_passed"]


def test_cli_stops_when_the_vocabulary_differs_from_v2_2(world, umls, tmp_path,  # noqa: F811
                                                          capsys):
    pool, csv, rg = world
    pool_dir = tmp_path / "pool"
    pool_dir.mkdir()
    pool.write_parquet(pool_dir / "pool_images.parquet")
    (pool_dir / "pool.json").write_text(json.dumps({"pool_sha256": "abc"}), encoding="utf-8")
    vocab_dir = tmp_path / "vocab"
    vocab_dir.mkdir()
    pl.DataFrame({"kind": ["Observation"], "string": ["x"], "mentions": [1], "studies": [1]},
                 schema={"kind": pl.String, "string": pl.String, "mentions": pl.UInt32,
                         "studies": pl.UInt32}).write_parquet(vocab_dir / "vocab_joined.parquet")
    assert main(["kg-link", "--pool-dir", str(pool_dir), "--plus-csv", str(csv),
                 "--radgraph-dir", str(rg), "--umls-dir", str(umls), "--vocab-dir",
                 str(vocab_dir), "--out", str(tmp_path / "link")]) == 1
    assert "differs" in capsys.readouterr().err

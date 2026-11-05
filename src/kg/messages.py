"""The mandatory warnings of the KG search (docs/v2-kg-design.md, "Rào chắn"), as codes with
parameters, rendered in the query's language. Code decides which warning a result needs; this
file only words it. Nothing here is a diagnosis: the score says how well an image fits the query.
"""

from __future__ import annotations

from dataclasses import dataclass, field

TEXT = {
    "narrower": {
        "en": "Matches only at the level of {ancestor}, not {name}.",
        "vi": "Chỉ khớp ở mức {ancestor}, không riêng {name}."},
    "anatomy_region": {
        "en": "Matched through the findings usually seen in {region}; the side is not considered.",
        "vi": "Khớp theo các finding thường gặp ở {region}; không xét bên."},
    "disease_inferred": {
        "en": "Inferred from signs, not a diagnosis ({name}).",
        "vi": "Suy từ dấu hiệu, không phải chẩn đoán ({name})."},
    "unrecognized": {
        "en": "Could not score \"{term}\": no concept of the knowledge graph fits it.",
        "vi": "Không chấm được \"{term}\": không có khái niệm nào của KG khớp."},
    "no_path": {
        "en": "Could not score \"{term}\": {name} has no path to a concept the image model scores.",
        "vi": "Không chấm được \"{term}\": {name} không nối tới khái niệm nào model ảnh chấm."},
    "weak_quality": {
        "en": "Low reliability for {name} (AUROC {auroc} on validation, below 0.65).",
        "vi": "Độ tin cậy thấp với {name} (AUROC {auroc} trên tập kiểm định, dưới 0,65)."},
    "negation_skipped": {
        "en": "Not excluded: \"no {name}\" is narrower than anything the image model scores.",
        "vi": "Không loại trừ: \"không {name}\" hẹp hơn mọi khái niệm model ảnh chấm."},
    "location_ignored": {
        "en": "The location \"{region}\" is not used for ranking; the findings are scored.",
        "vi": "Vị trí \"{region}\" không dùng để xếp hạng: câu hỏi được chấm theo dấu hiệu."},
    "side_ignored": {
        "en": "The side or severity in the question is not used for ranking.",
        "vi": "Bên hoặc mức độ trong câu hỏi không dùng để xếp hạng."},
    "uncertain_as_present": {
        "en": "\"Possible {name}\" is searched as {name}.",
        "vi": "\"Nghi ngờ {name}\" được tìm như {name}."},
    "llm_fallback": {
        "en": "The language model was unavailable ({reason}); the deterministic route was used.",
        "vi": "Model ngôn ngữ không dùng được ({reason}); đã dùng đường tất định."},
    "kg_unavailable": {
        "en": "The knowledge graph was unavailable; only the 14 labels were scored.",
        "vi": "KG không dùng được; chỉ chấm 14 nhãn."},
    "terms_dropped": {
        "en": "Without the knowledge graph \"{term}\" cannot be scored and was left out.",
        "vi": "Không có KG nên không chấm được \"{term}\"; đã bỏ."},
    "chosen_deterministic": {
        "en": "\"{term}\" was matched by name ({name}), without the language model.",
        "vi": "\"{term}\" được khớp theo tên ({name}), không qua model ngôn ngữ."},
}


@dataclass(frozen=True)
class Warn:
    code: str
    params: dict = field(default_factory=dict)

    def render(self, lang: str = "en") -> str:
        return TEXT[self.code][lang if lang in ("en", "vi") else "en"].format(**self.params)


def render_all(warnings: list[Warn], lang: str) -> list[str]:
    """Each distinct warning once, in order."""
    return list(dict.fromkeys(w.render(lang) for w in warnings))

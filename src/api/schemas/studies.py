"""GET /api/v1/studies/{study_id}: the study detail of the UI (spec 5.1; deferred by D15)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.api.schemas.search import LabelAssertion, View


class StudyReport(BaseModel):
    impression: str | None
    findings: str | None
    search_text: str


class StudyDetail(BaseModel):
    study_id: str
    image_id: str
    projection: View | None
    labels: list[LabelAssertion] = Field(description="all 14 CheXpert labels, resolved (G1)")
    report: StudyReport | None = Field(description="null when the canonical report is missing")
    build_id: str

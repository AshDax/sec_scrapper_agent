"""Pydantic schemas shared across every pipeline step.

BusinessRecord is the single source of truth for all extracted attributes.
Extend it here when you add new fields — every downstream tool picks up
the change automatically.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Core extraction record — maps 1-to-1 with the Delta table row
# ---------------------------------------------------------------------------


class BusinessRecord(BaseModel):
    """Structured business attributes extracted from an SEC filing."""

    business_name: Optional[str] = None
    parent_name: Optional[str] = None
    business_phone: Optional[str] = None
    address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip_code: Optional[str] = None
    industry_description: Optional[str] = None
    naics_sic_codes: list[str] = Field(default_factory=list)
    num_employees: Optional[str] = None
    is_manufacturer: Optional[bool] = None
    risk_factor_keywords: list[str] = Field(default_factory=list)
    revenue_mentions: list[str] = Field(default_factory=list)
    website_domain: Optional[str] = None
    business_purpose_summary: Optional[str] = None
    is_open: Optional[bool] = None

    # ---- helpers -----------------------------------------------------------

    CORE_FIELDS: list[str] = [
        "business_name", "parent_name", "business_phone",
        "address", "city", "state", "zip_code",
        "industry_description", "naics_sic_codes", "num_employees",
        "is_manufacturer", "risk_factor_keywords", "revenue_mentions",
        "website_domain", "business_purpose_summary", "is_open",
    ]

    def filled_fields(self) -> list[str]:
        filled = []
        for f in self.CORE_FIELDS:
            v = getattr(self, f)
            if v is not None and v != [] and v != "":
                filled.append(f)
        return filled

    def missing_fields(self) -> list[str]:
        return [f for f in self.CORE_FIELDS if f not in self.filled_fields()]

    def fill_rate(self) -> float:
        return len(self.filled_fields()) / len(self.CORE_FIELDS)


# ---------------------------------------------------------------------------
# Evaluation result
# ---------------------------------------------------------------------------


class EvaluationResult(BaseModel):
    valid: bool = False
    confidence: float = 0.0
    fill_rate: float = 0.0
    filled_fields: list[str] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# LangGraph state — flows through every node
# ---------------------------------------------------------------------------


class ExtractionState(BaseModel):
    """Mutable state that travels through the LangGraph pipeline."""

    raw_text: str = ""
    clean_text: str = ""

    scraper_result: dict = Field(default_factory=dict)
    retrieval_context: list[str] = Field(default_factory=list)

    record: dict = Field(default_factory=dict)
    evaluation: dict = Field(default_factory=dict)
    fill_rate: float = 0.0

    web_context: list[str] = Field(default_factory=list)
    iteration: int = 0

    class Config:
        arbitrary_types_allowed = True

"""Step 6 — Enrichment Agent

Derives computed fields that should NOT be hallucinated by an LLM:
  - is_manufacturer  — keyword match in business description
  - is_open          — keyword match for closure/bankruptcy
  - industry_description enhancement from NAICS code lookup

Add new derived fields here when you want deterministic logic
instead of LLM inference.

Usage:
    from agent_server.sec_extraction.tools.enrichment import enrich_record
    enriched = enrich_record(record_dict, clean_text)
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# NAICS code → industry description (top-level, extend as needed)
# ---------------------------------------------------------------------------

NAICS_LOOKUP: dict[str, str] = {
    "11": "Agriculture, Forestry, Fishing and Hunting",
    "21": "Mining, Quarrying, and Oil and Gas Extraction",
    "22": "Utilities",
    "23": "Construction",
    "31": "Manufacturing",
    "32": "Manufacturing",
    "33": "Manufacturing",
    "42": "Wholesale Trade",
    "44": "Retail Trade",
    "45": "Retail Trade",
    "48": "Transportation and Warehousing",
    "49": "Transportation and Warehousing",
    "51": "Information",
    "52": "Finance and Insurance",
    "53": "Real Estate and Rental and Leasing",
    "54": "Professional, Scientific, and Technical Services",
    "55": "Management of Companies and Enterprises",
    "56": "Administrative and Support and Waste Management",
    "61": "Educational Services",
    "62": "Health Care and Social Assistance",
    "71": "Arts, Entertainment, and Recreation",
    "72": "Accommodation and Food Services",
    "81": "Other Services (except Public Administration)",
    "92": "Public Administration",
}

SIC_MANUFACTURING_RANGES = range(2000, 4000)

MANUFACTURER_KEYWORDS = re.compile(
    r"(?:manufactur|produc(?:tion|e|ing)|fabricat|assembl|factory|plant|"
    r"production\s+facilit|industrial\s+component|precision[- ]engineer)",
    re.IGNORECASE,
)

CLOSURE_KEYWORDS = re.compile(
    r"(?:dissolv|dissolution|bankrupt|ceased\s+operations?|wound\s+down|"
    r"liquidat|deregistered|revoked|no\s+longer\s+operat)",
    re.IGNORECASE,
)


def _naics_industry(codes: list[str]) -> str | None:
    """Look up industry description from the first recognised NAICS code."""
    for code in codes:
        prefix = code[:2]
        if prefix in NAICS_LOOKUP:
            return f"{NAICS_LOOKUP[prefix]} (NAICS {code})"
    return None


def _is_manufacturer_from_codes(codes: list[str]) -> bool | None:
    """Check if any SIC code falls in the manufacturing range."""
    for code in codes:
        try:
            if int(code) in SIC_MANUFACTURING_RANGES:
                return True
        except ValueError:
            continue
    return None


def enrich_record(record_dict: dict, source_text: str) -> dict:
    """Add derived fields to the record.

    Mutates and returns *record_dict* in place.
    """
    naics = record_dict.get("primary_naics_code_id") or ""
    sic = record_dict.get("primary_sic_code_id") or ""
    codes = [c for c in [naics, sic] if c]

    # ---- in_business -------------------------------------------------------
    if not record_dict.get("in_business"):
        is_closed = bool(CLOSURE_KEYWORDS.search(source_text))
        record_dict["in_business"] = "No" if is_closed else "Yes"

    # ---- company_active_indicator ------------------------------------------
    if record_dict.get("company_active_indicator") is None:
        record_dict["company_active_indicator"] = record_dict.get("in_business") != "No"

    # ---- place_type from SIC codes -----------------------------------------
    if not record_dict.get("place_type"):
        code_mfg = _is_manufacturer_from_codes(codes)
        keyword_mfg = bool(MANUFACTURER_KEYWORDS.search(source_text))
        if code_mfg or keyword_mfg:
            record_dict["place_type"] = "Manufacturer"

    # ---- company_sic_name / company_naics_name from codes ------------------
    if not record_dict.get("company_sic_name") and sic:
        naics_desc = _naics_industry([sic])
        if naics_desc:
            record_dict["company_sic_name"] = naics_desc
    if not record_dict.get("company_naics_name") and naics:
        naics_desc = _naics_industry([naics])
        if naics_desc:
            record_dict["company_naics_name"] = naics_desc

    return record_dict

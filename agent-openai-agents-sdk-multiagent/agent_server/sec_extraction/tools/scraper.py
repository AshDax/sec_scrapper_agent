"""Step 2 — Scraper Agent

Regex-based attribute extraction.  Fast, deterministic, and requires no
LLM calls.  Returns a partial BusinessRecord dict — downstream steps fill
in whatever the scraper misses.

Add new patterns to PATTERNS to teach the scraper about new attributes.

Usage:
    from agent_server.sec_extraction.tools.scraper import scrape_attributes
    partial = scrape_attributes(clean_text)
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Pattern registry — { field_name: [regex, ...] }
# Each regex should use a capturing group for the value of interest.
# ---------------------------------------------------------------------------

PATTERNS: dict[str, list[str]] = {
    "business_name": [
        r"(?:EXACT NAME OF REGISTRANT|Company Name|Registrant)[:\s]+([A-Z][\w\s&.,'-]+(?:Inc|Corp|LLC|Ltd|Co|LP|Company|Corporation|Group|Holdings)\.?)",
        r"^([A-Z][A-Z\s&.,'-]+(?:INC|CORP|LLC|LTD|CO|LP|COMPANY|CORPORATION|GROUP|HOLDINGS)\.?)\s*$",
    ],
    "business_phone": [
        r"\((\d{3})\)\s*(\d{3})[-.](\d{4})",
        r"(?:Telephone|Phone|Tel)[:\s]*\(?(\d{3})\)?[-.\s]?(\d{3})[-.\s]?(\d{4})",
    ],
    "website_domain": [
        r"(?:https?://)?(?:www\.)?([a-zA-Z0-9][-a-zA-Z0-9]*\.(?:com|org|net|io|co|gov|edu)(?:\.[a-zA-Z]{2})?)",
    ],
    "num_employees": [
        r"(?:approximately|about|nearly|over|more than)?\s*(\d{1,3}(?:,\d{3})*)\s+(?:full[- ]?time\s+)?employees",
        r"(\d{1,3}(?:,\d{3})*)\s+(?:people|personnel|workers|staff)",
        r"(?:headcount|workforce)\s+(?:of\s+)?(?:approximately\s+)?(\d{1,3}(?:,\d{3})*)",
    ],
    "revenue_mentions": [
        r"(\$\s*[\d,]+(?:\.\d+)?\s*(?:billion|million|thousand|B|M|K))",
        r"(?:revenue|net\s+sales|total\s+revenue)\s+(?:of|was|were|totaled)?\s*(\$\s*[\d,]+(?:\.\d+)?(?:\s*(?:billion|million))?)",
    ],
    "address": [
        r"(\d+\s+[\w\s]+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Boulevard|Blvd|Way|Lane|Ln|Place|Pl|Suite|Ste)\.?(?:\s*,?\s*(?:Suite|Ste)\.?\s*\d+)?)",
    ],
    "zip_code": [
        r"\b([A-Z]{2})\s+(\d{5}(?:-\d{4})?)\b",
    ],
    "naics_sic_codes": [
        r"(?:NAICS|SIC)\s*(?:code|Code)?:?\s*(\d{4,6})",
        r"Standard\s+Industrial\s+Classification\s*(?:code)?\s*:?\s*(\d{4})",
    ],
    "risk_factor_keywords": [
        r"(?:risk\s+factor|risks?\s+include)[s:]?\s*[-•]\s*(.+?)(?:\n|$)",
    ],
}


def _first_match(text: str, patterns: list[str]) -> str | None:
    """Return the full match (group 0) of the first matching pattern."""
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(0).strip()
    return None


def _all_matches(text: str, patterns: list[str]) -> list[str]:
    """Return all unique matches across all patterns."""
    found: list[str] = []
    for pat in patterns:
        for m in re.finditer(pat, text, re.IGNORECASE | re.MULTILINE):
            val = (m.group(1) if m.lastindex else m.group(0)).strip()
            if val and val not in found:
                found.append(val)
    return found


def scrape_attributes(text: str) -> dict:
    """Run all regex patterns against *text* and return a partial record dict.

    Returns only the fields that had at least one match — downstream steps
    fill in the rest.
    """
    result: dict = {}

    # Single-value fields
    for field in ("business_name", "business_phone", "website_domain", "num_employees", "address"):
        patterns = PATTERNS.get(field, [])
        val = _first_match(text, patterns)
        if val:
            result[field] = val

    # Extract state + zip from the zip_code pattern
    for pat in PATTERNS.get("zip_code", []):
        m = re.search(pat, text)
        if m:
            result["state"] = m.group(1)
            result["zip_code"] = m.group(2)
            break

    # List-value fields
    for field in ("naics_sic_codes", "revenue_mentions", "risk_factor_keywords"):
        patterns = PATTERNS.get(field, [])
        vals = _all_matches(text, patterns)
        if vals:
            result[field] = vals

    return result

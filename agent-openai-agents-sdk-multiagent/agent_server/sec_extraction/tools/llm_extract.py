"""Step 4 — LLM Extractor Agent

Uses a chat model (Databricks or Groq) with structured output to produce
a full extraction record from the filing text + any scraper/retrieval context.

The schema is loaded from attribute_registry (financial_datum_attributes.csv
and place_attributes.csv).

Usage:
    from agent_server.sec_extraction.tools.llm_extract import llm_extract_record
    from agent_server.sec_extraction.config import get_config
    record_dict = llm_extract_record(clean_text, scraper_result, context_chunks, get_config())
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage

if TYPE_CHECKING:
    from agent_server.sec_extraction.config import ExtractionConfig

from agent_server.sec_extraction.attribute_registry import get_registry

logger = logging.getLogger(__name__)

SEC_EXTRACTABLE_ATTRS = [
    "company_name", "company_legal_name", "company_ein", "cik",
    "company_address", "company_city", "company_state", "company_postal_code",
    "company_phone", "website",
    "name", "street", "city", "state", "postal_code", "phone",
    "primary_sic_code_id", "primary_naics_code_id",
    "company_sic_code", "company_sic_name", "company_naics_code", "company_naics_name",
    "location_employee_count", "corporate_employee_count",
    "revenue", "net_income", "gross_profit", "cost_of_revenue",
    "total_assets", "total_liabilities_and_equity",
    "operating_expenses", "operating_income",
    "cash", "current_assets", "shareholders_equity",
    "long_term_debt", "short_term_debt", "total_debt",
    "stock_ticker_symbol", "stock_exchange_code",
    "report_date", "fiscal_year_end_month",
    "company_year_founded", "company_description",
    "in_business", "company_active_indicator",
]


def _build_system_prompt() -> str:
    """Compact system prompt — only SEC-extractable fields to stay within token limits."""
    registry = get_registry()
    extractable = set(SEC_EXTRACTABLE_ATTRS)
    attrs = [a for a in registry.attributes if a["attribute"] in extractable]

    schema_lines = ["{"]
    for i, attr in enumerate(attrs):
        comma = "," if i < len(attrs) - 1 else ""
        schema_lines.append(f'  "{attr["attribute"]}": {attr["json_type"]}{comma}')
    schema_lines.append("}")
    schema_block = "\n".join(schema_lines)

    return (
        "You are an SEC filing data extractor. "
        "Return ONLY valid JSON (no markdown, no commentary) matching this schema:\n\n"
        f"{schema_block}\n\n"
        "Rules: use document values only; null for missing fields; "
        "boolean fields default true unless closure/dissolution evidence; "
        "numeric financial values should be raw numbers (not formatted strings)."
    )


def llm_extract_record(
    text: str,
    scraper_result: dict,
    context_chunks: list[str],
    config: ExtractionConfig,
) -> dict:
    """Run LLM structured extraction and return a BusinessRecord dict.

    Args:
        text:           Clean filing text.
        scraper_result: Partial dict from the scraper step (merged as hints).
        context_chunks: Supplementary text from Vector Search.
        config:         Pipeline configuration.

    Returns:
        Dict matching BusinessRecord fields.
    """
    llm = config.get_llm(temperature=0)

    context_block = ""
    if context_chunks:
        joined = "\n---\n".join(c[:500] for c in context_chunks[:3])
        context_block = f"\n\n[Context]\n{joined}"

    scraper_block = ""
    if scraper_result:
        extractable = set(SEC_EXTRACTABLE_ATTRS)
        relevant_hints = {k: v for k, v in scraper_result.items() if k in extractable}
        if relevant_hints:
            scraper_block = "\n\n[Hints]\n" + json.dumps(relevant_hints, separators=(",", ":"))

    user_content = (
        f"[Filing]\n{text[:4000]}"
        f"{context_block}"
        f"{scraper_block}"
        "\n\nExtract now."
    )

    try:
        prompt = _build_system_prompt()
        response = llm.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content=user_content),
        ])

        raw = response.content.strip()

        # Strip markdown code fences if present
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        parsed = json.loads(raw)

        # Filter to only valid registry attributes
        registry = get_registry()
        valid_keys = set(registry.attribute_names)
        record = {k: v for k, v in parsed.items() if k in valid_keys}

        # Merge with scraper result for any fields we didn't get
        for k, v in scraper_result.items():
            if k in valid_keys and record.get(k) is None:
                record[k] = v

        return record

    except json.JSONDecodeError:
        logger.warning("LLM returned non-JSON — falling back to scraper result")
        return scraper_result
    except Exception as exc:
        logger.warning("LLM extraction failed: %s", exc)
        return scraper_result

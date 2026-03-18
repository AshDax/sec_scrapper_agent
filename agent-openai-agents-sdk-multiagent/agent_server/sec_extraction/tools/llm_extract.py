"""Step 4 — LLM Extractor Agent

Uses a chat model (Databricks or Groq) with structured output to produce
a full extraction record from the filing text + any scraper/retrieval context.

Only the first MAX_FILING_CHARS of the document are sent in a single LLM call
to stay within the model's context limit (e.g. Groq 8K tokens).

Usage:
    from agent_server.sec_extraction.tools.llm_extract import llm_extract_record
    from agent_server.sec_extraction.config import get_config
    record_dict, llm_record = llm_extract_record(clean_text, scraper_result, context_chunks, get_config())
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

# Max characters to send in one request (fits within Groq 8K token context with system + hints).
MAX_FILING_CHARS = 6_000
MAX_HINTS_CHARS = 400

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
) -> tuple[dict, dict]:
    """Run a single LLM extraction on up to MAX_FILING_CHARS of the document.

    Returns (full_record, llm_record) where full_record is merged with scraper
    and llm_record is what the LLM produced alone.
    """
    registry = get_registry()
    valid_keys = set(registry.attribute_names)

    filing_excerpt = (text or "")[:MAX_FILING_CHARS]

    context_block = ""
    if context_chunks:
        joined = "\n---\n".join(c[:300] for c in context_chunks[:2])
        context_block = f"\n\n[Context]\n{joined}"[:800]

    scraper_block = ""
    if scraper_result:
        relevant_hints = {k: v for k, v in scraper_result.items() if k in set(SEC_EXTRACTABLE_ATTRS)}
        if relevant_hints:
            raw_hints = json.dumps(relevant_hints, separators=(",", ":"))
            scraper_block = "\n\n[Hints]\n" + raw_hints[:MAX_HINTS_CHARS]

    user_content = (
        f"[Filing]\n{filing_excerpt}"
        f"{context_block}"
        f"{scraper_block}"
        "\n\nExtract all fields present. Use null for missing."
    )

    llm_record: dict = {}
    try:
        llm = config.get_llm(temperature=0)
        prompt = _build_system_prompt()
        response = llm.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content=user_content),
        ])
        raw = response.content.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        parsed = json.loads(raw)
        llm_record = {k: v for k, v in parsed.items() if k in valid_keys}
    except json.JSONDecodeError:
        logger.warning("LLM returned non-JSON")
    except Exception as exc:
        logger.warning("LLM extraction failed: %s", exc)

    record = dict(llm_record)
    for k, v in scraper_result.items():
        if k in valid_keys and (record.get(k) is None or record.get(k) == "" or record.get(k) == []):
            record[k] = v

    logger.info(
        "llm_extract: sent %d chars, got %d fields from LLM, %d total after scraper merge",
        len(filing_excerpt),
        sum(1 for v in llm_record.values() if v is not None and v != "" and v != []),
        sum(1 for v in record.values() if v is not None and v != [] and v != ""),
    )
    return record, llm_record

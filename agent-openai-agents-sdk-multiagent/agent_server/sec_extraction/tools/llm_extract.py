"""Step 4 — LLM Extractor Agent

Uses a chat model (Databricks or Groq) with structured output to produce
a full BusinessRecord from the filing text + any scraper/retrieval context.

The prompt is designed to be self-contained — extend it when you add
new fields to BusinessRecord.

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

from agent_server.sec_extraction.schemas import BusinessRecord

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a precise SEC filing data extractor.  Given the filing
text and any supplementary context, extract ALL of the following fields.

Return ONLY valid JSON matching this schema (no markdown fences, no commentary):

{
  "business_name": "string or null",
  "parent_name": "string or null",
  "business_phone": "string or null",
  "address": "string or null",
  "city": "string or null",
  "state": "string or null",
  "zip_code": "string or null",
  "industry_description": "string or null",
  "naics_sic_codes": ["string", ...],
  "num_employees": "string or null",
  "is_manufacturer": true/false/null,
  "risk_factor_keywords": ["string", ...],
  "revenue_mentions": ["string", ...],
  "website_domain": "string or null",
  "business_purpose_summary": "one paragraph or null",
  "is_open": true/false/null
}

Rules:
- Keep original values from the document (don't invent data).
- For is_manufacturer, look for: manufactur, produc, fabricat, assembl, plant, factory.
- For is_open, look for: dissolution, bankruptcy, ceased operations, wound down, liquidat.
  Default to true if no closure evidence.
- For revenue_mentions, include the dollar amount AND context (e.g. "$487M total revenue FY2024").
- For risk_factor_keywords, extract concise phrases (e.g. "supply chain disruption").
"""


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
        context_block = "\n\n--- Supplementary Context ---\n" + "\n---\n".join(context_chunks)

    scraper_block = ""
    if scraper_result:
        scraper_block = (
            "\n\n--- Scraper hints (regex matches — may need refinement) ---\n"
            + json.dumps(scraper_result, indent=2)
        )

    user_content = (
        f"SEC Filing Text:\n{text[:12000]}"
        f"{context_block}"
        f"{scraper_block}"
        "\n\nExtract all fields now."
    )

    try:
        response = llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_content),
        ])

        raw = response.content.strip()

        # Strip markdown code fences if present
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        parsed = json.loads(raw)

        record = BusinessRecord(**parsed)
        return record.model_dump()

    except json.JSONDecodeError:
        logger.warning("LLM returned non-JSON — falling back to scraper result")
        return scraper_result
    except Exception as exc:
        logger.warning("LLM extraction failed: %s", exc)
        return scraper_result

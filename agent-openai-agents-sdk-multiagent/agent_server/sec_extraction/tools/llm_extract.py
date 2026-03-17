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


def _build_system_prompt() -> str:
    """Build the LLM system prompt from the attribute registry."""
    registry = get_registry()
    schema_block = registry.get_json_schema_block()
    desc_block = registry.get_descriptions_block()

    return f"""You are a precise SEC filing and business data extractor. Given the filing
text and any supplementary context, extract ALL of the following fields.

Return ONLY valid JSON matching this schema (no markdown fences, no commentary):

{schema_block}

Attribute descriptions:
{desc_block}

Rules:
- Keep original values from the document (don't invent data).
- For boolean fields (in_business, is_manufacturer, etc.), infer from context.
- For is_open/in_business, look for: dissolution, bankruptcy, ceased operations, wound down.
  Default to true if no closure evidence.
- For revenue/financial fields, include numeric values when present.
- For array fields, use empty list [] when no values found.
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

"""Step 5 — Self-Evaluation Agent

Validates a BusinessRecord using an LLM judge and computes a fill rate.
The fill rate drives the conditional edge in the LangGraph workflow:
if it's below the threshold the pipeline falls back to web search.

Usage:
    from agent_server.sec_extraction.tools.evaluate import evaluate_record
    from agent_server.sec_extraction.config import get_config
    evaluation = evaluate_record(record_dict, source_text, get_config())
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
from agent_server.sec_extraction.schemas import EvaluationResult

logger = logging.getLogger(__name__)

EVAL_PROMPT = """You are a data quality auditor.  Given a structured business
record and the source SEC filing text, evaluate the extraction quality.

Return ONLY valid JSON:
{
  "valid": true/false,
  "confidence": 0.0-1.0,
  "issues": ["issue1", "issue2", ...]
}

Check for:
- Hallucinated values not present in the source
- Incorrect data types (phone should be digits, zip should be 5 digits, etc.)
- Missing critical fields (name, street, company_name)
- Implausible values (negative employees, future dates)
"""


def compute_fill_rate(record_dict: dict) -> tuple[float, list[str], list[str]]:
    """Return (fill_rate, filled_fields, missing_fields) using the attribute registry."""
    registry = get_registry()
    rate, missing = registry.compute_fill_rate(record_dict)
    filled = [name for name in registry.attribute_names if name not in set(missing)]
    return rate, filled, missing


def _llm_filled_fields(llm_record: dict | None) -> list[str]:
    """List of attribute names that have a non-empty value in the LLM-only record."""
    if not llm_record:
        return []
    return [
        k for k, v in llm_record.items()
        if v is not None and v != "" and v != []
    ]


def evaluate_record(
    record_dict: dict,
    source_text: str,
    config: ExtractionConfig,
    llm_record: dict | None = None,
) -> dict:
    """Run LLM evaluation + fill rate on a BusinessRecord dict.

    If llm_record is provided (extraction from LLM before scraper fill),
    the result includes llm_filled_fields and llm_record in the evaluation.
    """
    fill_rate, filled, missing = compute_fill_rate(record_dict)
    llm_filled = _llm_filled_fields(llm_record) if llm_record else []

    llm_eval = {"valid": True, "confidence": fill_rate, "issues": []}

    try:
        llm = config.get_llm(temperature=0)

        user_content = (
            f"Record:\n{json.dumps(record_dict, indent=2)}\n\n"
            f"Source text (first 4000 chars):\n{source_text[:4000]}"
        )

        response = llm.invoke([
            SystemMessage(content=EVAL_PROMPT),
            HumanMessage(content=user_content),
        ])

        raw = response.content.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        llm_eval = json.loads(raw)

    except Exception as exc:
        logger.warning("LLM evaluation failed: %s — using fill-rate only", exc)

    return EvaluationResult(
        valid=llm_eval.get("valid", fill_rate >= 0.5),
        confidence=llm_eval.get("confidence", fill_rate),
        fill_rate=fill_rate,
        filled_fields=filled,
        missing_fields=missing,
        issues=llm_eval.get("issues", []),
        llm_filled_fields=llm_filled,
        llm_record=llm_record or {},
    ).model_dump()

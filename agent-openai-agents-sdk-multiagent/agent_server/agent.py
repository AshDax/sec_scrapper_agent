"""
SEC Filing Information Extraction Agent — LangChain + LangGraph.

Wraps the LangGraph extraction pipeline in the MLflow Responses API
so it can be served as a Databricks App with a chat interface.

Pipeline:
  text_extract → scraper → retrieval → llm_extract → enrichment
  → evaluate → (web_fallback if low fill rate) → end

Each step is a separate module in agent_server/sec_extraction/tools/.
"""

from __future__ import annotations

import json
import logging
from typing import AsyncGenerator

import mlflow
from dotenv import load_dotenv
from mlflow.genai.agent_server import invoke, stream
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)

from agent_server.sec_extraction.config import ExtractionConfig
from agent_server.sec_extraction.workflow import extraction_workflow
from agent_server.sec_extraction.schemas import BusinessRecord

load_dotenv(dotenv_path=".env", override=True)

logging.getLogger("mlflow.utils.autologging_utils").setLevel(logging.ERROR)
mlflow.langchain.autolog()

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_user_text(request: ResponsesAgentRequest) -> str:
    """Pull the latest user message from the request."""
    for item in reversed(request.input):
        dumped = item.model_dump() if hasattr(item, "model_dump") else item
        if dumped.get("role") == "user":
            content = dumped.get("content", "")
            if isinstance(content, list):
                return " ".join(
                    c.get("text", "") for c in content if isinstance(c, dict)
                )
            return str(content)
    return ""


def _get_session_id(request: ResponsesAgentRequest) -> str | None:
    if request.context and request.context.conversation_id:
        return request.context.conversation_id
    if request.custom_inputs and isinstance(request.custom_inputs, dict):
        return request.custom_inputs.get("session_id")
    return None


def _format_result(result: dict) -> str:
    """Turn a pipeline result dict into a human-readable response."""
    record = result.get("record", {})
    evaluation = result.get("evaluation", {})
    fill_rate = evaluation.get("fill_rate", 0)

    biz_name = record.get("business_name", "Unknown Company")
    status = "Open" if record.get("is_open", True) else "Closed"

    lines = [
        f"## Extraction Results — {biz_name}",
        f"**Status:** {status} | **Fill Rate:** {fill_rate:.0%} | "
        f"**Confidence:** {evaluation.get('confidence', 0):.0%}",
        "",
    ]

    lines.append("### Extracted Attributes")
    lines.append("")
    lines.append("| Attribute | Value |")
    lines.append("|-----------|-------|")

    labels = {
        "business_name": "Business Name",
        "parent_name": "Parent Company",
        "business_phone": "Phone",
        "address": "Address",
        "city": "City",
        "state": "State",
        "zip_code": "Zip Code",
        "industry_description": "Industry",
        "naics_sic_codes": "NAICS/SIC Codes",
        "num_employees": "Employees",
        "is_manufacturer": "Manufacturer?",
        "risk_factor_keywords": "Risk Factors",
        "revenue_mentions": "Revenue",
        "website_domain": "Website",
        "business_purpose_summary": "Business Purpose",
        "is_open": "Business Open?",
    }

    for field, label in labels.items():
        val = record.get(field)
        if val is None:
            display = "*not found*"
        elif isinstance(val, bool):
            display = "Yes" if val else "No"
        elif isinstance(val, list):
            display = ", ".join(str(v) for v in val) if val else "*none*"
        else:
            display = str(val)
        lines.append(f"| {label} | {display} |")

    missing = evaluation.get("missing_fields", [])
    if missing:
        lines.append("")
        lines.append(f"**Missing fields ({len(missing)}):** {', '.join(missing)}")

    issues = evaluation.get("issues", [])
    if issues:
        lines.append("")
        lines.append("**Issues:** " + "; ".join(issues))

    lines.append("")
    lines.append("### Raw JSON")
    lines.append("```json")
    lines.append(json.dumps(record, indent=2, default=str))
    lines.append("```")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# MLflow Responses API handlers
# ---------------------------------------------------------------------------


@invoke()
async def invoke_handler(request: ResponsesAgentRequest) -> ResponsesAgentResponse:
    if session_id := _get_session_id(request):
        mlflow.update_current_trace(metadata={"mlflow.trace.session": session_id})

    user_text = _get_user_text(request)
    config = ExtractionConfig()

    result = extraction_workflow(user_text, config)

    response_text = _format_result(result)

    return ResponsesAgentResponse(
        output=[
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": response_text}],
            }
        ]
    )


@stream()
async def stream_handler(
    request: ResponsesAgentRequest,
) -> AsyncGenerator[ResponsesAgentStreamEvent, None]:
    if session_id := _get_session_id(request):
        mlflow.update_current_trace(metadata={"mlflow.trace.session": session_id})

    user_text = _get_user_text(request)
    config = ExtractionConfig()

    # Stream progress per workflow node
    from agent_server.sec_extraction.workflow import build_extraction_workflow

    workflow = build_extraction_workflow(config)
    initial_state = {
        "raw_text": user_text,
        "clean_text": "",
        "scraper_result": {},
        "retrieval_context": [],
        "record": {},
        "evaluation": {},
        "fill_rate": 0.0,
        "web_context": [],
        "iteration": 0,
    }

    step_names = {
        "text_extract": "Parsing document...",
        "scraper": "Running regex extraction...",
        "retrieval": "Searching SEC filings database...",
        "llm_extract": "Extracting attributes with LLM...",
        "enrichment": "Enriching derived fields...",
        "evaluate": "Evaluating extraction quality...",
        "web_fallback": "Searching web for missing data...",
        "re_evaluate": "Re-evaluating after web search...",
    }

    item_id = "msg_stream_001"
    final_result = initial_state

    # Emit stream start
    yield {
        "type": "response.output_item.added",
        "item": {"id": item_id, "type": "message", "role": "assistant", "content": []},
    }

    accumulated_text = ""

    for event in workflow.stream(
        initial_state,
        config={"configurable": {"ext_config": config}},
        stream_mode="updates",
    ):
        for node_name, node_output in event.items():
            final_result.update(node_output)
            step_msg = step_names.get(node_name, f"Running {node_name}...")
            accumulated_text += f"**{step_msg}** ✓\n\n"

            yield {
                "type": "response.output_text.delta",
                "item_id": item_id,
                "delta": f"**{step_msg}** ✓\n\n",
            }

    # Emit final formatted result
    response_text = _format_result(final_result)
    accumulated_text += response_text

    yield {
        "type": "response.output_text.delta",
        "item_id": item_id,
        "delta": response_text,
    }

    yield {
        "type": "response.output_item.done",
        "item": {
            "id": item_id,
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": accumulated_text}],
        },
    }

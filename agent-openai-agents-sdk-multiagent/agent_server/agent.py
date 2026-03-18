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
from uuid import uuid4

import mlflow
from dotenv import load_dotenv
from mlflow.genai.agent_server import invoke, stream
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)

from agent_server.sec_extraction.attribute_registry import get_registry
from agent_server.sec_extraction.config import ExtractionConfig
from agent_server.sec_extraction.workflow import extraction_workflow

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
    registry = get_registry()
    labels = registry.get_labels_for_ui()

    record = result.get("record", {})
    evaluation = result.get("evaluation", {})
    fill_rate = evaluation.get("fill_rate", 0)

    biz_name = record.get("name") or record.get("company_name") or "Unknown Company"
    in_biz = record.get("in_business", "")
    status = "Closed" if str(in_biz).lower() in ("no", "false", "closed") else "Open"

    lines = [
        f"## Extraction Results — {biz_name}",
        f"**Status:** {status} | **Fill Rate:** {fill_rate:.0%} | "
        f"**Confidence:** {evaluation.get('confidence', 0):.0%}",
        "",
    ]

    # Show only attributes that have values
    filled = {k: v for k, v in record.items() if v is not None and v != [] and v != ""}
    if filled:
        lines.append("### Extracted Attributes")
        lines.append("")
        lines.append("| Attribute | Value |")
        lines.append("|-----------|-------|")

        for field, val in filled.items():
            _, label = labels.get(field, ("📌", field.replace("_", " ").title()))
            if isinstance(val, bool):
                display = "Yes" if val else "No"
            elif isinstance(val, list):
                display = ", ".join(str(v) for v in val)
            else:
                display = str(val)
            if len(display) > 200:
                display = display[:200] + "..."
            lines.append(f"| {label} | {display} |")

    missing = evaluation.get("missing_fields", [])
    if missing:
        lines.append("")
        lines.append(f"**Missing fields ({len(missing)}):** {', '.join(missing[:20])}")
        if len(missing) > 20:
            lines.append(f"  ...and {len(missing) - 20} more")

    issues = evaluation.get("issues", [])
    if issues:
        lines.append("")
        lines.append("**Issues:** " + "; ".join(issues))

    lines.append("")
    lines.append("### Raw JSON")
    structured = {"record": record, "evaluation": evaluation}
    lines.append("```json")
    lines.append(json.dumps(structured, indent=2, default=str))
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

    # ResponsesAgentResponse requires each output message to have an "id" field
    message_id = str(uuid4())
    return ResponsesAgentResponse(
        output=[
            {
                "id": message_id,
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

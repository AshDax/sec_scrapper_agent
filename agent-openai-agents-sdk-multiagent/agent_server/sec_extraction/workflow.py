"""LangGraph Workflow — the core extraction pipeline.

Nodes:
  scraper → llm_extract → enrichment → evaluate
                                                │
                                      fill_rate < threshold?
                                      ┌─────┴─────┐
                                      ▼           ▼
                                 web_fallback    END
                                      │
                                      ▼
                                 re_evaluate
                                      │
                                      ▼
                                     END

Usage:
    from agent_server.sec_extraction.workflow import extraction_workflow
    result = extraction_workflow("Your SEC filing text...")
    print(result["record"])
    print(result["evaluation"])
"""

from __future__ import annotations

import logging
from typing import Any, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph

from agent_server.sec_extraction.config import ExtractionConfig, get_config
from agent_server.sec_extraction.tools.enrichment import enrich_record
from agent_server.sec_extraction.tools.evaluate import evaluate_record
from agent_server.sec_extraction.tools.llm_extract import llm_extract_record
from agent_server.sec_extraction.tools.scraper import scrape_attributes
from agent_server.sec_extraction.tools.text_extraction import extract_text
from agent_server.sec_extraction.tools.web_search import web_search

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Typed state — LangGraph needs this to preserve all keys across nodes
# ---------------------------------------------------------------------------


class WorkflowState(TypedDict, total=False):
    raw_text: str
    clean_text: str
    scraper_result: dict
    record: dict
    llm_record: dict
    evaluation: dict
    fill_rate: float
    web_context: list
    iteration: int


# ---------------------------------------------------------------------------
# Node functions — each returns a partial state update dict
# ---------------------------------------------------------------------------


def scraper_node(state: WorkflowState, config: RunnableConfig | None = None) -> dict:
    text = state.get("raw_text", "")
    result = scrape_attributes(text)
    logger.info("scraper: found %d fields via regex", len(result))
    return {"scraper_result": result, "clean_text": text}


def llm_extract_node(state: WorkflowState, config: RunnableConfig | None = None) -> dict:
    ext_config = _get_ext_config(config)
    clean = state.get("clean_text", "")
    if not (clean and clean.strip()):
        raw = state.get("raw_text", "")
        if raw and raw.strip():
            clean = extract_text(raw[:200_000])[:12000]
            logger.warning("clean_text was empty; used fallback from raw_text (%d chars)", len(clean))
    record, llm_record = llm_extract_record(
        text=clean,
        scraper_result=state.get("scraper_result", {}),
        context_chunks=[],
        config=ext_config,
    )
    logger.info("llm_extract: produced record with %d non-null fields (%d from LLM)",
                sum(1 for v in record.values() if v is not None and v != []),
                sum(1 for v in llm_record.values() if v is not None and v != [] and v != ""))
    return {"record": record, "llm_record": llm_record}


def enrichment_node(state: WorkflowState, config: RunnableConfig | None = None) -> dict:
    record = dict(state.get("record", {}))
    enriched = enrich_record(record, state.get("clean_text", ""))
    logger.info("enrichment: is_manufacturer=%s, is_open=%s",
                enriched.get("is_manufacturer"), enriched.get("is_open"))
    return {"record": enriched}


def evaluate_node(state: WorkflowState, config: RunnableConfig | None = None) -> dict:
    ext_config = _get_ext_config(config)
    source_text = state.get("clean_text", "")
    if not (source_text and source_text.strip()):
        raw = state.get("raw_text", "")
        if raw and raw.strip():
            source_text = extract_text(raw[:200_000])[:4000]
    evaluation = evaluate_record(
        record_dict=state.get("record", {}),
        source_text=source_text,
        config=ext_config,
        llm_record=state.get("llm_record"),
    )
    logger.info("evaluate: fill_rate=%.2f, valid=%s, llm_fields=%d",
                evaluation.get("fill_rate", 0), evaluation.get("valid"),
                len(evaluation.get("llm_filled_fields", [])))
    return {
        "evaluation": evaluation,
        "fill_rate": evaluation.get("fill_rate", 0),
    }


SEARCHABLE_FIELDS = {
    "company_name", "company_address", "company_phone", "revenue",
    "net_income", "total_assets", "location_employee_count",
    "primary_sic_code_id", "company_description", "website",
}


def web_fallback_node(state: WorkflowState, config: RunnableConfig | None = None) -> dict:
    ext_config = _get_ext_config(config)
    record = state.get("record", {})
    name = record.get("company_name") or record.get("name") or ""
    missing = state.get("evaluation", {}).get("missing_fields", [])
    useful_missing = [f for f in missing if f in SEARCHABLE_FIELDS][:3]
    query = f"{name} {' '.join(useful_missing)} SEC filing" if useful_missing else f"{name} SEC filing 10-K"
    snippets = web_search(query, ext_config)
    logger.info("web_fallback: got %d snippets for '%s'", len(snippets), query[:60])

    record, llm_record = llm_extract_record(
        text=state.get("clean_text", ""),
        scraper_result=state.get("scraper_result", {}),
        context_chunks=snippets,
        config=ext_config,
    )
    enriched = enrich_record(record, state.get("clean_text", ""))
    return {"record": enriched, "llm_record": llm_record, "web_context": snippets, "iteration": 1}


def re_evaluate_node(state: WorkflowState, config: RunnableConfig | None = None) -> dict:
    ext_config = _get_ext_config(config)
    evaluation = evaluate_record(
        record_dict=state.get("record", {}),
        source_text=state.get("clean_text", ""),
        config=ext_config,
        llm_record=state.get("llm_record"),
    )
    logger.info("re_evaluate: fill_rate=%.2f (after web fallback)", evaluation.get("fill_rate", 0))
    return {
        "evaluation": evaluation,
        "fill_rate": evaluation.get("fill_rate", 0),
    }


# ---------------------------------------------------------------------------
# Conditional edge
# ---------------------------------------------------------------------------


def should_fallback(state: WorkflowState) -> str:
    """Route to web_fallback if fill rate is below threshold and we haven't tried yet."""
    fill_rate = state.get("fill_rate", 0)
    iteration = state.get("iteration", 0)
    threshold = 0.5

    if fill_rate < threshold and iteration < 1:
        return "web_fallback"
    return "end"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def _get_ext_config(config: RunnableConfig | None) -> ExtractionConfig:
    """Extract ExtractionConfig from LangGraph configurable or use default."""
    if config and "configurable" in config:
        ext = config["configurable"].get("ext_config")
        if ext:
            return ext
    return get_config()


def build_extraction_workflow(ext_config: ExtractionConfig | None = None) -> Any:
    """Compile the LangGraph extraction pipeline."""
    graph = StateGraph(WorkflowState)

    graph.add_node("scraper", scraper_node)
    graph.add_node("llm_extract", llm_extract_node)
    graph.add_node("enrichment", enrichment_node)
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("web_fallback", web_fallback_node)
    graph.add_node("re_evaluate", re_evaluate_node)

    graph.set_entry_point("scraper")
    graph.add_edge("scraper", "llm_extract")
    graph.add_edge("llm_extract", "enrichment")
    graph.add_edge("enrichment", "evaluate")

    graph.add_conditional_edges(
        "evaluate",
        should_fallback,
        {"web_fallback": "web_fallback", "end": END},
    )
    graph.add_edge("web_fallback", "re_evaluate")
    graph.add_edge("re_evaluate", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------


def extraction_workflow(
    document: str,
    config: ExtractionConfig | None = None,
) -> dict:
    """Run the full extraction pipeline on a single document.

    Returns dict with: record, scraper_result, evaluation, fill_rate,
    and extra_attributes (scraper fields not in the registry).
    """
    cfg = config or get_config()
    workflow = build_extraction_workflow(cfg)

    initial_state: WorkflowState = {
        "raw_text": document,
        "clean_text": "",
        "scraper_result": {},
        "record": {},
        "evaluation": {},
        "fill_rate": 0.0,
        "web_context": [],
        "iteration": 0,
    }

    result = workflow.invoke(
        initial_state,
        config={"configurable": {"ext_config": cfg}},
    )

    from agent_server.sec_extraction.attribute_registry import get_registry
    registry_keys = set(get_registry().attribute_names)
    scraper_result = result.get("scraper_result", {})
    extra_attributes = {
        k: v for k, v in scraper_result.items()
        if k not in registry_keys and v is not None
    }

    return {
        "record": result.get("record", {}),
        "scraper_result": scraper_result,
        "extra_attributes": extra_attributes,
        "evaluation": result.get("evaluation", {}),
        "fill_rate": result.get("fill_rate", 0.0),
    }

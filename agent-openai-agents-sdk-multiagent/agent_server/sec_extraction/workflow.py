"""LangGraph Workflow — the core extraction pipeline.

Nodes:
  text_extract → scraper → retrieval → llm_extract → enrichment → evaluate
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
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph

from agent_server.sec_extraction.config import ExtractionConfig, get_config
from agent_server.sec_extraction.schemas import ExtractionState
from agent_server.sec_extraction.tools.enrichment import enrich_record
from agent_server.sec_extraction.tools.evaluate import evaluate_record
from agent_server.sec_extraction.tools.llm_extract import llm_extract_record
from agent_server.sec_extraction.tools.retrieval import retrieve_context
from agent_server.sec_extraction.tools.scraper import scrape_attributes
from agent_server.sec_extraction.tools.text_extraction import extract_text
from agent_server.sec_extraction.tools.web_search import web_search

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Node functions — each returns a partial state update dict
# ---------------------------------------------------------------------------


def text_extract_node(state: dict, config: RunnableConfig | None = None) -> dict:
    raw = state.get("raw_text", "")
    clean = extract_text(raw)
    logger.info("text_extract: %d chars → %d chars", len(raw), len(clean))
    return {"clean_text": clean}


def scraper_node(state: dict, config: RunnableConfig | None = None) -> dict:
    clean = state.get("clean_text", "")
    result = scrape_attributes(clean)
    logger.info("scraper: found %d fields via regex", len(result))
    return {"scraper_result": result}


def retrieval_node(state: dict, config: RunnableConfig | None = None) -> dict:
    ext_config = _get_ext_config(config)
    clean = state.get("clean_text", "")
    name = state.get("scraper_result", {}).get("business_name", "")
    query = f"{name} SEC filing business description employees revenue" if name else clean[:200]
    chunks = retrieve_context(query, ext_config)
    logger.info("retrieval: got %d chunks from vector search", len(chunks))
    return {"retrieval_context": chunks}


def llm_extract_node(state: dict, config: RunnableConfig | None = None) -> dict:
    ext_config = _get_ext_config(config)
    record = llm_extract_record(
        text=state.get("clean_text", ""),
        scraper_result=state.get("scraper_result", {}),
        context_chunks=state.get("retrieval_context", []),
        config=ext_config,
    )
    logger.info("llm_extract: produced record with %d non-null fields",
                sum(1 for v in record.values() if v is not None and v != []))
    return {"record": record}


def enrichment_node(state: dict, config: RunnableConfig | None = None) -> dict:
    record = dict(state.get("record", {}))
    enriched = enrich_record(record, state.get("clean_text", ""))
    logger.info("enrichment: is_manufacturer=%s, is_open=%s",
                enriched.get("is_manufacturer"), enriched.get("is_open"))
    return {"record": enriched}


def evaluate_node(state: dict, config: RunnableConfig | None = None) -> dict:
    ext_config = _get_ext_config(config)
    evaluation = evaluate_record(
        record_dict=state.get("record", {}),
        source_text=state.get("clean_text", ""),
        config=ext_config,
    )
    logger.info("evaluate: fill_rate=%.2f, valid=%s",
                evaluation.get("fill_rate", 0), evaluation.get("valid"))
    return {
        "evaluation": evaluation,
        "fill_rate": evaluation.get("fill_rate", 0),
    }


def web_fallback_node(state: dict, config: RunnableConfig | None = None) -> dict:
    ext_config = _get_ext_config(config)
    name = state.get("record", {}).get("business_name", "")
    missing = state.get("evaluation", {}).get("missing_fields", [])
    query = f"{name} {' '.join(missing[:3])} SEC filing"
    snippets = web_search(query, ext_config)
    logger.info("web_fallback: got %d snippets for '%s'", len(snippets), query[:60])

    all_context = state.get("retrieval_context", []) + snippets
    record = llm_extract_record(
        text=state.get("clean_text", ""),
        scraper_result=state.get("scraper_result", {}),
        context_chunks=all_context,
        config=ext_config,
    )
    enriched = enrich_record(record, state.get("clean_text", ""))
    return {"record": enriched, "web_context": snippets, "iteration": 1}


def re_evaluate_node(state: dict, config: RunnableConfig | None = None) -> dict:
    ext_config = _get_ext_config(config)
    evaluation = evaluate_record(
        record_dict=state.get("record", {}),
        source_text=state.get("clean_text", ""),
        config=ext_config,
    )
    logger.info("re_evaluate: fill_rate=%.2f (after web fallback)", evaluation.get("fill_rate", 0))
    return {
        "evaluation": evaluation,
        "fill_rate": evaluation.get("fill_rate", 0),
    }


# ---------------------------------------------------------------------------
# Conditional edge
# ---------------------------------------------------------------------------


def should_fallback(state: dict) -> str:
    """Route to web_fallback if fill rate is below threshold and we haven't tried yet."""
    fill_rate = state.get("fill_rate", 0)
    iteration = state.get("iteration", 0)
    threshold = 0.5  # overridden at runtime via config if needed

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
    """Compile the LangGraph extraction pipeline.

    Returns a compiled StateGraph that accepts ExtractionState.
    """
    graph = StateGraph(dict)

    graph.add_node("text_extract", text_extract_node)
    graph.add_node("scraper", scraper_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("llm_extract", llm_extract_node)
    graph.add_node("enrichment", enrichment_node)
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("web_fallback", web_fallback_node)
    graph.add_node("re_evaluate", re_evaluate_node)

    graph.set_entry_point("text_extract")
    graph.add_edge("text_extract", "scraper")
    graph.add_edge("scraper", "retrieval")
    graph.add_edge("retrieval", "llm_extract")
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

    Args:
        document: Raw SEC filing text or HTML.
        config:   Pipeline configuration (uses defaults if None).

    Returns:
        Dict with keys: record, evaluation, fill_rate, and more.
    """
    cfg = config or get_config()
    workflow = build_extraction_workflow(cfg)

    initial_state = {
        "raw_text": document,
        "clean_text": "",
        "scraper_result": {},
        "retrieval_context": [],
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
    return result

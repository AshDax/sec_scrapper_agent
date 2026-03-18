"""ReAct Supervisor Agent — alternative to the fixed LangGraph workflow.

Instead of a linear pipeline, the supervisor uses a ReAct loop to decide
which tool to call next based on observations.  This gives the agent more
flexibility but less predictability.

Use the fixed workflow (workflow.py) for batch/production runs, and the
supervisor for interactive exploration.

Usage:
    from agent_server.sec_extraction.supervisor import build_supervisor
    from agent_server.sec_extraction.config import get_config
    supervisor = build_supervisor(get_config())
    result = supervisor.invoke({"input": "Extract from this SEC filing: ..."})
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from langchain_core.tools import tool

if TYPE_CHECKING:
    from agent_server.sec_extraction.config import ExtractionConfig

from agent_server.sec_extraction.tools.enrichment import enrich_record
from agent_server.sec_extraction.tools.evaluate import compute_fill_rate, evaluate_record
from agent_server.sec_extraction.tools.llm_extract import llm_extract_record
from agent_server.sec_extraction.tools.scraper import scrape_attributes
from agent_server.sec_extraction.tools.text_extraction import extract_text
from agent_server.sec_extraction.tools.web_search import web_search

# ---------------------------------------------------------------------------
# Tools (wrapped for LangChain tool interface)
# ---------------------------------------------------------------------------

_config_ref: ExtractionConfig | None = None


@tool
def clean_text(raw_text: str) -> str:
    """Parse raw SEC filing text/HTML into clean plaintext."""
    return extract_text(raw_text)


@tool
def regex_scrape(text: str) -> str:
    """Extract attributes using regex patterns. Returns JSON dict of found fields."""
    result = scrape_attributes(text)
    return json.dumps(result, indent=2)


@tool
def vector_search(query: str) -> str:
    """Search SEC filing Vector Search index for relevant context. (Currently disabled — returns empty.)"""
    return "No results found. (Retrieval is disabled.)"


@tool
def extract_with_llm(text: str, scraper_hints: str, context: str) -> str:
    """Use LLM to extract a full BusinessRecord from text + context.
    scraper_hints and context should be JSON strings.
    """
    from agent_server.sec_extraction.config import get_config
    cfg = _config_ref or get_config()
    try:
        hints = json.loads(scraper_hints) if scraper_hints else {}
    except json.JSONDecodeError:
        hints = {}
    chunks = context.split("\n---\n") if context else []
    record, _ = llm_extract_record(text[:8000], hints, chunks, cfg)
    return json.dumps(record, indent=2)


@tool
def enrich(record_json: str, source_text: str) -> str:
    """Add derived fields (is_manufacturer, is_open, industry from NAICS)."""
    try:
        record = json.loads(record_json)
    except json.JSONDecodeError:
        return record_json
    enriched = enrich_record(record, source_text)
    return json.dumps(enriched, indent=2)


@tool
def evaluate(record_json: str, source_text: str) -> str:
    """Validate extraction quality and compute fill rate."""
    from agent_server.sec_extraction.config import get_config
    cfg = _config_ref or get_config()
    try:
        record = json.loads(record_json)
    except json.JSONDecodeError:
        return json.dumps({"valid": False, "fill_rate": 0, "issues": ["Invalid JSON"]})
    result = evaluate_record(record, source_text, cfg)
    return json.dumps(result, indent=2)


@tool
def search_web(query: str) -> str:
    """Search the web for supplementary business information."""
    from agent_server.sec_extraction.config import get_config
    cfg = _config_ref or get_config()
    snippets = web_search(query, cfg)
    return "\n---\n".join(snippets) if snippets else "No results found."


SUPERVISOR_TOOLS = [
    clean_text, regex_scrape, vector_search,
    extract_with_llm, enrich, evaluate, search_web,
]

SUPERVISOR_SYSTEM = """You are an SEC filing extraction supervisor.

Your job: extract structured business attributes from SEC filings.

Strategy (follow this order):
1. Call `clean_text` on the raw filing first.
2. Call `regex_scrape` on the clean text for quick pattern matches.
3. Call `vector_search` with the company name to get context from the filings database.
4. Call `extract_with_llm` passing the clean text, scraper hints, and retrieved context.
5. Call `enrich` to add derived fields (is_manufacturer, is_open).
6. Call `evaluate` to check extraction quality.
7. If fill_rate < 0.5, call `search_web` for missing info, then re-extract and re-evaluate.

Return the final enriched BusinessRecord JSON as your answer.
"""


def build_supervisor(config: ExtractionConfig | None = None):
    """Build a ReAct supervisor agent with all extraction tools.

    Returns a LangGraph compiled graph that accepts {"messages": [...]}.
    """
    global _config_ref

    from langgraph.prebuilt import create_react_agent
    from agent_server.sec_extraction.config import get_config

    cfg = config or get_config()
    _config_ref = cfg

    llm = cfg.get_llm(temperature=0)

    return create_react_agent(
        model=llm,
        tools=SUPERVISOR_TOOLS,
        prompt=SUPERVISOR_SYSTEM,
    )

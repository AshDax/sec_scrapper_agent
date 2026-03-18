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
import os
import tempfile
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import mlflow
import pandas as pd
from langchain_core.tools import tool

if TYPE_CHECKING:
    from agent_server.sec_extraction.config import ExtractionConfig

from agent_server.sec_extraction.tools.enrichment import enrich_record
from agent_server.sec_extraction.tools.evaluate import compute_fill_rate, evaluate_record
from agent_server.sec_extraction.tools.llm_extract import llm_extract_record
from agent_server.sec_extraction.tools.scraper import scrape_attributes
from agent_server.sec_extraction.tools.section_filtering import filter_sections as _filter_sections
from agent_server.sec_extraction.tools.text_extraction import extract_text
from agent_server.sec_extraction.tools.web_search import web_search

# ---------------------------------------------------------------------------
# MLflow setup
# ---------------------------------------------------------------------------

mlflow.set_experiment("sec_extraction")

# module-level run tracker so tools can contribute to the same run
_active_run_context: dict = {
    "run_id"          : None,
    "company"         : None,
    "scraper_output"  : {},
    "xbrl_output"     : {},
    "filtered_text"   : "",
    "sections_kept"   : 0,
    "sections_skipped": 0,
    "input_char_count": 0,
    "web_search_used" : False,
    "llm_calls"       : 0,
    "evaluate_results": {},
}


def _reset_run_context():
    """Reset the module-level tracker for a new extraction run."""
    _active_run_context.update({
        "run_id"          : None,
        "company"         : None,
        "scraper_output"  : {},
        "xbrl_output"     : {},
        "filtered_text"   : "",
        "sections_kept"   : 0,
        "sections_skipped": 0,
        "input_char_count": 0,
        "web_search_used" : False,
        "llm_calls"       : 0,
        "evaluate_results": {},
    })


def _log_run_to_mlflow(final_record: dict):
    """
    Log the completed extraction run to MLflow.
    Called once evaluate() confirms a good result.
    """
    company   = _active_run_context.get("company", "unknown")
    run_name  = f"sec_extraction_{company}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    eval_data = _active_run_context.get("evaluate_results", {})

    with mlflow.start_run(run_name=run_name) as run:

        # ── params ────────────────────────────────────────────────
        mlflow.log_params({
            "company"              : company,
            "min_chars_filter"     : 500,
            "max_chars_truncate"   : 8000,
            "extraction_model"     : "databricks-claude-sonnet-4-6",
            "web_search_used"      : _active_run_context["web_search_used"],
            "llm_calls"            : _active_run_context["llm_calls"],
        })

        # ── metrics ───────────────────────────────────────────────
        input_chars    = _active_run_context["input_char_count"]
        filtered_chars = len(_active_run_context["filtered_text"])
        fields_filled  = sum(1 for v in final_record.values() if v is not None)
        fields_total   = len(final_record)

        mlflow.log_metrics({
            "input_char_count"    : input_chars,
            "filtered_char_count" : filtered_chars,
            "char_reduction_pct"  : round(
                (input_chars - filtered_chars) / max(input_chars, 1) * 100, 2
            ),
            "sections_kept"       : _active_run_context["sections_kept"],
            "sections_skipped"    : _active_run_context["sections_skipped"],
            "total_fields_filled" : fields_filled,
            "total_fields_possible": fields_total,
            "fill_rate_pct"       : round(fields_filled / max(fields_total, 1) * 100, 2),
            "eval_fill_rate"      : eval_data.get("fill_rate", 0),
            "eval_confidence"     : eval_data.get("confidence", 0),
            "eval_issues_count"   : len(eval_data.get("issues", [])),
        })

        # ── tags ─────────────────────────────────────────────────
        mlflow.set_tags({
            "company"        : company,
            "pipeline_stage" : "full_extraction",
            "status"         : "success",
            "run_timestamp"  : datetime.now(timezone.utc).isoformat(),
        })

        # ── artifacts ────────────────────────────────────────────
        with tempfile.TemporaryDirectory() as tmp:

            # 1. filtered section text
            filt_path = os.path.join(tmp, "filtered_sections.txt")
            with open(filt_path, "w") as f:
                f.write(_active_run_context["filtered_text"])
            mlflow.log_artifact(filt_path, artifact_path="outputs")

            # 2. scraper output
            scraper_path = os.path.join(tmp, "scraper_output.json")
            with open(scraper_path, "w") as f:
                json.dump(_active_run_context["scraper_output"], f, indent=2, default=str)
            mlflow.log_artifact(scraper_path, artifact_path="outputs")

            # 3. final record
            record_path = os.path.join(tmp, "final_record.json")
            with open(record_path, "w") as f:
                json.dump(final_record, f, indent=2, default=str)
            mlflow.log_artifact(record_path, artifact_path="outputs")

            # 4. evaluation results
            eval_path = os.path.join(tmp, "evaluation.json")
            with open(eval_path, "w") as f:
                json.dump(eval_data, f, indent=2, default=str)
            mlflow.log_artifact(eval_path, artifact_path="outputs")

            # 5. field coverage CSV
            coverage_rows = []
            for field, value in final_record.items():
                coverage_rows.append({
                    "field"  : field,
                    "value"  : str(value) if value is not None else None,
                    "filled" : value is not None,
                    "source" : (
                        "scraper" if field in _active_run_context["scraper_output"]
                                  and _active_run_context["scraper_output"].get(field) is not None
                        else "llm"
                    ),
                })
            cov_path = os.path.join(tmp, "field_coverage.csv")
            pd.DataFrame(coverage_rows).to_csv(cov_path, index=False)
            mlflow.log_artifact(cov_path, artifact_path="outputs")

        _active_run_context["run_id"] = run.info.run_id
        print(f"✓ MLflow run logged: {run.info.run_id}")
        print(f"  Company    : {company}")
        print(f"  Fill rate  : {fields_filled}/{fields_total} fields")
        print(f"  Eval score : {eval_data.get('fill_rate', 0):.2f}")

        return run.info.run_id


# ---------------------------------------------------------------------------
# Config ref
# ---------------------------------------------------------------------------

_config_ref: ExtractionConfig | None = None


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool
def clean_text(raw_text: str) -> str:
    """Parse raw SEC filing text/HTML into clean plaintext."""
    # track input size for MLflow
    _active_run_context["input_char_count"] = len(raw_text)
    return extract_text(raw_text)


@tool
def regex_scrape(text: str) -> str:
    """Extract attributes using regex patterns. Returns JSON dict of found fields."""
    result = scrape_attributes(text)
    # track scraper output for MLflow artifact
    _active_run_context["scraper_output"] = result

    # try to extract company name for run naming
    company = (
        result.get("company_name")
        or result.get("name")
        or "unknown"
    )
    if company and _active_run_context["company"] is None:
        _active_run_context["company"] = str(company)[:50]

    return json.dumps(result, indent=2)


@tool
def filter_sections(text: str) -> str:
    """Filters useful sections from a SEC 10-K filing which probably contain extractable features."""
    result = _filter_sections(text)

    # track filtered text and section counts for MLflow
    _active_run_context["filtered_text"] = result

    # count kept vs skipped from the section markers in output
    kept    = result.count("=== Item")
    _active_run_context["sections_kept"]    = kept
    # approximate skipped from typical 10-K structure
    _active_run_context["sections_skipped"] = max(0, 16 - kept)

    return result


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

    # track LLM call count for MLflow
    _active_run_context["llm_calls"] += 1

    try:
        hints = json.loads(scraper_hints) if scraper_hints else {}
    except json.JSONDecodeError:
        hints = {}

    chunks = context.split("\n---\n") if context else []
    record = llm_extract_record(text[:8000], hints, chunks, cfg)
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
    """Validate extraction quality, compute fill rate, and log to MLflow."""
    from agent_server.sec_extraction.config import get_config
    cfg = _config_ref or get_config()

    try:
        record = json.loads(record_json)
    except json.JSONDecodeError:
        return json.dumps({
            "valid"    : False,
            "fill_rate": 0,
            "issues"   : ["Invalid JSON"]
        })

    result = evaluate_record(record, source_text, cfg)

    # store eval results for MLflow
    _active_run_context["evaluate_results"] = result

    # ── log to MLflow when evaluation is complete ─────────────────
    # this is the natural end of one extraction run
    try:
        _log_run_to_mlflow(record)
    except Exception as e:
        # never let MLflow logging break the agent pipeline
        print(f"  MLflow logging warning: {e}")

    return json.dumps(result, indent=2)


@tool
def search_web(query: str) -> str:
    """Search the web for supplementary business information."""
    from agent_server.sec_extraction.config import get_config
    cfg = _config_ref or get_config()

    # track that web search was used
    _active_run_context["web_search_used"] = True

    snippets = web_search(query, cfg)
    return "\n---\n".join(snippets) if snippets else "No results found."


SUPERVISOR_TOOLS = [
    clean_text, regex_scrape, filter_sections, vector_search,
    extract_with_llm, enrich, evaluate, search_web,
]

SUPERVISOR_SYSTEM = """You are an SEC filing extraction supervisor.

Your job: extract structured business attributes from SEC filings.

Strategy (follow this order):
1. Call `clean_text` on the raw filing first.
2. Call `regex_scrape` on the clean text for quick pattern matches.
3. Call `filter_sections` to get only the important 10-K sections.
4. Call `vector_search` with the company name to get context from the filings database.
5. Call `extract_with_llm` passing the clean text, scraper hints, and retrieved context.
6. Call `enrich` to add derived fields (is_manufacturer, is_open).
7. Call `evaluate` to check extraction quality — this also logs results to MLflow.
8. If fill_rate < 0.5, call `search_web` for missing info, then re-extract and re-evaluate.

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

    # reset context for each new supervisor build
    _reset_run_context()

    llm = cfg.get_llm(temperature=0)

    return create_react_agent(
        model  = llm,
        tools  = SUPERVISOR_TOOLS,
        prompt = SUPERVISOR_SYSTEM,
    )
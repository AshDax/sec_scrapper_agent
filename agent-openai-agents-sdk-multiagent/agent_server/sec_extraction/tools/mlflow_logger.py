"""Centralised MLflow logger for the SEC extraction agent pipeline.

Drop-in logger that any agent step can call without caring about
MLflow run management.  A single run is shared across all steps in
one extraction — open it at the start, close it at the end.

Usage (in any agent tool or workflow node):

    from agent_server.sec_extraction.tools.mlflow_logger import PipelineLogger

    # Start a run at the top of an extraction
    logger = PipelineLogger(company="AAPL")
    logger.start()

    # Each agent step logs its own output
    logger.log_scraper(scraper_output)
    logger.log_sections(filtered_text, sections_kept=8, sections_skipped=23)
    logger.log_llm_call(step="llm_extract", input_chars=8000, output_fields=12)
    logger.log_web_search(query="AAPL employees", results_count=5)
    logger.log_evaluation(eval_result_dict)
    logger.log_final_record(final_record_dict)

    # End the run — writes all artifacts and closes MLflow
    logger.end()

Designed to never raise — every method is wrapped in try/except so
MLflow issues never break the agent pipeline.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Experiment name — all agents log to the same experiment
# ---------------------------------------------------------------------------

EXPERIMENT_NAME = "sec_extraction_pipeline"


# ---------------------------------------------------------------------------
# PipelineLogger
# ---------------------------------------------------------------------------

class PipelineLogger:
    """Shared MLflow run for one end-to-end extraction.

    One instance per company/document.  Pass it through your pipeline
    or store it in LangGraph state so every node can call it.
    """

    def __init__(self, company: str = "unknown", run_name: str | None = None):
        self.company   = company
        self.run_name  = run_name or f"sec_{company}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self._run      = None          # mlflow.ActiveRun
        self._run_id   = None
        self._tmpdir   = None          # tempfile.TemporaryDirectory

        # accumulated data — flushed to artifacts at end()
        self._scraper_output:   dict = {}
        self._filtered_text:    str  = ""
        self._sections_kept:    int  = 0
        self._sections_skipped: int  = 0
        self._llm_calls:        list = []   # list of dicts
        self._web_searches:     list = []   # list of dicts
        self._eval_results:     dict = {}
        self._final_record:     dict = {}
        self._input_chars:      int  = 0
        self._metrics:          dict = {}   # extra metrics callers can push

    # ── lifecycle ─────────────────────────────────────────────────────────

    def start(self) -> "PipelineLogger":
        """Open an MLflow run.  Safe to call multiple times (no-op if already open)."""
        if self._run is not None:
            return self
        try:
            import mlflow
            mlflow.set_experiment(EXPERIMENT_NAME)
            self._run    = mlflow.start_run(run_name=self.run_name)
            self._run_id = self._run.info.run_id
            self._tmpdir = tempfile.mkdtemp()
            mlflow.set_tags({
                "company"        : self.company,
                "pipeline"       : "sec_extraction",
                "run_timestamp"  : datetime.now(timezone.utc).isoformat(),
            })
            logger.info("MLflow run started: %s", self._run_id)
        except Exception as exc:
            logger.warning("MLflow start failed (logging disabled): %s", exc)
        return self

    def end(self, status: str = "success") -> str | None:
        """Flush all artifacts, log final metrics, and close the run.

        Returns the run_id or None if MLflow was unavailable.
        """
        if self._run is None:
            return None
        try:
            import mlflow
            self._flush_metrics()
            self._flush_artifacts()
            mlflow.set_tag("status", status)
            mlflow.end_run()
            logger.info("MLflow run ended: %s (status=%s)", self._run_id, status)
        except Exception as exc:
            logger.warning("MLflow end failed: %s", exc)
        finally:
            self._run = None
        return self._run_id

    def fail(self, reason: str = "error") -> None:
        """End the run with a failed status."""
        try:
            import mlflow
            mlflow.set_tag("failure_reason", reason[:200])
        except Exception:
            pass
        self.end(status="failed")

    # ── per-step logging methods ───────────────────────────────────────────

    def log_input(self, raw_text: str) -> None:
        """Call at the very start — logs raw input size."""
        self._input_chars = len(raw_text)
        self._safe_log_metric("input_char_count", self._input_chars)

    def log_scraper(self, scraper_output: dict) -> None:
        """Call after the regex/XBRL scraper step."""
        self._scraper_output = scraper_output
        fields_found = sum(1 for v in scraper_output.values() if v is not None)
        self._safe_log_metric("scraper_fields_found", fields_found)

        # try to capture company name for tagging
        company = (
            scraper_output.get("company_name")
            or scraper_output.get("name")
            or self.company
        )
        if company and company != "unknown":
            self.company = str(company)[:80]
            self._safe_set_tag("company", self.company)

    def log_sections(
        self,
        filtered_text:    str,
        sections_kept:    int = 0,
        sections_skipped: int = 0,
    ) -> None:
        """Call after section_filtering — logs section stats."""
        self._filtered_text    = filtered_text
        self._sections_kept    = sections_kept
        self._sections_skipped = sections_skipped
        filtered_chars         = len(filtered_text)

        self._safe_log_metric("filtered_char_count",  filtered_chars)
        self._safe_log_metric("sections_kept",         sections_kept)
        self._safe_log_metric("sections_skipped",      sections_skipped)

        if self._input_chars > 0:
            reduction = (self._input_chars - filtered_chars) / self._input_chars * 100
            self._safe_log_metric("char_reduction_pct", round(reduction, 2))

    def log_llm_call(
        self,
        step:          str,
        input_chars:   int  = 0,
        output_fields: int  = 0,
        model:         str  = "",
        success:       bool = True,
    ) -> None:
        """Call each time the LLM is invoked."""
        entry = {
            "step"          : step,
            "input_chars"   : input_chars,
            "output_fields" : output_fields,
            "model"         : model,
            "success"       : success,
            "timestamp"     : datetime.now(timezone.utc).isoformat(),
        }
        self._llm_calls.append(entry)
        self._safe_log_metric("llm_call_count", len(self._llm_calls))

    def log_web_search(self, query: str, results_count: int = 0) -> None:
        """Call when the web fallback agent fires."""
        entry = {
            "query"         : query,
            "results_count" : results_count,
            "timestamp"     : datetime.now(timezone.utc).isoformat(),
        }
        self._web_searches.append(entry)
        self._safe_log_metric("web_search_count",        len(self._web_searches))
        self._safe_log_metric("web_search_results_total",
                              sum(e["results_count"] for e in self._web_searches))
        self._safe_set_tag("web_search_used", "true")

    def log_evaluation(self, eval_result: dict) -> None:
        """Call after evaluate_record() returns."""
        self._eval_results = eval_result
        self._safe_log_metric("fill_rate_pct",
                              round(eval_result.get("fill_rate", 0) * 100, 2))
        self._safe_log_metric("eval_confidence",
                              eval_result.get("confidence", 0))
        self._safe_log_metric("eval_issues_count",
                              len(eval_result.get("issues", [])))
        self._safe_set_tag("eval_valid",
                           str(eval_result.get("valid", False)))

    def log_final_record(self, record: dict) -> None:
        """Call with the final enriched record before end()."""
        self._final_record = record
        fields_filled   = sum(1 for v in record.values() if v is not None and v != [])
        fields_total    = len(record)
        self._safe_log_metric("total_fields_filled",    fields_filled)
        self._safe_log_metric("total_fields_possible",  fields_total)

    def log_metric(self, key: str, value: float) -> None:
        """Log an arbitrary extra metric from any agent step."""
        self._metrics[key] = value
        self._safe_log_metric(key, value)

    def log_tag(self, key: str, value: str) -> None:
        """Log an arbitrary tag from any agent step."""
        self._safe_set_tag(key, value)

    # ── property ──────────────────────────────────────────────────────────

    @property
    def run_id(self) -> str | None:
        return self._run_id

    @property
    def is_active(self) -> bool:
        return self._run is not None

    # ── internal helpers ──────────────────────────────────────────────────

    def _safe_log_metric(self, key: str, value: float) -> None:
        if self._run is None:
            return
        try:
            import mlflow
            mlflow.log_metric(key, value)
        except Exception as exc:
            logger.debug("MLflow metric log failed (%s): %s", key, exc)

    def _safe_set_tag(self, key: str, value: str) -> None:
        if self._run is None:
            return
        try:
            import mlflow
            mlflow.set_tag(key, str(value)[:500])
        except Exception as exc:
            logger.debug("MLflow tag set failed (%s): %s", key, exc)

    def _flush_metrics(self) -> None:
        """Log any final derived metrics before closing."""
        try:
            import mlflow
            if self._final_record and self._eval_results:
                fields_filled  = sum(
                    1 for v in self._final_record.values()
                    if v is not None and v != []
                )
                fields_total   = len(self._final_record)
                mlflow.log_metric("final_fill_rate_pct",
                                  round(fields_filled / max(fields_total, 1) * 100, 2))
        except Exception as exc:
            logger.debug("MLflow flush metrics failed: %s", exc)

    def _flush_artifacts(self) -> None:
        """Write all accumulated data to MLflow artifacts."""
        if self._run is None or not self._tmpdir:
            return
        try:
            import mlflow

            artifacts = {
                "scraper_output.json"  : json.dumps(self._scraper_output,
                                                     indent=2, default=str),
                "final_record.json"    : json.dumps(self._final_record,
                                                     indent=2, default=str),
                "evaluation.json"      : json.dumps(self._eval_results,
                                                     indent=2, default=str),
                "llm_calls.json"       : json.dumps(self._llm_calls,
                                                     indent=2, default=str),
                "web_searches.json"    : json.dumps(self._web_searches,
                                                     indent=2, default=str),
                "filtered_sections.txt": self._filtered_text,
            }

            for filename, content in artifacts.items():
                path = os.path.join(self._tmpdir, filename)
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content)
                mlflow.log_artifact(path, artifact_path="outputs")

            # field coverage CSV
            if self._final_record:
                rows = []
                for field, value in self._final_record.items():
                    scraper_had = self._scraper_output.get(field) is not None
                    rows.append({
                        "field"       : field,
                        "value"       : str(value) if value is not None else None,
                        "filled"      : value is not None and value != [],
                        "source"      : "scraper" if scraper_had else "llm",
                    })
                cov_path = os.path.join(self._tmpdir, "field_coverage.csv")
                pd.DataFrame(rows).to_csv(cov_path, index=False)
                mlflow.log_artifact(cov_path, artifact_path="outputs")

        except Exception as exc:
            logger.warning("MLflow artifact flush failed: %s", exc)


# ---------------------------------------------------------------------------
# Module-level convenience — one logger per process
# Used by supervisor.py which runs one company at a time
# ---------------------------------------------------------------------------

_active_logger: PipelineLogger | None = None


def get_logger(company: str = "unknown") -> PipelineLogger:
    """Return (or create) the module-level active logger.

    Call this from any tool when you don't want to pass the logger explicitly.
    """
    global _active_logger
    if _active_logger is None or not _active_logger.is_active:
        _active_logger = PipelineLogger(company=company)
        _active_logger.start()
    return _active_logger


def reset_logger(company: str = "unknown") -> PipelineLogger:
    """Force-create a fresh logger — call this at the start of each new extraction."""
    global _active_logger
    if _active_logger and _active_logger.is_active:
        _active_logger.end()
    _active_logger = PipelineLogger(company=company)
    _active_logger.start()
    return _active_logger


def end_logger(status: str = "success") -> str | None:
    """End the module-level logger and return the run_id."""
    global _active_logger
    if _active_logger:
        return _active_logger.end(status=status)
    return None
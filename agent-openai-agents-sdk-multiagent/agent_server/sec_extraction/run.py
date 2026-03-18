"""Batch runner — process documents with optional MLflow logging and Delta write.

Usage:
    from agent_server.sec_extraction.run import run_extraction

    results = run_extraction(
        documents=["filing1...", "filing2..."],
        use_mlflow=True,
        delta_table="catalog.schema.business_records",
    )
"""
"""Batch runner — process documents with optional MLflow logging and Delta write."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from agent_server.sec_extraction.config import ExtractionConfig, get_config
from agent_server.sec_extraction.workflow import extraction_workflow
from agent_server.sec_extraction.tools.mlflow_logger import reset_logger, end_logger

logger = logging.getLogger(__name__)


def run_extraction(
    documents: list[str],
    config: ExtractionConfig | None = None,
    use_mlflow: bool = True,
    delta_table: str | None = None,
) -> list[dict[str, Any]]:
    """Run the extraction pipeline on a batch of documents.

    Args:
        documents:   List of raw SEC filing texts.
        config:      Pipeline config (uses env-based defaults if None).
        use_mlflow:  Whether to log metrics/artifacts to MLflow.
        delta_table: Optional Delta table name to write results to.

    Returns:
        List of result dicts, each with 'record' and 'evaluation'.
    """
    cfg = config or get_config()
    delta_table = delta_table or cfg.delta_table
    results: list[dict[str, Any]] = []

    # ── parent MLflow run — wraps the whole batch ─────────────────────────
    parent_run = None
    if use_mlflow:
        try:
            import mlflow
            parent_run = mlflow.start_run(run_name="sec-extraction-batch")
            mlflow.log_param("total_documents", len(documents))
            mlflow.log_param("llm_model",       cfg.llm_model)
        except Exception as exc:
            logger.warning("MLflow parent run failed: %s", exc)
            use_mlflow = False

    total_fill = 0.0
    successful = 0

    for i, doc in enumerate(documents):
        logger.info("Processing document %d/%d ...", i + 1, len(documents))
        t0 = time.time()

        # ── reset per-document logger BEFORE workflow runs ─────────────────
        if use_mlflow:
            reset_logger(company=f"doc_{i}")

        try:
            result  = extraction_workflow(doc, cfg)
            elapsed = time.time() - t0

            fill_rate  = result.get("fill_rate", 0)
            confidence = result.get("evaluation", {}).get("confidence", 0)
            is_valid   = result.get("evaluation", {}).get("valid", False)

            total_fill += fill_rate
            if is_valid:
                successful += 1

            result["processing_time_s"] = round(elapsed, 2)
            results.append(result)

            # log per-doc metrics on the PARENT run directly
            if use_mlflow:
                import mlflow
                mlflow.log_metric(f"fill_rate_doc_{i}",       fill_rate,  step=i)
                mlflow.log_metric(f"confidence_doc_{i}",      confidence, step=i)
                mlflow.log_metric(f"processing_time_doc_{i}", elapsed,    step=i)

        except Exception as exc:
            logger.error("Document %d failed: %s", i, exc)
            results.append({"error": str(exc), "record": {}, "evaluation": {}})

        finally:
            # always end the per-document child logger
            if use_mlflow:
                end_logger()

    # ── close parent run with batch-level summary ─────────────────────────
    if use_mlflow and parent_run:
        try:
            import mlflow
            mlflow.log_metric("avg_fill_rate",          total_fill / max(len(documents), 1))
            mlflow.log_metric("successful_extractions",  successful)
            mlflow.log_metric("failed_extractions",      len(documents) - successful)
            mlflow.log_metric("success_rate_pct",
                              round(successful / max(len(documents), 1) * 100, 2))

            # log all records as a single artifact on the parent run
            mlflow.log_text(
                json.dumps(
                    [r.get("record", {}) for r in results],
                    indent=2, default=str
                ),
                "extraction_results.json",
            )
            mlflow.end_run()
            logger.info("MLflow parent run closed — avg fill rate: %.2f", 
                        total_fill / max(len(documents), 1))
        except Exception as exc:
            logger.warning("MLflow parent run close failed: %s", exc)

    if delta_table:
        _write_to_delta(results, delta_table)

    return results


def _write_to_delta(results: list[dict], table_name: str):
    """Write BusinessRecord dicts to a Delta table via Spark."""
    try:
        from pyspark.sql import SparkSession
        spark   = SparkSession.builder.getOrCreate()
        records = [r.get("record", {}) for r in results if r.get("record")]
        if not records:
            logger.warning("No records to write to Delta table")
            return
        df = spark.createDataFrame(records)
        df.write.mode("append").saveAsTable(table_name)
        logger.info("Wrote %d records to %s", len(records), table_name)
    except ImportError:
        logger.warning("PySpark not available — skipping Delta write.")
    except Exception as exc:
        logger.error("Delta write failed: %s", exc)
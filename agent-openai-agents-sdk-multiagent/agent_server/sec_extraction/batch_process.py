"""Batch processor — reads raw SEC filings from a Databricks Volume,
runs the extraction pipeline on each, and writes results to a Delta table.

Supports:
  - Reading files from Databricks Volumes via SDK
  - Progress tracking with resume capability (skips already-processed files)
  - Rate-limiting between LLM calls
  - Writing results as JSONL to the volume, then loading into a Delta table

Usage (CLI):
    uv run batch-sec-extraction \
        --volume-path /Volumes/hackathon_521302447211702/aa/raw_files \
        --delta-table hackathon_521302447211702.aa.sec_extracted_attributes \
        --warehouse-id <sql-warehouse-id>

    # Resume after a crash (skips files already in results JSONL):
    uv run batch-sec-extraction \
        --volume-path /Volumes/hackathon_521302447211702/aa/raw_files \
        --delta-table hackathon_521302447211702.aa.sec_extracted_attributes \
        --warehouse-id <sql-warehouse-id> \
        --resume
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

RESULTS_JSONL_FILENAME = "extraction_results.jsonl"
DEFAULT_VOLUME_PATH = "/Volumes/hackathon_521302447211702/aa/raw_files"
DEFAULT_DELTA_TABLE = "hackathon_521302447211702.aa.sec_extracted_attributes"
DEFAULT_DELAY_BETWEEN_FILES = 1.0


def _get_workspace_client():
    """Build a WorkspaceClient using the CLI profile from .env."""
    from databricks.sdk import WorkspaceClient

    profile = os.environ.get("DATABRICKS_CONFIG_PROFILE")
    if profile:
        return WorkspaceClient(profile=profile)
    return WorkspaceClient()


def _list_volume_files(volume_path: str) -> list[str]:
    """List all .txt files in the given Databricks Volume directory."""
    w = _get_workspace_client()
    files = []
    for entry in w.files.list_directory_contents(volume_path):
        name = entry.path
        if name and name.lower().endswith(".txt"):
            files.append(name)
    logger.info("Found %d .txt files in %s", len(files), volume_path)
    return sorted(files)


def _read_volume_file(file_path: str) -> str:
    """Download and read a single file from a Databricks Volume."""
    w = _get_workspace_client()
    resp = w.files.download(file_path)
    content = resp.contents.read()
    if isinstance(content, bytes):
        return content.decode("utf-8", errors="ignore")
    return content


def _load_processed_files(results_path: Path) -> set[str]:
    """Load set of already-processed source filenames from a local JSONL file."""
    processed = set()
    if results_path.exists():
        with open(results_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    src = rec.get("_source_file", "")
                    if src:
                        processed.add(src)
                except json.JSONDecodeError:
                    continue
    return processed


def _flatten_record(result: dict, source_file: str, processing_time: float) -> dict:
    """Flatten a pipeline result into a single row for Delta."""
    record = dict(result.get("record", {}))

    record["_source_file"] = Path(source_file).name
    record["_fill_rate"] = result.get("fill_rate", 0.0)
    record["_processing_time_s"] = round(processing_time, 2)
    record["_valid"] = result.get("evaluation", {}).get("valid", False)
    record["_confidence"] = result.get("evaluation", {}).get("confidence", 0.0)
    record["_issues"] = json.dumps(result.get("evaluation", {}).get("issues", []))
    record["_llm_filled_fields"] = json.dumps(
        result.get("evaluation", {}).get("llm_filled_fields", [])
    )
    record["_error"] = result.get("error")

    for k, v in record.items():
        if isinstance(v, (list, dict)):
            record[k] = json.dumps(v, default=str)
        elif isinstance(v, bool):
            record[k] = v
        elif v is not None and not isinstance(v, (int, float, str)):
            record[k] = str(v)

    return record


def _upload_jsonl_to_volume(local_path: Path, volume_dir: str):
    """Upload the results JSONL file to the Databricks Volume."""
    w = _get_workspace_client()
    dest_path = f"{volume_dir}/{RESULTS_JSONL_FILENAME}"
    with open(local_path, "rb") as f:
        w.files.upload(dest_path, f, overwrite=True)
    logger.info("Uploaded results to %s", dest_path)
    return dest_path


def _create_delta_table(
    volume_jsonl_path: str,
    delta_table: str,
    warehouse_id: str,
):
    """Create or replace a Delta table from the JSONL file on the volume."""
    from databricks.sdk.service.sql import Disposition, Format, StatementState

    w = _get_workspace_client()

    sql = f"""
    CREATE OR REPLACE TABLE {delta_table}
    AS SELECT * FROM read_files(
        '{volume_jsonl_path}',
        format => 'json',
        inferColumnTypes => 'true'
    )
    """
    logger.info("Creating Delta table %s from %s ...", delta_table, volume_jsonl_path)
    logger.info("SQL: %s", sql.strip())

    # Submit with wait_timeout="0s" (async), then poll until done.
    stmt = w.statement_execution.execute_statement(
        warehouse_id=warehouse_id,
        statement=sql,
        wait_timeout="0s",
        disposition=Disposition.EXTERNAL_LINKS,
        format=Format.JSON_ARRAY,
    )
    statement_id = stmt.statement_id

    terminal_states = {
        StatementState.SUCCEEDED,
        StatementState.FAILED,
        StatementState.CANCELED,
        StatementState.CLOSED,
    }

    for _ in range(120):  # poll up to 10 minutes (120 x 5s)
        time.sleep(5)
        stmt = w.statement_execution.get_statement(statement_id)
        state = stmt.status.state if stmt.status else None
        logger.info("SQL statement %s — state: %s", statement_id, state)
        if state in terminal_states:
            break
    else:
        logger.error("SQL statement timed out after 10 minutes")
        raise RuntimeError("SQL statement timed out")

    if stmt.status and stmt.status.error:
        logger.error("Delta table creation failed: %s", stmt.status.error.message)
        raise RuntimeError(f"SQL failed: {stmt.status.error.message}")

    if stmt.status and stmt.status.state != StatementState.SUCCEEDED:
        raise RuntimeError(f"SQL ended in unexpected state: {stmt.status.state}")

    logger.info("Delta table %s created successfully", delta_table)
    return delta_table


def _discover_warehouse() -> str | None:
    """Try to find an available SQL warehouse."""
    try:
        w = _get_workspace_client()
        warehouses = list(w.warehouses.list())
        for wh in warehouses:
            if wh.state and wh.state.value in ("RUNNING", "STARTING"):
                logger.info("Discovered SQL warehouse: %s (%s)", wh.name, wh.id)
                return wh.id
        if warehouses:
            wh = warehouses[0]
            logger.info("Using first available warehouse: %s (%s)", wh.name, wh.id)
            return wh.id
    except Exception as exc:
        logger.warning("Could not discover SQL warehouses: %s", exc)
    return None


def batch_process(
    volume_path: str = DEFAULT_VOLUME_PATH,
    delta_table: str = DEFAULT_DELTA_TABLE,
    warehouse_id: str | None = None,
    resume: bool = False,
    delay: float = DEFAULT_DELAY_BETWEEN_FILES,
    max_files: int | None = None,
    skip_delta: bool = False,
    verbose: bool = False,
):
    """Main batch processing entry point.

    Args:
        volume_path:   Databricks Volume directory containing .txt files.
        delta_table:   Target Delta table (catalog.schema.table).
        warehouse_id:  SQL warehouse ID for Delta write.
        resume:        If True, skip files already in the local results JSONL.
        delay:         Seconds to wait between files (rate-limiting for LLM).
        max_files:     Process at most N files (for testing).
        skip_delta:    If True, only produce the JSONL — skip Delta table creation.
        verbose:       Enable verbose logging.
    """
    from agent_server.sec_extraction.config import get_config
    from agent_server.sec_extraction.workflow import extraction_workflow

    cfg = get_config()
    results_path = Path(RESULTS_JSONL_FILENAME)

    # --- 1. List files -------------------------------------------------------
    print(f"\n{'='*60}")
    print("  SEC Filing Batch Extraction")
    print(f"{'='*60}")
    print(f"  Volume path  : {volume_path}")
    print(f"  Delta table  : {delta_table}")
    print(f"  Resume mode  : {resume}")
    print(f"{'='*60}\n")

    all_files = _list_volume_files(volume_path)
    if not all_files:
        print("No .txt files found in the volume. Exiting.")
        return

    # --- 2. Resume filtering -------------------------------------------------
    processed = set()
    if resume:
        processed = _load_processed_files(results_path)
        print(f"  Already processed: {len(processed)} files")

    pending = [f for f in all_files if Path(f).name not in processed]
    if max_files:
        pending = pending[:max_files]

    print(f"  Total files      : {len(all_files)}")
    print(f"  Pending          : {len(pending)}")
    print()

    if not pending:
        print("All files already processed. Nothing to do.")
        if not skip_delta:
            _finalize_delta(results_path, volume_path, delta_table, warehouse_id)
        return

    # --- 3. Process each file ------------------------------------------------
    success_count = 0
    error_count = 0
    total_fill = 0.0
    start_time = time.time()

    with open(results_path, "a", encoding="utf-8") as out_f:
        for idx, file_path in enumerate(pending):
            fname = Path(file_path).name
            print(f"[{idx+1}/{len(pending)}] Processing {fname} ...", end=" ", flush=True)

            t0 = time.time()
            try:
                content = _read_volume_file(file_path)
                if not content or not content.strip():
                    raise ValueError("File is empty")

                result = extraction_workflow(content, cfg)
                elapsed = time.time() - t0
                fill_rate = result.get("fill_rate", 0.0)
                total_fill += fill_rate
                success_count += 1

                row = _flatten_record(result, file_path, elapsed)
                out_f.write(json.dumps(row, default=str) + "\n")
                out_f.flush()

                print(
                    f"OK  fill={fill_rate:.0%}  "
                    f"fields={sum(1 for v in result.get('record', {}).values() if v is not None and v != '' and v != [])}  "
                    f"time={elapsed:.1f}s"
                )

            except Exception as exc:
                elapsed = time.time() - t0
                error_count += 1
                error_row = {
                    "_source_file": fname,
                    "_error": str(exc),
                    "_processing_time_s": round(elapsed, 2),
                    "_fill_rate": 0.0,
                    "_valid": False,
                    "_confidence": 0.0,
                    "_issues": "[]",
                    "_llm_filled_fields": "[]",
                }
                out_f.write(json.dumps(error_row, default=str) + "\n")
                out_f.flush()
                print(f"FAIL  {exc}")

            if delay > 0 and idx < len(pending) - 1:
                time.sleep(delay)

    # --- 4. Summary ----------------------------------------------------------
    total_time = time.time() - start_time
    avg_fill = total_fill / max(success_count, 1)

    print(f"\n{'='*60}")
    print("  Batch Processing Complete")
    print(f"{'='*60}")
    print(f"  Processed    : {success_count + error_count}")
    print(f"  Succeeded    : {success_count}")
    print(f"  Failed       : {error_count}")
    print(f"  Avg fill rate: {avg_fill:.1%}")
    print(f"  Total time   : {total_time:.0f}s ({total_time/60:.1f}m)")
    print(f"  Results file : {results_path}")
    print(f"{'='*60}\n")

    # --- 5. Write to Delta ---------------------------------------------------
    if not skip_delta:
        _finalize_delta(results_path, volume_path, delta_table, warehouse_id)


def _finalize_delta(
    results_path: Path,
    volume_path: str,
    delta_table: str,
    warehouse_id: str | None,
):
    """Upload JSONL to volume and create the Delta table."""
    if not results_path.exists():
        print("No results file found. Skipping Delta write.")
        return

    wh_id = warehouse_id
    if not wh_id:
        print("Discovering SQL warehouse ...")
        wh_id = _discover_warehouse()
        if not wh_id:
            print(
                "ERROR: No SQL warehouse found. Provide --warehouse-id or "
                "start a SQL warehouse in your Databricks workspace."
            )
            print(f"Results are saved locally at: {results_path}")
            return

    print(f"Uploading results to volume ...")
    volume_jsonl = _upload_jsonl_to_volume(results_path, volume_path)

    print(f"Creating Delta table {delta_table} ...")
    _create_delta_table(volume_jsonl, delta_table, wh_id)
    print(f"\nDelta table ready: {delta_table}")
    print("Query it with: SELECT * FROM " + delta_table)


def main():
    parser = argparse.ArgumentParser(
        description="Batch-process SEC filings from a Databricks Volume into a Delta table",
    )
    parser.add_argument(
        "--volume-path",
        default=os.environ.get("SEC_VOLUME_PATH", DEFAULT_VOLUME_PATH),
        help="Volume directory containing raw .txt filings",
    )
    parser.add_argument(
        "--delta-table",
        default=os.environ.get("SEC_DELTA_TABLE", DEFAULT_DELTA_TABLE),
        help="Target Delta table (catalog.schema.table)",
    )
    parser.add_argument(
        "--warehouse-id",
        default=os.environ.get("SEC_WAREHOUSE_ID"),
        help="SQL Warehouse ID (auto-discovered if omitted)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip files already in the local results JSONL",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=float(os.environ.get("SEC_LLM_DELAY_BETWEEN_CALLS", DEFAULT_DELAY_BETWEEN_FILES)),
        help="Seconds between files (rate-limit LLM calls)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Process at most N files (useful for testing)",
    )
    parser.add_argument(
        "--skip-delta",
        action="store_true",
        help="Only produce the local JSONL — skip Delta table creation",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=".env", override=True)
    except ImportError:
        pass

    batch_process(
        volume_path=args.volume_path,
        delta_table=args.delta_table,
        warehouse_id=args.warehouse_id,
        resume=args.resume,
        delay=args.delay,
        max_files=args.max_files,
        skip_delta=args.skip_delta,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()

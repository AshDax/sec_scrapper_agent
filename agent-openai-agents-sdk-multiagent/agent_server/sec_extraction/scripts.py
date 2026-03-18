"""CLI entry point for SEC attribute extraction.

Usage:
    uv run run-sec-extraction -d "Your SEC filing text..." -o results.json
    uv run run-sec-extraction -f path/to/filing.html -o results.json
    uv run run-sec-extraction -f filing.txt --delta-table catalog.schema.table
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="SEC Filing Attribute Extraction (LangChain + LangGraph)",
    )
    parser.add_argument("-d", "--document", help="SEC filing text (inline)")
    parser.add_argument("-f", "--file", help="Path to SEC filing file")
    parser.add_argument("-o", "--output", help="Output JSON file path")
    parser.add_argument("--delta-table", help="Delta table for output (catalog.schema.table)")
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLflow logging")
    parser.add_argument("--supervisor", action="store_true",
                        help="Use ReAct supervisor instead of fixed workflow")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    # Load .env if present
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=".env", override=True)
    except ImportError:
        pass

    # Get document text
    document = args.document
    if args.file:
        path = Path(args.file)
        if not path.exists():
            print(f"Error: file not found: {path}", file=sys.stderr)
            sys.exit(1)
        document = path.read_text(encoding="utf-8", errors="ignore")
        if not document or not document.strip():
            print(f"Error: file is empty: {path}", file=sys.stderr)
            sys.exit(1)
        print(f"Loaded {len(document):,} chars from {path}", file=sys.stderr)

    if not document or not document.strip():
        print("Error: provide --document or --file with non-empty content", file=sys.stderr)
        sys.exit(1)

    # Run extraction
    if args.supervisor:
        from agent_server.sec_extraction.supervisor import build_supervisor
        from agent_server.sec_extraction.config import get_config

        from langchain_core.messages import HumanMessage

        # Supervisor sends the prompt to the LLM; keep document within context limit.
        MAX_SUPERVISOR_DOC_CHARS = 6_000
        doc_excerpt = document[:MAX_SUPERVISOR_DOC_CHARS]
        if len(document) > MAX_SUPERVISOR_DOC_CHARS:
            print(f"Supervisor: using first {MAX_SUPERVISOR_DOC_CHARS:,} chars (full doc {len(document):,})", file=sys.stderr)
        agent = build_supervisor(get_config())
        result = agent.invoke({"messages": [HumanMessage(content=f"Extract attributes from this SEC filing:\n\n{doc_excerpt}")]})
        messages = result.get("messages", [])
        final_text = messages[-1].content if messages else ""
        output = {"supervisor_output": final_text}
    else:
        from agent_server.sec_extraction.run import run_extraction

        results = run_extraction(
            documents=[document],
            use_mlflow=not args.no_mlflow,
            delta_table=args.delta_table,
        )
        output = results[0] if results else {"error": "No results"}

    # Output
    formatted = json.dumps(output, indent=2, default=str)

    if args.output:
        Path(args.output).write_text(formatted)
        print(f"Results written to {args.output}")
    else:
        print(formatted)


if __name__ == "__main__":
    main()

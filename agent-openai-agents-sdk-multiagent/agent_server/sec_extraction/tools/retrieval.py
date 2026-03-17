"""Step 3 — Retrieval Agent

Searches the Databricks Vector Search index for SEC filing chunks
relevant to the current document.  Returns a list of text snippets
that are passed to the LLM extractor as supplementary context.

If Vector Search is not configured or unavailable the function
returns an empty list so the pipeline can continue without it.

Usage:
    from agent_server.sec_extraction.tools.retrieval import retrieve_context
    from agent_server.sec_extraction.config import get_config
    chunks = retrieve_context("Acme Corp employee count", get_config())
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_server.sec_extraction.config import ExtractionConfig

logger = logging.getLogger(__name__)


def retrieve_context(query: str, config: ExtractionConfig) -> list[str]:
    """Search the Vector Search index and return matching text chunks.

    Args:
        query:  Natural-language search string.
        config: Pipeline configuration (carries endpoint + index names).

    Returns:
        Up to 5 text snippets.  Empty list on any failure.
    """
    if not query.strip():
        return []

    try:
        from databricks.vector_search.client import VectorSearchClient

        vsc = VectorSearchClient()
        index = vsc.get_index(
            endpoint_name=config.vector_search_endpoint,
            index_name=config.vector_search_index,
        )

        results = index.similarity_search(
            query_text=query,
            columns=["section_text"],
            num_results=5,
        )

        result_block = results.get("result", {})
        if result_block.get("row_count", 0) == 0:
            return []

        rows = result_block.get("data_array", [])
        return [row[0] for row in rows if row and row[0]]

    except ImportError:
        logger.warning(
            "databricks-vector-search not installed — skipping retrieval. "
            "Install with: uv add databricks-vectorsearch"
        )
        return []
    except Exception as exc:
        logger.warning("Vector Search retrieval failed: %s", exc)
        return []

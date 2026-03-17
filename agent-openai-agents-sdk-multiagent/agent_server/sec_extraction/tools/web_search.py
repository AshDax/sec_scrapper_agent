"""Step 7 — Web Search Fallback Agent

Fires when the evaluation fill rate is below the threshold.
Searches DuckDuckGo (or Serper if configured) for supplementary
information about the company.

Usage:
    from agent_server.sec_extraction.tools.web_search import web_search
    snippets = web_search("Pinnacle Manufacturing Corp employees revenue")
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_server.sec_extraction.config import ExtractionConfig

logger = logging.getLogger(__name__)


def web_search(query: str, config: ExtractionConfig | None = None) -> list[str]:
    """Search the web and return text snippets.

    Tries Serper API first (if key is set), then falls back to DuckDuckGo.

    Args:
        query:  Search string (e.g. "Acme Corp revenue employees").
        config: Optional pipeline config for API keys.

    Returns:
        Up to 5 text snippets.  Empty list on any failure.
    """
    if config and config.serper_api_key:
        return _serper_search(query, config.serper_api_key)
    return _ddg_search(query)


def _ddg_search(query: str) -> list[str]:
    try:
        from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))

        return [
            f"{r.get('title', '')}: {r.get('body', '')}"
            for r in results
            if r.get("body")
        ]
    except Exception as exc:
        logger.warning("DuckDuckGo search failed: %s", exc)
        return []


def _serper_search(query: str, api_key: str) -> list[str]:
    try:
        import requests

        resp = requests.post(
            "https://google.serper.dev/search",
            json={"q": query, "num": 5},
            headers={"X-API-KEY": api_key},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        snippets = []
        for item in data.get("organic", []):
            title = item.get("title", "")
            snippet = item.get("snippet", "")
            if snippet:
                snippets.append(f"{title}: {snippet}")
        return snippets[:5]

    except Exception as exc:
        logger.warning("Serper search failed: %s — falling back to DuckDuckGo", exc)
        return _ddg_search(query)

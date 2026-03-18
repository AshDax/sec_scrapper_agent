"""Step 7 — Web Search Fallback Agent

Fires when the evaluation fill rate is below the threshold.
Searches DuckDuckGo (or Serper if configured) for supplementary
information about the company.

Usage:
    from agent_server.sec_extraction.tools.web_search import web_search
    snippets = web_search("Pinnacle Manufacturing Corp employees revenue")
"""
%pip install google-search-results
%pip install transformers torch duckduckgo-search
%pip install duckduckgo-search --upgrade

from __future__ import annotations
from serpapi import GoogleSearch
from typing import List, Dict
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_server.sec_extraction.config import ExtractionConfig

logger = logging.getLogger("web_search")
logger.setLevel(logging.INFO)

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


def _ddg_search(query: str, max_results=5) -> list[str]:
    """
    Performs a DuckDuckGo search and returns a list of text results.

    Args:
        query (str): The search query.
        max_results (int): Max number of search results to return.

    Returns:
        List[str]: List of "title: body" strings.
    """
    try:
        from duckduckgo_search import DDGS


        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))

        # Filter out results without body
        return [
            f"{r.get('title', '')}: {r.get('body', '')}"
            for r in results
            if r.get("body")
        ]

    except Exception as exc:
        logger.warning("DuckDuckGo search failed: %s", exc)
        return []


def _serper_search(query: str, api_key: str, engine: str = "google", num_results: int = 5) -> list[str]:
    """
    Perform web search using SerpAPI and return structured results.
    Fully self-contained and safe.
    """
    try:
        # Step 1: Build parameters
        params = {
            "q": query,
            "engine": engine,
            "num": num_results,
            "api_key": api_key
        }

        # Step 2: Execute search
        search = GoogleSearch(params)
        raw_response = search.get_dict()

        # Debug (optional)
        # print(raw_response)

        # Step 3: Parse results safely
        results = raw_response.get("organic_results", [])

        parsed_results = []
        for r in results:
            parsed_results.append({
                "title": r.get("title", ""),
                "snippet": r.get("snippet", ""),
                "link": r.get("link", "")
            })

        return parsed_results

    except Exception as exc:
        logger.warning("Serper search failed: %s — falling back to DuckDuckGo", exc)
        return _ddg_search(query)

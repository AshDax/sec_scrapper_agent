"""Centralised configuration — all knobs live here.

Every value is read from an environment variable so nothing is hard-coded.
Extend this class when you add new integrations.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache

from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)


class ExtractionConfig:
    """Configuration for the SEC extraction pipeline."""

    def __init__(self):
        # LLM
        self.llm_model: str = os.environ.get(
            "SEC_EXTRACTION_LLM_MODEL", "databricks-claude-sonnet-4-5"
        )
        self.groq_api_key: str | None = os.environ.get("GROQ_API_KEY")
        self.openai_api_base: str | None = os.environ.get("OPENAI_API_BASE")

        # Databricks workspace (for model serving + vector search)
        self.databricks_host: str | None = os.environ.get("DATABRICKS_HOST")
        self.databricks_token: str | None = os.environ.get("DATABRICKS_TOKEN")

        # Vector Search
        self.vector_search_endpoint: str = os.environ.get(
            "SEC_VECTOR_SEARCH_ENDPOINT", "sec_endpoint"
        )
        self.vector_search_index: str = os.environ.get(
            "SEC_VECTOR_SEARCH_INDEX",
            "hackathon_521302447211702.aa.vsi_using_bge_large_on_sections",
        )

        # Delta output
        self.delta_table: str | None = os.environ.get("SEC_DELTA_TABLE")

        # Web fallback
        self.serper_api_key: str | None = os.environ.get("SERPER_API_KEY")

        # Thresholds
        self.fill_rate_threshold: float = float(
            os.environ.get("SEC_FILL_RATE_THRESHOLD", "0.5")
        )

    # ---- LLM factory -------------------------------------------------------

    def _resolve_databricks_credentials(self) -> tuple[str, str]:
        """Get Databricks host + token, falling back to the CLI profile."""
        host = self.databricks_host
        token = self.databricks_token

        if host and token:
            return host, token

        try:
            from databricks.sdk import WorkspaceClient

            w = WorkspaceClient()
            host = host or w.config.host
            # CLI/OAuth auth doesn't populate .token directly —
            # call authenticate() to get the actual bearer token.
            headers = w.config.authenticate()
            auth_header = headers.get("Authorization", "")
            token = token or auth_header.replace("Bearer ", "")
            return host, token
        except Exception as exc:
            logger.warning("Databricks credential resolution failed: %s", exc)

        logger.warning("No Databricks credentials found — LLM calls may fail")
        return host or "", token or ""

    def get_llm(self, temperature: float = 0) -> BaseChatModel:
        """Return the configured LangChain chat model.

        Priority: GROQ_API_KEY → Databricks-hosted model (via langchain-openai).
        """
        from langchain_openai import ChatOpenAI

        if self.groq_api_key:
            return ChatOpenAI(
                model=self.llm_model,
                api_key=self.groq_api_key,
                base_url=self.openai_api_base or "https://api.groq.com/openai/v1",
                temperature=temperature,
            )

        host, token = self._resolve_databricks_credentials()
        return ChatOpenAI(
            model=self.llm_model,
            api_key=token,
            base_url=f"{host}/serving-endpoints",
            temperature=temperature,
        )


@lru_cache(maxsize=1)
def get_config() -> ExtractionConfig:
    return ExtractionConfig()

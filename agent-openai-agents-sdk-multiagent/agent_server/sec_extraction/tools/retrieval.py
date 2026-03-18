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

def get_embedding(text):
    """
    Returns a flat list of floats suitable for VSI using databricks-bge-large-en
    """
    response = deploy_client.predict(
        endpoint="databricks-bge-large-en",
        inputs={"input": text}
    )
    embedding = response.data[0]["embedding"]
    return [float(x) for x in embedding]


def retrieve_context(query: str, config: ExtractionConfig) -> list[str]:
    """Search the Vector Search index and return matching text chunks.

    Args:
        query:  Natural-language search string.
        config: Pipeline configuration (carries endpoint + index names).

    Returns:
        Up to 5 text snippets.  Empty list on any failure.
    
    Retrieves minimal chunks per company for all query parts.
    Deduplicates chunks across query parts and returns a DataFrame with:
    ["id","company","section_text","score","query_part"]
    
    If a chunk matches multiple query parts, all matched parts are concatenated.
    """
    # if not query.strip():
    #     return []

    try:
        from databricks.vector_search.client import VectorSearchClient
        import pandas as pd
        import numpy as np
        from databricks.sdk import WorkspaceClient
        from databricks.vector_search.client import VectorSearchClient
        from mlflow.deployments import get_deploy_client
        vsc = VectorSearchClient()
        deploy_client = get_deploy_client("databricks")

        vsc = VectorSearchClient()
        index = vsc.get_index(
            endpoint_name=config.vector_search_endpoint,
            index_name=config.vector_search_index,
        )
   
    # Split query into keyword:description parts
        query_parts = [q.strip() for q in query.split(",")]

        # Dictionary to track chunks by ID
        chunk_map = {}  # key = chunk id, value = dict with chunk info + matched query parts

        for company in company_names:
            for part in query_parts:
                emb_part = get_embedding(part)

                # -----------------------------
                # VSI search
                # -----------------------------
                results = index.similarity_search(
                    query_vector=emb_part,
                    columns=["id", "company", "section_text"],
                    num_results=1,
                    filters={"company": company}  # or {"company": company}
                )

                if not results:
                    continue

                # -----------------------------
                # Extract columns and rows from raw API (data_array)
                # -----------------------------
                columns = [col["name"] for col in results["manifest"]["columns"]]
                rows = results["result"]["data_array"]

                df_results = pd.DataFrame(rows, columns=columns)

                # Ensure score is numeric
                df_results["score"] = df_results["score"].astype(float)

                # -----------------------------
                # Find top chunk(s) with max score
                # -----------------------------
                max_score = df_results["score"].max()
                top_chunks = df_results[df_results["score"] == max_score]

                # -----------------------------
                # Add chunks to map and concatenate query parts if needed
                # -----------------------------
                for _, r in top_chunks.iterrows():
                    chunk_id = r["id"]
                    if chunk_id not in chunk_map:
                        # First time this chunk is selected
                        chunk_map[chunk_id] = {
                            "id": r["id"],
                            "company": r["company"],
                            "section_text": r["section_text"],
                            "score": r["score"],
                            "query_parts": [part]
                        }
                    else:
                        # Already selected; append query part if not already present
                        if part not in chunk_map[chunk_id]["query_parts"]:
                            chunk_map[chunk_id]["query_parts"].append(part)

        # -----------------------------
        # Convert all selected chunks to final DataFrame
        # -----------------------------
        df_final = pd.DataFrame([
            {
                "id": v["id"],
                "company": v["company"],
                "section_text": v["section_text"],
                "score": v["score"],
                "query_part": ", ".join(v["query_parts"])  # concatenate all matched query parts
            }
            for v in chunk_map.values()
        ])
        logger.info("Retrieval agent working completed successfully!")
        spark_df = spark.createDataFrame(df_final)
        spark_df.write.format("delta").option("mergeSchema", "true").mode("overwrite").saveAsTable("retrieved_chunks_through_vsi")
        return df_final


    #     results = index.similarity_search(
    #         query_text=query,
    #         columns=["section_text"],
    #         num_results=5,
    #     )

    #     result_block = results.get("result", {})
    #     if result_block.get("row_count", 0) == 0:
    #         return []

    #     rows = result_block.get("data_array", [])
    #     return [row[0] for row in rows if row and row[0]]

    # except ImportError:
    #     logger.warning(
    #         "databricks-vector-search not installed — skipping retrieval. "
    #         "Install with: uv add databricks-vectorsearch"
    #     )
    #     return []
    # except Exception as exc:
    #     logger.warning("Vector Search retrieval failed: %s", exc)
    #     return []





"""
SEC Filing Information Extraction Agent.

Multi-agent orchestrator that extracts structured business attributes from
SEC filings (10-K, 10-Q, 8-K). The pipeline:

  1. Scraper Agent  — regex-based extraction for structured patterns
  2. Research Agent — RAG via Databricks Vector Search + web search fallback
  3. Extraction     — LLM reasoning over gathered context (the agent itself)
  4. Evaluation     — self-validation of extracted values (the agent itself)

The output is structured JSON with attribute values, confidence scores,
sources, and evidence — ready for writing to a Delta table.
"""

import json
import logging
import re
from contextlib import nullcontext
from typing import AsyncGenerator

import litellm
import mlflow
from agents import Agent, Runner, function_tool, set_default_openai_api, set_default_openai_client
from agents.tracing import set_trace_processors
from databricks_openai import AsyncDatabricksOpenAI
from databricks_openai.agents import McpServer
from mlflow.genai.agent_server import invoke, stream
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)

from agent_server.utils import (
    build_mcp_url,
    get_session_id,
    process_agent_stream_events,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

VECTOR_SEARCH_CATALOG = "hackathon_521302447211702"
VECTOR_SEARCH_SCHEMA = "default"
VECTOR_SEARCH_INDEX = "embedding_vsi"

TARGET_ATTRIBUTES = [
    "business_name",
    "parent_name",
    "business_phone",
    "address_city_state_zip",
    "industry_description",
    "naics_sic_candidates",
    "number_of_employees",
    "is_manufacturer",
    "risk_factor_keywords",
    "revenue_mentions",
    "website_domain",
    "business_purpose_summary",
]

# ---------------------------------------------------------------------------
# Client setup
# ---------------------------------------------------------------------------

set_default_openai_client(AsyncDatabricksOpenAI())
set_default_openai_api("chat_completions")
set_trace_processors([])
mlflow.openai.autolog()
logging.getLogger("mlflow.utils.autologging_utils").setLevel(logging.ERROR)
litellm.suppress_debug_info = True

# ---------------------------------------------------------------------------
# Tools — Scraper Agent
# ---------------------------------------------------------------------------

SCRAPER_PATTERNS = {
    "business_phone": [
        r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}",
        r"\+?1?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}",
    ],
    "website_domain": [
        r"(?:https?://)?(?:www\.)?([a-zA-Z0-9][-a-zA-Z0-9]*\.[a-zA-Z]{2,}(?:\.[a-zA-Z]{2,})?)",
    ],
    "number_of_employees": [
        r"(?:approximately|about|nearly|over|more than)?\s*(\d{1,3}(?:,\d{3})*)\s+(?:full[- ]?time\s+)?employees",
        r"(?:headcount|workforce|staff)\s+(?:of\s+)?(?:approximately\s+)?(\d{1,3}(?:,\d{3})*)",
        r"(\d{1,3}(?:,\d{3})*)\s+(?:people|personnel|workers)",
    ],
    "revenue_mentions": [
        r"\$\s*[\d,]+(?:\.\d+)?\s*(?:billion|million|thousand|B|M|K)",
        r"(?:revenue|net\s+sales|total\s+revenue)\s+(?:of|was|were|totaled)?\s*\$\s*[\d,]+(?:\.\d+)?",
        r"(?:revenue|net\s+sales)\s+(?:increased|decreased|grew)?\s*(?:to|by)?\s*\$\s*[\d,]+(?:\.\d+)?",
    ],
    "address_city_state_zip": [
        r"\d+\s+[\w\s]+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Boulevard|Blvd|Way|Lane|Ln|Place|Pl|Suite|Ste)\.?\s*,?\s*(?:Suite|Ste\.?\s*\d+\s*,?\s*)?[\w\s]+,\s*[A-Z]{2}\s+\d{5}(?:-\d{4})?",
    ],
    "naics_sic_candidates": [
        r"(?:NAICS|SIC)\s*(?:code|Code)?:?\s*(\d{4,6})",
        r"Standard\s+Industrial\s+Classification\s*(?:code)?\s*:?\s*(\d{4})",
    ],
    "business_name": [
        r"(?:EXACT NAME OF REGISTRANT|Company Name|Registrant)[:\s]+([A-Z][\w\s&.,'-]+(?:Inc|Corp|LLC|Ltd|Co|LP|Company|Corporation|Group|Holdings)\.?)",
    ],
}


@function_tool
def scraper_extract(attribute: str, text: str) -> str:
    """Extract a structured attribute from text using regex patterns.

    Best for: phone numbers, websites, employee counts, revenue figures,
    addresses, NAICS/SIC codes, and business names that follow standard
    SEC filing formatting.

    Args:
        attribute: The attribute to extract (e.g. 'business_phone', 'revenue_mentions').
        text: The document text to search within.

    Returns:
        JSON with matched values and method, or 'NOT_FOUND' if no pattern matched.
    """
    patterns = SCRAPER_PATTERNS.get(attribute)
    if not patterns:
        return json.dumps({
            "status": "no_pattern",
            "message": f"No regex pattern available for '{attribute}'. Use LLM reasoning instead.",
        })

    all_matches = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
        all_matches.extend(matches)

    if all_matches:
        unique = list(dict.fromkeys(str(m).strip() for m in all_matches))[:10]
        return json.dumps({
            "status": "found",
            "attribute": attribute,
            "matches": unique,
            "match_count": len(unique),
            "method": "regex_scraper",
        })

    return json.dumps({
        "status": "not_found",
        "attribute": attribute,
        "message": "No regex matches found. Try vector search or web search.",
    })


# ---------------------------------------------------------------------------
# Tools — Web Search Agent
# ---------------------------------------------------------------------------


@function_tool
def web_search(query: str) -> str:
    """Search the web for supplementary business information.

    Use as a fallback when SEC filing context is insufficient.
    Good for: company websites, current employee counts, business status,
    recent news about the company.

    Args:
        query: Specific search query (e.g. 'Acme Corp number of employees 2024').

    Returns:
        Up to 5 search result snippets with titles and URLs.
    """
    try:
        from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))

        if not results:
            return json.dumps({"status": "no_results", "query": query})

        output = []
        for r in results:
            output.append({
                "title": r.get("title", ""),
                "snippet": r.get("body", ""),
                "url": r.get("href", ""),
            })

        return json.dumps({"status": "found", "query": query, "results": output})
    except Exception as e:
        return json.dumps({"status": "error", "query": query, "error": str(e)})


# ---------------------------------------------------------------------------
# Supervisor Agent — instructions & creation
# ---------------------------------------------------------------------------

SUPERVISOR_INSTRUCTIONS = """You are an SEC Filing Information Extraction Supervisor — a specialized AI agent that extracts structured business attributes from SEC filings and classifies businesses as open or closed.

## Your Extraction Strategy

For EACH attribute, follow this prioritized approach:

### Step 1: Regex Scraper (fastest, most reliable for structured data)
Use the `scraper_extract` tool for attributes with clear patterns:
- business_phone, website_domain, number_of_employees
- revenue_mentions, address_city_state_zip, naics_sic_candidates, business_name

### Step 2: Vector Search (SEC Filing RAG)
Use the SEC Filing Search tools (MCP) to search for relevant context in the SEC filings database.
Formulate SPECIFIC queries like:
- "number of employees" or "total headcount"
- "revenue" or "net sales" or "total revenue"
- "principal office" or "headquarters address"
- "SIC code" or "NAICS code" or "industry classification"
- "business description" or "company overview"
- "parent company" or "subsidiary of"

### Step 3: LLM Reasoning (your own analysis)
Once you have context from scraper and/or vector search, use your reasoning to:
- Extract the attribute value from the context
- Assess confidence (0.0-1.0)
- Cite evidence from the source text

### Step 4: Self-Validation
Critically evaluate each extraction:
- Is the value plausible? (employee count should be realistic, phone should be valid format)
- Is it consistent with other extracted attributes?
- Is the evidence strong enough to support the confidence score?

### Step 5: Web Search (fallback)
If SEC filing context is insufficient, use `web_search` with specific queries:
- "[company name] number of employees"
- "[company name] headquarters address"
- "[company name] website"

## Target Attributes

Extract ALL of these for each filing:

| Attribute | Description | Scraper? |
|-----------|-------------|----------|
| business_name | Legal entity name | Yes |
| parent_name | Parent company (if subsidiary) | No |
| business_phone | Primary phone number | Yes |
| address_city_state_zip | Full business address | Yes |
| industry_description | What the business does | No |
| naics_sic_candidates | Industry classification codes | Yes |
| number_of_employees | Employee count | Yes |
| is_manufacturer | Manufacturing business? (true/false) | No |
| risk_factor_keywords | Key risk factors mentioned | No |
| revenue_mentions | Revenue figures with context | Yes |
| website_domain | Company website URL | Yes |
| business_purpose_summary | Brief business purpose | No |

## is_manufacturer Detection
Look for keywords: "manufactur", "produc", "fabricat", "assembl", "plant", "factory", "production facility" in the business description or industry section. Return true/false with reasoning.

## Business Status (open/closed) Detection
Look for: "dissolution", "bankruptcy", "ceased operations", "wound down", "liquidat", "deregistered", "revoked". Default to "open" unless clear evidence of closure.

## Output Format

ALWAYS return your final answer as structured JSON (inside a markdown code block):

```json
{
  "company_name": "...",
  "filing_type": "10-K/10-Q/8-K/etc.",
  "attributes": {
    "business_name": {
      "value": "extracted value or null",
      "confidence": 0.85,
      "source": "scraper|vector_search|web_search|llm_inference",
      "evidence": "brief quote or reasoning"
    },
    "parent_name": { ... },
    "business_phone": { ... },
    "address_city_state_zip": { ... },
    "industry_description": { ... },
    "naics_sic_candidates": { ... },
    "number_of_employees": { ... },
    "is_manufacturer": { ... },
    "risk_factor_keywords": { ... },
    "revenue_mentions": { ... },
    "website_domain": { ... },
    "business_purpose_summary": { ... }
  },
  "business_status": "open|closed",
  "status_reasoning": "evidence for open/closed classification"
}
```

## Important Guidelines
1. Be precise — cite evidence from the source text.
2. Assign confidence scores honestly (0.0-1.0).
3. If an attribute truly cannot be found, set value to null with confidence 0.0.
4. For numeric values (employees, revenue), preserve the original format from the filing.
5. Process ALL attributes — don't stop early.
6. If the user provides SEC filing text directly, extract from that text.
7. If the user asks about a specific company, search for their filings first.
"""


async def init_mcp_server():
    """Create a Vector Search MCP server for SEC filing retrieval."""
    try:
        vs_path = f"/api/2.0/mcp/vector-search/{VECTOR_SEARCH_CATALOG}/{VECTOR_SEARCH_SCHEMA}/{VECTOR_SEARCH_INDEX}"
        return McpServer(
            url=build_mcp_url(vs_path),
            name="SEC Filing Search — retrieve relevant excerpts from SEC filings (10-K, 10-Q, 8-K)",
        )
    except Exception as e:
        logging.warning(f"Vector Search MCP server initialization failed: {e}")
        return nullcontext()


def create_extraction_agent(mcp_server) -> Agent:
    """Build the SEC Filing Extraction Supervisor agent."""
    mcp_servers = []
    if mcp_server and not isinstance(mcp_server, type(nullcontext())):
        mcp_servers = [mcp_server]

    return Agent(
        name="SEC Filing Extraction Supervisor",
        instructions=SUPERVISOR_INSTRUCTIONS,
        model="databricks-claude-sonnet-4-5",
        mcp_servers=mcp_servers,
        tools=[scraper_extract, web_search],
    )


# ---------------------------------------------------------------------------
# MLflow Responses API handlers
# ---------------------------------------------------------------------------


@invoke()
async def invoke_handler(request: ResponsesAgentRequest) -> ResponsesAgentResponse:
    if session_id := get_session_id(request):
        mlflow.update_current_trace(metadata={"mlflow.trace.session": session_id})

    async with await init_mcp_server() as mcp_server:
        agent = create_extraction_agent(mcp_server)
        messages = [i.model_dump() for i in request.input]
        result = await Runner.run(agent, messages)
        return ResponsesAgentResponse(output=[item.to_input_item() for item in result.new_items])


@stream()
async def stream_handler(request: ResponsesAgentRequest) -> AsyncGenerator[ResponsesAgentStreamEvent, None]:
    if session_id := get_session_id(request):
        mlflow.update_current_trace(metadata={"mlflow.trace.session": session_id})

    async with await init_mcp_server() as mcp_server:
        agent = create_extraction_agent(mcp_server)
        messages = [i.model_dump() for i in request.input]
        result = Runner.run_streamed(agent, input=messages)

        async for event in process_agent_stream_events(result.stream_events()):
            yield event

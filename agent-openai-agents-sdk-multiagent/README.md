# SEC Filing Attribute Extraction Agent

Multi-agent SEC filing attribute extraction built with **LangChain** and **LangGraph**, served via **MLflow Responses API** on **Databricks Apps**, with a **Streamlit** UI.

Extracts 16 structured business attributes from raw SEC filings using a pipeline of specialised agents: regex scraping, vector search retrieval, LLM extraction, rule-based enrichment, self-evaluation, and web fallback.

---

## Repository Location

This project lives inside a larger monorepo. The root of **this** agent is:

```
app-templates/
└── agent-openai-agents-sdk-multiagent/   ← YOU ARE HERE
```

All commands below should be run from this directory:

```bash
cd agent-openai-agents-sdk-multiagent
```

---

## Project Structure

```
agent-openai-agents-sdk-multiagent/
│
├── agent_server/                    # Backend — agent logic + server
│   ├── agent.py                     # MLflow Responses API wrapper (invoke/stream)
│   ├── start_server.py              # FastAPI + MLflow AgentServer bootstrap
│   ├── utils.py                     # Databricks auth helpers (OBO, workspace client)
│   ├── evaluate_agent.py            # Evaluation harness (MLflow scorers)
│   │
│   └── sec_extraction/              # LangChain/LangGraph extraction pipeline
│       ├── __init__.py
│       ├── schemas.py               # Pydantic models: BusinessRecord, EvaluationResult, ExtractionState
│       ├── config.py                # Centralised config — env vars, LLM factory, credentials
│       ├── workflow.py              # LangGraph StateGraph (fixed pipeline)
│       ├── supervisor.py            # ReAct supervisor agent (flexible alternative)
│       ├── run.py                   # Batch runner — MLflow logging + Delta writes
│       ├── scripts.py               # CLI entry point (uv run run-sec-extraction)
│       │
│       └── tools/                   # One file per agent/step
│           ├── __init__.py
│           ├── text_extraction.py   # Step 1 — Clean raw HTML/text
│           ├── scraper.py           # Step 2 — Regex-based attribute extraction
│           ├── retrieval.py         # Step 3 — Databricks Vector Search (RAG)
│           ├── llm_extract.py       # Step 4 — LLM structured extraction
│           ├── enrichment.py        # Step 5 — Derived fields (is_manufacturer, is_open, NAICS)
│           ├── evaluate.py          # Step 6 — Self-evaluation + fill rate
│           └── web_search.py        # Step 7 — Web fallback (DuckDuckGo / Serper)
│
├── ui/
│   └── app.py                       # Streamlit UI — paste text, upload files, view results
│
├── scripts/
│   ├── quickstart.py                # One-command setup (auth, experiment, deps)
│   ├── start_app.py                 # Launches backend + Streamlit frontend together
│   ├── discover_tools.py            # Discover available Databricks workspace resources
│   └── grant_lakebase_permissions.py
│
├── .env                             # Local config (profile, experiment ID, vector search)
├── .env.example                     # Template for .env
├── databricks.yml                   # Databricks Asset Bundle config (deploy, resources)
├── pyproject.toml                   # Python deps + script entry points
├── uv.lock                          # Locked dependencies
└── README.md                        # This file
```

---

## Architecture

```
                        ┌─────────────────────┐
                        │   Raw SEC Filing     │
                        │   (text / HTML)      │
                        └─────────┬───────────┘
                                  │
                                  ▼
                   ┌──────────────────────────┐
                   │  1. Text Extraction       │  Strip HTML, normalise whitespace
                   │     (text_extraction.py)   │
                   └─────────────┬────────────┘
                                 │
                                 ▼
                   ┌──────────────────────────┐
                   │  2. Regex Scraper         │  Pattern-match phone, address, NAICS, etc.
                   │     (scraper.py)           │
                   └─────────────┬────────────┘
                                 │
                                 ▼
                   ┌──────────────────────────┐
                   │  3. Vector Search         │  Query Databricks VSI for relevant chunks
                   │     Retrieval (RAG)        │  Index: hackathon_521302447211702.aa
                   │     (retrieval.py)         │         .vsi_using_bge_large_on_sections
                   └─────────────┬────────────┘
                                 │
                                 ▼
                   ┌──────────────────────────┐
                   │  4. LLM Extraction        │  Structured extraction via Databricks-hosted LLM
                   │     (llm_extract.py)       │  Model: databricks-claude-sonnet-4-5
                   └─────────────┬────────────┘
                                 │
                                 ▼
                   ┌──────────────────────────┐
                   │  5. Enrichment            │  Derive is_manufacturer, is_open,
                   │     (enrichment.py)        │  expand NAICS → industry description
                   └─────────────┬────────────┘
                                 │
                                 ▼
                   ┌──────────────────────────┐
                   │  6. Self-Evaluation       │  Compute fill_rate, flag missing fields
                   │     (evaluate.py)          │
                   └─────────────┬────────────┘
                                 │
                          fill_rate < 0.5?
                         ┌───────┴───────┐
                         │ YES           │ NO
                         ▼               ▼
          ┌──────────────────────┐     DONE ──► BusinessRecord (JSON)
          │  7. Web Fallback     │
          │     (web_search.py)  │  DuckDuckGo / Serper API
          └──────────┬──────────┘
                     │
                     ▼
          Re-extract with LLM (step 4+5)
                     │
                     ▼
          Re-evaluate (step 6) ──► DONE
```

### Two Execution Modes

| Mode | File | Best for |
|------|------|----------|
| **Fixed pipeline** (LangGraph) | `workflow.py` | Production / batch — deterministic, predictable |
| **ReAct supervisor** (LangChain Agent) | `supervisor.py` | Interactive exploration — LLM decides tool order |

### Output Schema — `BusinessRecord`

| Field | Type | Description |
|-------|------|-------------|
| `business_name` | `str` | Legal entity name |
| `parent_name` | `str` | Parent company (if any) |
| `business_phone` | `str` | Primary phone number |
| `address` | `str` | Street address |
| `city` | `str` | City |
| `state` | `str` | State |
| `zip_code` | `str` | ZIP/postal code |
| `industry_description` | `str` | Industry narrative |
| `naics_sic_codes` | `list[str]` | NAICS / SIC codes |
| `num_employees` | `str` | Employee count |
| `is_manufacturer` | `bool` | Manufacturing indicator |
| `risk_factor_keywords` | `list[str]` | Risk keywords from filings |
| `revenue_mentions` | `list[str]` | Revenue figures mentioned |
| `website_domain` | `str` | Company website |
| `business_purpose_summary` | `str` | Business description |
| `is_open` | `bool` | Whether business is currently operating |

---

## Quick Start

### Prerequisites

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/) (Python package manager)
- [Databricks CLI](https://docs.databricks.com/aws/en/dev-tools/cli/install)

### 1. Set up authentication

```bash
uv run quickstart
```

Or manually:

```bash
databricks auth login --host https://dbc-7c699d6a-b2cd.cloud.databricks.com
```

Verify your profile works:

```bash
databricks auth profiles
```

### 2. Configure environment

Copy `.env.example` to `.env` (if not already done) and fill in the values:

```bash
cp .env.example .env
```

Key variables in `.env`:

```env
DATABRICKS_CONFIG_PROFILE=hackathon_ws

MLFLOW_EXPERIMENT_ID=2932940041946963
MLFLOW_TRACKING_URI="databricks://hackathon_ws"

SEC_VECTOR_SEARCH_ENDPOINT=sec_endpoint
SEC_VECTOR_SEARCH_INDEX=hackathon_521302447211702.aa.vsi_using_bge_large_on_sections
```

### 3. Run locally

Start both the backend API server and the Streamlit UI:

```bash
uv run start-app
```

This launches:
- **Backend** at `http://localhost:8000` (MLflow AgentServer + FastAPI)
- **Streamlit UI** at `http://localhost:3000` (opens in browser)

### 4. Run via CLI (no UI)

Extract from inline text:

```bash
uv run run-sec-extraction --document "ACME CORP 10-K Filing ..."
```

Extract from a file:

```bash
uv run run-sec-extraction --file path/to/filing.txt --output results.json
```

Use the ReAct supervisor instead of the fixed pipeline:

```bash
uv run run-sec-extraction --file path/to/filing.txt --mode supervisor
```

### 5. Query the API directly

Non-streaming:

```bash
curl -X POST http://localhost:8000/invocations \
  -H "Content-Type: application/json" \
  -d '{ "input": [{ "role": "user", "content": "Extract: ACME CORP, 123 Main St..." }] }'
```

Streaming:

```bash
curl -X POST http://localhost:8000/invocations \
  -H "Content-Type: application/json" \
  -d '{ "input": [{ "role": "user", "content": "Extract: ACME CORP..." }], "stream": true }'
```

---

## Advanced Server Options

```bash
uv run start-server --reload    # Hot-reload on code changes
uv run start-server --port 8001 # Custom port
uv run start-server --workers 4 # Multi-worker
```

---

## Evaluating the Agent

Update test cases in `agent_server/evaluate_agent.py`, then:

```bash
uv run agent-evaluate
```

Results are logged to your MLflow experiment — open the Databricks UI to inspect.

---

## Deploying to Databricks Apps

### Validate

```bash
databricks bundle validate --profile hackathon_ws
```

### Deploy

```bash
databricks bundle deploy --profile hackathon_ws
```

### Start

```bash
databricks bundle run agent_openai_agents_sdk_multiagent --profile hackathon_ws
```

### View logs

```bash
databricks apps logs agent-sec-extraction --follow --profile hackathon_ws
```

### Query the deployed app

```bash
TOKEN=$(databricks auth token --profile hackathon_ws | jq -r .access_token)

curl -X POST https://<app-url>.databricksapps.com/invocations \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{ "input": [{ "role": "user", "content": "Extract: ..." }] }'
```

---

## Key Dependencies

| Package | Purpose |
|---------|---------|
| `langchain` / `langchain-core` | Agent framework, prompt templates, tool interface |
| `langchain-openai` | ChatOpenAI for Databricks model serving endpoints |
| `langgraph` | StateGraph for the fixed extraction pipeline |
| `mlflow` | Tracing, experiment tracking, AgentServer |
| `databricks-agents` | MLflow Responses API integration |
| `duckduckgo-search` | Web search fallback |
| `streamlit` | Frontend UI |

---

## Extending the Pipeline

Each tool in `agent_server/sec_extraction/tools/` is a standalone module. To add a new step:

1. Create a new file in `tools/` (e.g., `tools/my_new_step.py`)
2. Implement a function that takes state inputs and returns extracted data
3. Add the node in `workflow.py`:
   ```python
   graph.add_node("my_step", my_step_node)
   graph.add_edge("previous_step", "my_step")
   ```
4. Add new fields to `BusinessRecord` in `schemas.py` if needed
5. Wrap it as a `@tool` in `supervisor.py` for the ReAct agent

---

## Collaboration

To collaborate with others on this repo:

1. Each team member runs `databricks auth login` to set up their own profile
2. Copy `.env.example` to `.env` and fill in their profile name
3. The `.env` file is gitignored — credentials stay local
4. Shared resources (experiment ID, vector search index) are configured in `databricks.yml`

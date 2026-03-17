"""
SEC Filing Attribute Extractor — Streamlit UI

Provides a rich interface to extract structured business attributes from SEC
filings.  Supports pasting raw text, uploading files, or searching by company
name.  Includes a "Quick Scrape" mode (regex-only, no backend needed) and a
full "AI Extraction" mode that calls the agent backend.
"""

import json
import os
import re
import sys
import time

import pandas as pd
import requests
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from agent_server.sec_extraction.attribute_registry import get_registry

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="SEC Filing Extractor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown(
    """
<style>
    /* Header */
    .block-container { padding-top: 2rem; }

    /* Confidence colours */
    .conf-high  { color: #22c55e; font-weight: 700; }
    .conf-med   { color: #eab308; font-weight: 700; }
    .conf-low   { color: #ef4444; font-weight: 700; }

    /* Source badges */
    .badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 4px;
    }
    .badge-scraper       { background: #dbeafe; color: #1d4ed8; }
    .badge-vector_search { background: #dcfce7; color: #15803d; }
    .badge-web_search    { background: #fef3c7; color: #a16207; }
    .badge-llm_inference { background: #f3e8ff; color: #7e22ce; }

    /* Status */
    .status-open   { color: #22c55e; font-weight: 700; font-size: 1.3rem; }
    .status-closed { color: #ef4444; font-weight: 700; font-size: 1.3rem; }

    /* Metric override */
    [data-testid="stMetric"] { background: rgba(255,255,255,0.04); border-radius: 8px; padding: 12px; }

    div.stButton > button[kind="primary"] { width: 100%; }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

API_URL = os.environ.get("API_PROXY", "http://localhost:8000/invocations")

_registry = get_registry()
ATTRIBUTE_LABELS = _registry.get_labels_for_ui()

QUICK_SCRAPE_PATTERNS = {
    "phone": [
        r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}",
        r"\+?1?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}",
    ],
    "website": [
        r"(?:https?://)?(?:www\.)?([a-zA-Z0-9][-a-zA-Z0-9]*\.[a-zA-Z]{2,}(?:\.[a-zA-Z]{2,})?)",
    ],
    "location_employee_count": [
        r"(?:approximately|about|nearly|over|more than)?\s*(\d{1,3}(?:,\d{3})*)\s+(?:full[- ]?time\s+)?employees",
        r"(\d{1,3}(?:,\d{3})*)\s+(?:people|personnel|workers)",
    ],
    "revenue": [
        r"\$\s*[\d,]+(?:\.\d+)?\s*(?:billion|million|thousand|B|M|K)",
        r"(?:revenue|net\s+sales|total\s+revenue)\s+(?:of|was|were|totaled)?\s*\$\s*[\d,]+(?:\.\d+)?",
    ],
    "street": [
        r"\d+\s+[\w\s]+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Boulevard|Blvd|Way|Lane|Ln|Place|Pl|Suite|Ste)\.?(?:\s*,?\s*(?:Suite|Ste)\.?\s*\d+)?",
    ],
    "postal_code": [
        r"\b[A-Z]{2}\s+(\d{5}(?:-\d{4})?)\b",
    ],
    "primary_sic_code_id": [
        r"(?:SIC)\s*(?:code|Code)?:?\s*(\d{4,6})",
        r"Standard\s+Industrial\s+Classification\s*(?:code)?\s*:?\s*(\d{4})",
    ],
    "primary_naics_code_id": [
        r"(?:NAICS)\s*(?:code|Code)?:?\s*(\d{4,6})",
    ],
    "name": [
        r"(?:EXACT NAME OF REGISTRANT|Company Name|Registrant)[:\s]+([A-Z][\w\s&.,'-]+(?:Inc|Corp|LLC|Ltd|Co|LP|Company|Corporation|Group|Holdings)\.?)",
    ],
    "company_ein": [
        r"(?:EIN|Employer\s+Identification\s+Number)[:\s]*(\d{2}-?\d{7})",
    ],
    "company_year_founded": [
        r"(?:founded|incorporated|established)\s+(?:in\s+)?(\d{4})",
    ],
    "cik": [
        r"(?:CIK|Central\s+Index\s+Key)[:\s]*(\d{7,10})",
        r"Commission\s+File\s+Number[:\s]*([\d-]+)",
    ],
}

SAMPLE_FILING = """UNITED STATES SECURITIES AND EXCHANGE COMMISSION
Washington, D.C. 20549

FORM 10-K

ANNUAL REPORT PURSUANT TO SECTION 13 OR 15(d) OF THE
SECURITIES EXCHANGE ACT OF 1934

For the fiscal year ended December 31, 2024

Commission File Number: 001-38846

EXACT NAME OF REGISTRANT: PINNACLE MANUFACTURING CORPORATION
State of incorporation: Delaware
IRS Employer Identification Number: 82-4921573

Address of principal executive offices:
4200 Industrial Boulevard, Suite 300, Austin, TX 78745
Telephone: (512) 555-7890
Website: www.pinnaclemfg.com

ITEM 1. BUSINESS

Pinnacle Manufacturing Corporation ("the Company") is a leading manufacturer
and distributor of precision-engineered industrial components. Founded in 2003,
the Company designs, manufactures, and sells high-performance fasteners,
brackets, and structural components used in aerospace, automotive, and
construction industries.

The Company operates three production facilities in Austin, TX, Denver, CO,
and Charlotte, NC, with a combined manufacturing floor space of approximately
450,000 square feet.

As of December 31, 2024, the Company employed approximately 2,850 full-time
employees and 340 part-time and contract workers.

Standard Industrial Classification code: 3462
NAICS Code: 332111

ITEM 1A. RISK FACTORS

The following risk factors could materially affect our business:

- Supply chain disruptions: The Company relies on specialized raw materials
  including titanium alloys and high-grade steel from a limited number of
  suppliers. Any disruption could impact production.

- Customer concentration: Approximately 35% of revenue is derived from our
  top three customers in the aerospace sector.

- Regulatory compliance: Our products must meet stringent quality standards
  including AS9100D and ISO 9001:2015 certifications.

- Cybersecurity threats: Increasing sophistication of cyber attacks poses
  risks to our manufacturing control systems and customer data.

- Economic downturn sensitivity: Demand for industrial components is
  cyclical and closely tied to capital expenditure trends.

ITEM 6. SELECTED FINANCIAL DATA

For the fiscal year ended December 31, 2024:
- Total revenue was $487.3 million, an increase of 12% from $435.1 million
  in the prior year.
- Net income was $52.8 million compared to $41.2 million in 2023.
- Revenue from aerospace segment: $218.4 million
- Revenue from automotive segment: $156.7 million
- Revenue from construction segment: $112.2 million

Pinnacle Manufacturing Corporation is a subsidiary of Apex Industrial Group, Inc.
"""


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def extract_response_text(data: dict) -> str:
    """Pull the assistant's text from an MLflow Responses API payload."""
    for item in reversed(data.get("output", [])):
        if item.get("type") == "message" and item.get("role") == "assistant":
            content = item.get("content")
            if isinstance(content, list):
                for c in content:
                    if isinstance(c, dict) and c.get("type") == "output_text":
                        return c.get("text", "")
            if isinstance(content, str):
                return content
    texts = []
    for item in data.get("output", []):
        content = item.get("content")
        if isinstance(content, list):
            for c in content:
                if isinstance(c, dict) and c.get("text"):
                    texts.append(c["text"])
        elif isinstance(content, str):
            texts.append(content)
    return "\n".join(texts) if texts else json.dumps(data, indent=2)


def _agent_record_to_ui_format(record: dict, evaluation: dict | None = None) -> dict:
    """Transform agent's extraction record to UI display format using the registry."""
    ev = evaluation or {}
    conf = ev.get("confidence", 0.8)
    attrs: dict = {}

    for key, val in record.items():
        if key not in ATTRIBUTE_LABELS:
            continue
        if isinstance(val, list):
            display_val = ", ".join(str(v) for v in val) if val else None
        else:
            display_val = val
        attrs[key] = {
            "value": display_val,
            "confidence": conf if display_val is not None else 0.0,
            "source": "llm_inference",
            "evidence": "",
        }

    # Fill in missing registry attributes so the UI can show them
    for key in ATTRIBUTE_LABELS:
        if key not in attrs:
            attrs[key] = {"value": None, "confidence": 0.0, "source": "llm_inference", "evidence": ""}

    biz_name = record.get("name") or record.get("company_name") or "Unknown Company"
    in_biz = record.get("in_business", "")
    status = "closed" if str(in_biz).lower() in ("no", "false", "closed") else "open"
    reasoning = "; ".join(ev.get("issues", [])) or "Extracted from SEC filing"

    return {
        "company_name": biz_name,
        "filing_type": "Unknown",
        "attributes": attrs,
        "business_status": status,
        "status_reasoning": reasoning,
    }


def _is_agent_record(d: dict) -> bool:
    """Check if a dict looks like an agent extraction record (has registry fields)."""
    registry_keys = set(_registry.attribute_names)
    return len(set(d.keys()) & registry_keys) >= 3


def parse_extraction_json(text: str) -> dict | None:
    """Extract the JSON block from the agent's markdown-wrapped response and normalize to UI format."""
    def _try_parse(raw: str) -> dict | None:
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return None
        if not isinstance(parsed, dict):
            return None
        record = parsed.get("record", parsed)
        evaluation = parsed.get("evaluation") if "evaluation" in parsed else None
        if _is_agent_record(record):
            return _agent_record_to_ui_format(record, evaluation)
        return parsed

    # Try ```json ... ``` block first
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if match:
        result = _try_parse(match.group(1).strip())
        if result:
            return result

    # Try the entire text as JSON
    return _try_parse(text)


def confidence_class(score: float) -> str:
    if score >= 0.8:
        return "conf-high"
    if score >= 0.5:
        return "conf-med"
    return "conf-low"


def source_badge(source: str) -> str:
    css = f"badge-{source}" if f"badge-{source}" in (
        "badge-scraper",
        "badge-vector_search",
        "badge-web_search",
        "badge-llm_inference",
    ) else "badge-llm_inference"
    return f'<span class="badge {css}">{source.replace("_", " ").title()}</span>'


def quick_scrape(text: str) -> dict:
    """Run regex-only extraction (no backend needed)."""
    results = {}
    for attr, patterns in QUICK_SCRAPE_PATTERNS.items():
        all_matches = []
        for pattern in patterns:
            all_matches.extend(re.findall(pattern, text, re.IGNORECASE | re.MULTILINE))
        unique = list(dict.fromkeys(str(m).strip() for m in all_matches if str(m).strip()))[:5]
        if unique:
            results[attr] = {
                "value": unique[0] if len(unique) == 1 else ", ".join(unique),
                "confidence": 0.85,
                "source": "scraper",
                "evidence": f"Regex matched {len(unique)} occurrence(s)",
            }
        else:
            results[attr] = {
                "value": None,
                "confidence": 0.0,
                "source": "scraper",
                "evidence": "No regex match found",
            }

    for attr in ATTRIBUTE_LABELS:
        if attr not in results:
            results[attr] = {
                "value": None,
                "confidence": 0.0,
                "source": "scraper",
                "evidence": "No regex pattern for this attribute — requires AI extraction",
            }

    is_mfg = bool(re.search(
        r"(?:manufactur|produc(?:tion|e)|fabricat|assembl|factory|plant)",
        text,
        re.IGNORECASE,
    ))
    results["is_manufacturer"] = {
        "value": is_mfg,
        "confidence": 0.7 if is_mfg else 0.3,
        "source": "scraper",
        "evidence": "Keyword match for manufacturing terms" if is_mfg else "No manufacturing keywords found",
    }

    return {
        "company_name": results.get("name", {}).get("value") or results.get("company_name", {}).get("value") or "Unknown",
        "filing_type": "Unknown",
        "attributes": results,
        "business_status": "open",
        "status_reasoning": "No closure indicators found (regex-only mode)",
    }


# ---------------------------------------------------------------------------
# Display functions
# ---------------------------------------------------------------------------


def render_company_header(result: dict):
    """Top-level company overview bar."""
    col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
    with col1:
        st.markdown(f"### {result.get('company_name', 'Unknown Company')}")
    with col2:
        filing = result.get("filing_type", "N/A")
        st.metric("Filing Type", filing)
    with col3:
        status = result.get("business_status", "unknown")
        label = "Open" if status == "open" else "Closed" if status == "closed" else status.title()
        st.metric("Status", label)
    with col4:
        attrs = result.get("attributes", {})
        found = sum(1 for a in attrs.values() if a.get("value") is not None)
        st.metric("Attributes Found", f"{found} / {len(attrs)}")

    reasoning = result.get("status_reasoning", "")
    if reasoning:
        st.caption(f"Status reasoning: {reasoning}")


def render_attribute_cards(result: dict):
    """3-column grid of attribute cards with confidence bars."""
    attrs = result.get("attributes", {})
    items = list(attrs.items())

    for row_start in range(0, len(items), 3):
        cols = st.columns(3)
        for col_idx, (attr, details) in enumerate(items[row_start : row_start + 3]):
            icon, label = ATTRIBUTE_LABELS.get(attr, ("📌", attr.replace("_", " ").title()))
            with cols[col_idx]:
                with st.container(border=True):
                    st.markdown(f"**{icon} {label}**")

                    value = details.get("value")
                    confidence = details.get("confidence", 0)
                    source = details.get("source", "unknown")
                    evidence = details.get("evidence", "")

                    if value is None:
                        st.markdown("*Not found*")
                    elif isinstance(value, bool):
                        st.markdown(f"**{'Yes' if value else 'No'}**")
                    elif isinstance(value, list):
                        st.markdown(f"**{', '.join(str(v) for v in value)}**")
                    else:
                        display = str(value)
                        if len(display) > 120:
                            display = display[:120] + "..."
                        st.markdown(f"**{display}**")

                    conf_color = confidence_class(confidence)
                    st.progress(confidence)
                    st.markdown(
                        f'<span class="{conf_color}">{confidence:.0%}</span> '
                        f"{source_badge(source)}",
                        unsafe_allow_html=True,
                    )

                    if evidence:
                        with st.expander("Evidence"):
                            st.caption(evidence)


def render_insights(result: dict):
    """Charts and analytics for the extraction results."""
    attrs = result.get("attributes", {})
    if not attrs:
        return

    st.markdown("---")
    st.subheader("📊 Extraction Insights")

    tab_conf, tab_src, tab_risk, tab_table = st.tabs([
        "Confidence Scores",
        "Source Distribution",
        "Risk Factors",
        "Data Table",
    ])

    with tab_conf:
        conf_df = pd.DataFrame([
            {
                "Attribute": ATTRIBUTE_LABELS.get(k, ("", k))[1],
                "Confidence": v.get("confidence", 0),
            }
            for k, v in attrs.items()
        ])
        conf_df = conf_df.sort_values("Confidence", ascending=True)
        st.bar_chart(conf_df.set_index("Attribute"), horizontal=True, height=420)

        avg_conf = conf_df["Confidence"].mean()
        found_count = sum(1 for v in attrs.values() if v.get("value") is not None)
        c1, c2, c3 = st.columns(3)
        c1.metric("Average Confidence", f"{avg_conf:.0%}")
        c2.metric("Attributes Found", f"{found_count}/{len(attrs)}")
        c3.metric("High Confidence (>=80%)", sum(1 for v in attrs.values() if v.get("confidence", 0) >= 0.8))

    with tab_src:
        source_counts: dict[str, int] = {}
        for v in attrs.values():
            s = v.get("source", "unknown")
            source_counts[s] = source_counts.get(s, 0) + 1
        src_df = pd.DataFrame(
            [{"Source": k.replace("_", " ").title(), "Count": v} for k, v in source_counts.items()]
        )
        if not src_df.empty:
            st.bar_chart(src_df.set_index("Source"), height=300)
        st.caption("Distribution of extraction methods used across all attributes.")

    with tab_risk:
        risk = attrs.get("risk_factor_keywords", {})
        risk_val = risk.get("value")
        if risk_val:
            if isinstance(risk_val, list):
                for rf in risk_val:
                    st.markdown(f"- ⚠️ {rf}")
            elif isinstance(risk_val, str):
                for line in risk_val.split(","):
                    line = line.strip()
                    if line:
                        st.markdown(f"- ⚠️ {line}")
        else:
            st.info("No risk factors extracted.")

        purpose = attrs.get("business_purpose_summary", {}).get("value")
        if purpose:
            st.markdown("---")
            st.markdown("**Business Purpose**")
            st.info(purpose)

    with tab_table:
        table_data = []
        for k, v in attrs.items():
            _, label = ATTRIBUTE_LABELS.get(k, ("", k))
            table_data.append({
                "Attribute": label,
                "Value": str(v.get("value", "N/A")),
                "Confidence": f"{v.get('confidence', 0):.0%}",
                "Source": v.get("source", "N/A"),
                "Evidence": v.get("evidence", ""),
            })
        st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)


def render_exports(result: dict):
    """Download buttons for JSON and CSV."""
    st.markdown("---")
    col1, col2, _ = st.columns([1, 1, 4])
    with col1:
        st.download_button(
            "📥 Download JSON",
            data=json.dumps(result, indent=2, default=str),
            file_name="sec_extraction_result.json",
            mime="application/json",
        )
    with col2:
        attrs = result.get("attributes", {})
        rows = []
        for k, v in attrs.items():
            rows.append({
                "attribute": k,
                "value": v.get("value"),
                "confidence": v.get("confidence"),
                "source": v.get("source"),
                "evidence": v.get("evidence"),
            })
        csv = pd.DataFrame(rows).to_csv(index=False)
        st.download_button(
            "📥 Download CSV",
            data=csv,
            file_name="sec_extraction_result.csv",
            mime="text/csv",
        )


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "extraction_result" not in st.session_state:
    st.session_state.extraction_result = None
if "raw_response" not in st.session_state:
    st.session_state.raw_response = None
if "filing_text" not in st.session_state:
    st.session_state.filing_text = ""

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## SEC Filing Extractor")
    st.caption("AI-powered attribute extraction from SEC filings")
    st.markdown("---")

    st.markdown("### Configuration")
    api_url = st.text_input("Backend API URL", value=API_URL, help="URL of the agent backend `/invocations` endpoint")

    st.markdown("---")
    st.markdown(f"### Target Attributes ({len(ATTRIBUTE_LABELS)})")
    for group_name, group_attrs in _registry.groups.items():
        with st.expander(f"{group_name} ({len(group_attrs)})"):
            for a in group_attrs:
                icon, label = ATTRIBUTE_LABELS.get(a["attribute"], ("📌", a["display_name"]))
                st.markdown(f"{icon} {label}")

    st.markdown("---")
    st.markdown("### Extraction Modes")
    st.markdown(
        "**Quick Scrape** — regex only, instant, no backend\n\n"
        "**AI Extraction** — full agent pipeline with RAG, web search, and LLM reasoning"
    )

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

st.title("🔍 SEC Filing Attribute Extractor")
st.caption(
    "Extract structured business attributes from SEC filings using a multi-agent AI pipeline — "
    "regex scraper, vector search RAG, web search, and LLM reasoning."
)

# ---------------------------------------------------------------------------
# Input section
# ---------------------------------------------------------------------------

st.markdown("### Input")
input_tab1, input_tab2, input_tab3 = st.tabs(["📝 Paste Text", "📁 Upload File", "🔎 Search by Company"])

with input_tab1:
    col_text, col_help = st.columns([4, 1])
    with col_text:
        filing_input = st.text_area(
            "Paste SEC filing text (10-K, 10-Q, 8-K, etc.)",
            value=st.session_state.filing_text,
            height=350,
            placeholder="Paste the full text or relevant sections of an SEC filing here...",
            key="text_input",
        )
    with col_help:
        st.markdown("#### Quick Start")
        if st.button("📄 Load Sample", use_container_width=True):
            st.session_state.filing_text = SAMPLE_FILING
            st.rerun()
        st.caption(
            "Load a sample 10-K filing to test the extraction pipeline immediately."
        )

with input_tab2:
    uploaded = st.file_uploader(
        "Upload an SEC filing document",
        type=["txt", "html", "htm", "csv"],
        help="Supports .txt, .html, and .csv files",
    )
    if uploaded:
        raw_bytes = uploaded.read()
        filing_input = raw_bytes.decode("utf-8", errors="ignore")
        st.text_area("File preview (first 3 000 chars)", filing_input[:3000], height=200, disabled=True)

with input_tab3:
    company_query = st.text_input(
        "Company name or CIK number",
        placeholder="e.g. Pinnacle Manufacturing Corp or CIK 0001234567",
    )
    if company_query:
        filing_input = (
            f"Search for SEC filings and extract all business attributes for: {company_query}. "
            "Use the vector search to find their latest filing, then extract all 12 attributes."
        )

# Determine the final input text
input_text = ""
if "filing_input" in dir() and filing_input:
    input_text = filing_input
elif st.session_state.filing_text:
    input_text = st.session_state.filing_text

# ---------------------------------------------------------------------------
# Action buttons
# ---------------------------------------------------------------------------

st.markdown("")
btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 4])

with btn_col1:
    do_quick = st.button("⚡ Quick Scrape", use_container_width=True, help="Regex-only, instant, no backend needed")

with btn_col2:
    do_full = st.button(
        "🚀 AI Extraction",
        type="primary",
        use_container_width=True,
        help="Full agent pipeline — scraper + vector search + web search + LLM",
    )

# ---------------------------------------------------------------------------
# Quick Scrape
# ---------------------------------------------------------------------------

if do_quick:
    if not input_text.strip():
        st.error("Please provide filing text first (paste or upload).")
    else:
        with st.spinner("Running regex extraction..."):
            result = quick_scrape(input_text)
            st.session_state.extraction_result = result
            st.session_state.raw_response = None

# ---------------------------------------------------------------------------
# Full AI Extraction
# ---------------------------------------------------------------------------

if do_full:
    if not input_text.strip():
        st.error("Please provide filing text or a company name first.")
    else:
        progress = st.progress(0, text="Initializing extraction pipeline...")

        prompt = (
            "Extract ALL business attributes from the following SEC filing text. "
            "Use scraper_extract first for structured patterns (phone, address, employees, "
            "revenue, NAICS/SIC, website, business name). Then use vector search and web search "
            "for remaining attributes. Return the complete structured JSON with all 12 attributes, "
            "business_status, and status_reasoning.\n\n"
            "SEC Filing Text:\n"
            "---\n"
            f"{input_text}\n"
            "---"
        )

        progress.progress(15, text="Calling extraction agent...")

        try:
            t0 = time.time()
            resp = requests.post(
                api_url,
                json={"input": [{"role": "user", "content": prompt}]},
                timeout=180,
            )
            elapsed = time.time() - t0

            progress.progress(80, text="Parsing agent response...")

            if resp.status_code == 200:
                data = resp.json()
                text = extract_response_text(data)
                result = parse_extraction_json(text)

                progress.progress(100, text=f"Done in {elapsed:.1f}s")
                time.sleep(0.5)
                progress.empty()

                if result:
                    st.session_state.extraction_result = result
                    st.session_state.raw_response = text
                else:
                    st.session_state.extraction_result = None
                    st.session_state.raw_response = text
                    st.warning("Could not parse structured JSON from the agent response.")
            else:
                progress.empty()
                st.error(f"Backend returned HTTP {resp.status_code}. Check that the agent server is running.")
                try:
                    st.code(resp.text[:2000])
                except Exception:
                    pass

        except requests.ConnectionError:
            progress.empty()
            st.error(
                "Cannot connect to the agent backend. "
                f"Make sure the server is running at **{api_url}**.\n\n"
                "Start it with: `uv run start-server`"
            )
        except requests.Timeout:
            progress.empty()
            st.error("Request timed out (180 s). Try with a shorter document or check backend logs.")

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

if st.session_state.extraction_result:
    st.markdown("---")
    render_company_header(st.session_state.extraction_result)
    st.markdown("")
    render_attribute_cards(st.session_state.extraction_result)
    render_insights(st.session_state.extraction_result)
    render_exports(st.session_state.extraction_result)

if st.session_state.raw_response and not st.session_state.extraction_result:
    st.markdown("---")
    st.subheader("Raw Agent Response")
    st.markdown(st.session_state.raw_response)

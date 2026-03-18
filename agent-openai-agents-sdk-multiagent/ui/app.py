"""
SEC Pilot — Demo Dashboard
Data Axle | Hackathon 2026
"""

import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from agent_server.sec_extraction.attribute_registry import get_registry

_registry = get_registry()
ATTRIBUTE_LABELS = _registry.get_labels_for_ui()
TARGET_ATTR_COUNT = 46  # SEC_EXTRACTABLE_ATTRS in llm_extract.py

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="SEC Pilot — Data Axle",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------

BRAND_DARK = "#0B1929"
BRAND_ACCENT = "#00C853"
BRAND_ACCENT2 = "#00B8D4"
BRAND_CARD_BG = "#FFFFFF"
BRAND_SURFACE = "#F4F6F9"
BRAND_TEXT = "#1A2332"
BRAND_MUTED = "#6B7B8D"

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', -apple-system, sans-serif;
    }}
    .block-container {{
        padding-top: 0rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }}
    header[data-testid="stHeader"] {{
        background: transparent !important;
        backdrop-filter: none !important;
    }}

    /* Hide deploy button + Streamlit branding */
    .stDeployButton, #MainMenu, footer,
    button[kind="header"], [data-testid="stToolbar"] {{
        display: none !important;
        visibility: hidden !important;
    }}

    /* Hero */
    .hero {{
        background: linear-gradient(135deg, {BRAND_DARK} 0%, #132F4C 60%, #1A3A5C 100%);
        padding: 2rem 2.5rem 1.2rem 2.5rem;
        border-radius: 0 0 16px 16px;
        margin: -1rem -1rem 1.5rem -1rem;
        color: white;
        display: flex;
        align-items: center;
        justify-content: space-between;
        position: relative;
        z-index: 1;
    }}
    .hero-left {{ flex: 1; }}
    .hero-right {{
        text-align: right;
        font-size: 0.85rem;
        color: #94A3B8;
    }}
    .hero-brand {{
        font-size: 1.1rem;
        letter-spacing: 4px;
        text-transform: uppercase;
        color: {BRAND_ACCENT};
        font-weight: 800;
        margin-bottom: 2px;
    }}
    .hero-title {{
        font-size: 1.8rem;
        font-weight: 800;
        margin: 0;
        line-height: 1.2;
    }}
    .hero-sub {{
        font-size: 0.9rem;
        color: #94A3B8;
        margin-top: 4px;
    }}
    .hero-badge {{
        display: inline-block;
        background: rgba(0,200,83,0.15);
        color: {BRAND_ACCENT};
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-top: 8px;
        border: 1px solid rgba(0,200,83,0.3);
    }}

    /* KPI */
    .kpi-card {{
        background: {BRAND_CARD_BG};
        border-radius: 12px;
        padding: 1.1rem 1.2rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        border: 1px solid #E8ECF0;
        text-align: center;
    }}
    .kpi-value {{
        font-size: 2rem;
        font-weight: 800;
        color: {BRAND_TEXT};
        line-height: 1;
        margin-bottom: 4px;
    }}
    .kpi-value-green {{ color: {BRAND_ACCENT}; }}
    .kpi-value-blue  {{ color: {BRAND_ACCENT2}; }}
    .kpi-label {{
        font-size: 0.75rem;
        color: {BRAND_MUTED};
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}

    /* Section headers */
    .section-header {{
        font-size: 1.25rem;
        font-weight: 700;
        color: {BRAND_TEXT};
        margin: 1.5rem 0 0.8rem 0;
        display: flex;
        align-items: center;
        gap: 8px;
    }}
    .section-header span {{
        background: linear-gradient(135deg, {BRAND_ACCENT}, {BRAND_ACCENT2});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}

    /* Architecture — dark theme */
    .arch-container {{
        background: linear-gradient(135deg, {BRAND_DARK}, #132F4C);
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        border: 1px solid #1E3A5F;
    }}
    .arch-flow {{
        display: flex;
        align-items: stretch;
        justify-content: center;
        gap: 0;
        flex-wrap: nowrap;
        overflow-x: auto;
        padding: 0.5rem 0;
    }}
    .arch-node {{
        flex: 0 0 auto;
        min-width: 125px;
        padding: 0.8rem 0.9rem;
        border-radius: 10px;
        text-align: center;
        transition: transform 0.2s;
    }}
    .arch-node:hover {{ transform: translateY(-3px); }}
    .arch-node-icon {{ font-size: 1.4rem; margin-bottom: 4px; }}
    .arch-node-title {{
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.3px;
    }}
    .arch-node-desc {{
        font-size: 0.65rem;
        margin-top: 2px;
        opacity: 0.85;
    }}
    .arch-arrow {{
        display: flex;
        align-items: center;
        font-size: 1.1rem;
        color: #4A6A8A;
        padding: 0 3px;
        flex: 0 0 auto;
    }}
    .node-ingest  {{ background: rgba(59,130,246,0.15); color: #93C5FD; border: 1px solid rgba(59,130,246,0.3); }}
    .node-scrape  {{ background: rgba(249,115,22,0.15); color: #FDBA74; border: 1px solid rgba(249,115,22,0.3); }}
    .node-filter  {{ background: rgba(34,197,94,0.15);  color: #86EFAC; border: 1px solid rgba(34,197,94,0.3); }}
    .node-llm     {{ background: rgba(168,85,247,0.15); color: #C4B5FD; border: 1px solid rgba(168,85,247,0.3); }}
    .node-enrich  {{ background: rgba(236,72,153,0.15); color: #F9A8D4; border: 1px solid rgba(236,72,153,0.3); }}
    .node-eval    {{ background: rgba(20,184,166,0.15); color: #5EEAD4; border: 1px solid rgba(20,184,166,0.3); }}
    .node-web     {{ background: rgba(234,179,8,0.15);  color: #FDE68A; border: 1px solid rgba(234,179,8,0.3); }}
    .node-delta   {{ background: rgba(59,130,246,0.15); color: #93C5FD; border: 1px solid rgba(59,130,246,0.3); }}

    .stats-bar {{
        display: flex;
        gap: 1.2rem;
        flex-wrap: wrap;
        margin-top: 1rem;
    }}
    .stat-chip {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(255,255,255,0.06);
        padding: 6px 14px;
        border-radius: 8px;
        font-size: 0.78rem;
        color: #94A3B8;
        font-weight: 500;
        border: 1px solid rgba(255,255,255,0.08);
    }}
    .stat-chip b {{ color: {BRAND_ACCENT}; }}

    /* Streamlit metric overrides */
    [data-testid="stMetric"] {{
        background: {BRAND_CARD_BG} !important;
        border-radius: 10px;
        padding: 14px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        border: 1px solid #E8ECF0;
    }}
    [data-testid="stMetric"] label,
    [data-testid="stMetric"] [data-testid="stMetricLabel"] {{
        color: {BRAND_MUTED} !important;
    }}
    [data-testid="stMetric"] [data-testid="stMetricValue"] {{
        color: {BRAND_TEXT} !important;
    }}
    div[data-testid="stExpander"] {{
        background: {BRAND_CARD_BG};
        border-radius: 10px;
        border: 1px solid #E8ECF0;
    }}
    .dataframe {{ font-size: 0.85rem; }}
    button[data-baseweb="tab"] {{
        font-weight: 600 !important;
        font-size: 0.85rem !important;
    }}

    /* Live result cards — grouped by category color */
    .res-group-title {{
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin: 1rem 0 0.4rem 0;
        padding: 4px 12px;
        border-radius: 6px;
        display: inline-block;
    }}
    .res-group-identity  {{ background: #EFF6FF; color: #1E40AF; }}
    .res-group-location  {{ background: #F0FDF4; color: #166534; }}
    .res-group-financial {{ background: #FFF7ED; color: #C2410C; }}
    .res-group-industry  {{ background: #FAF5FF; color: #7E22CE; }}
    .res-group-other     {{ background: #F1F5F9; color: #475569; }}

    .result-card {{
        background: {BRAND_CARD_BG};
        border-radius: 10px;
        padding: 0.65rem 1rem;
        margin-bottom: 6px;
    }}
    .result-card-identity  {{ border-left: 3px solid #3B82F6; border: 1px solid #DBEAFE; }}
    .result-card-location  {{ border-left: 3px solid #22C55E; border: 1px solid #DCFCE7; }}
    .result-card-financial {{ border-left: 3px solid #F97316; border: 1px solid #FED7AA; }}
    .result-card-industry  {{ border-left: 3px solid #A855F7; border: 1px solid #E9D5FF; }}
    .result-card-other     {{ border-left: 3px solid #94A3B8; border: 1px solid #E2E8F0; }}

    .result-card-label {{
        font-size: 0.7rem;
        color: {BRAND_MUTED};
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.3px;
    }}
    .result-card-value {{
        font-size: 0.95rem;
        font-weight: 700;
        color: {BRAND_TEXT};
        word-break: break-word;
    }}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Load & deduplicate data
# ---------------------------------------------------------------------------

@st.cache_data
def load_data():
    jsonl_path = Path(__file__).parent.parent / "extraction_results.jsonl"
    records = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    # Deduplicate: keep the LAST entry per source file (later run = better)
    seen: dict[str, int] = {}
    for i, r in enumerate(records):
        seen[r.get("_source_file", f"unknown_{i}")] = i
    deduped = [records[i] for i in sorted(seen.values())]
    return pd.DataFrame(deduped)


df = load_data()

# ---------------------------------------------------------------------------
# Derived columns
# ---------------------------------------------------------------------------

df["_has_error"] = df["_error"].notna() & (df["_error"] != "")
df["_revenue_b"] = pd.to_numeric(df.get("revenue"), errors="coerce") / 1e9
df["_net_income_b"] = pd.to_numeric(df.get("net_income"), errors="coerce") / 1e9
df["_total_assets_b"] = pd.to_numeric(df.get("total_assets"), errors="coerce") / 1e9
df["_state"] = df["company_state"].fillna(df.get("state", ""))

total_companies = len(df)
success_count = int((~df["_has_error"]).sum())
valid_count = int(df["_valid"].sum())
avg_time = df["_processing_time_s"].mean()
total_time_m = df["_processing_time_s"].sum() / 60
revenue_coverage = int(df["revenue"].notna().sum())
total_revenue = df["_revenue_b"].sum()

CORE_EXTRACTED_FIELDS = [
    "company_name", "company_address", "company_city", "company_state",
    "company_postal_code", "company_phone", "company_ein", "cik",
    "primary_sic_code_id", "company_sic_name", "revenue", "net_income",
    "total_assets", "shareholders_equity", "cash", "report_date",
    "fiscal_year_end_month", "stock_ticker_symbol", "stock_exchange_code",
]

field_coverage = {}
for col in CORE_EXTRACTED_FIELDS:
    if col in df.columns:
        filled = df[col].notna() & (df[col] != "") & (df[col] != "null")
        field_coverage[col] = int(filled.sum())


# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------

st.markdown(f"""
<div class="hero">
    <div class="hero-left">
        <div class="hero-brand">DATA AXLE</div>
        <div class="hero-title">SEC Pilot — Filing Extraction Pipeline</div>
        <div class="hero-sub">
            Autonomous multi-agent pipeline extracting structured business attributes
            from S&P 500 SEC 10-K filings at scale
        </div>
        <div class="hero-badge">⚡ {total_companies} Companies Processed</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------

k1, k2, k3, k4, k5 = st.columns(5)

with k1:
    st.markdown(f"""<div class="kpi-card">
        <div class="kpi-value">{total_companies}</div>
        <div class="kpi-label">SEC Filings Processed</div>
    </div>""", unsafe_allow_html=True)

with k2:
    st.markdown(f"""<div class="kpi-card">
        <div class="kpi-value kpi-value-green">{valid_count}</div>
        <div class="kpi-label">Successful Extractions</div>
    </div>""", unsafe_allow_html=True)

with k3:
    st.markdown(f"""<div class="kpi-card">
        <div class="kpi-value">{revenue_coverage}</div>
        <div class="kpi-label">Revenue Extracted</div>
    </div>""", unsafe_allow_html=True)

with k4:
    st.markdown(f"""<div class="kpi-card">
        <div class="kpi-value kpi-value-blue">{avg_time:.1f}s</div>
        <div class="kpi-label">Avg Processing Time</div>
    </div>""", unsafe_allow_html=True)

with k5:
    st.markdown(f"""<div class="kpi-card">
        <div class="kpi-value kpi-value-green">${total_revenue:,.0f}B</div>
        <div class="kpi-label">Total Revenue Captured</div>
    </div>""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------

st.markdown('<div class="section-header"><span>⬡</span> Extraction Architecture</div>', unsafe_allow_html=True)

st.markdown(f"""
<div class="arch-container">
    <div class="arch-flow">
        <div class="arch-node node-ingest">
            <div class="arch-node-icon">📥</div>
            <div class="arch-node-title">Ingest</div>
            <div class="arch-node-desc">Read raw .txt from<br/>Databricks Volume</div>
        </div>
        <div class="arch-arrow">→</div>
        <div class="arch-node node-scrape">
            <div class="arch-node-icon">🔍</div>
            <div class="arch-node-title">Scraper</div>
            <div class="arch-node-desc">SEC Header +<br/>XBRL parsing</div>
        </div>
        <div class="arch-arrow">→</div>
        <div class="arch-node node-filter">
            <div class="arch-node-icon">📄</div>
            <div class="arch-node-title">Section Filter</div>
            <div class="arch-node-desc">Extract 10-K,<br/>drop low-value §</div>
        </div>
        <div class="arch-arrow">→</div>
        <div class="arch-node node-llm">
            <div class="arch-node-icon">🧠</div>
            <div class="arch-node-title">LLM Extract</div>
            <div class="arch-node-desc">Structured output<br/>via Groq / GPT</div>
        </div>
        <div class="arch-arrow">→</div>
        <div class="arch-node node-enrich">
            <div class="arch-node-icon">🏭</div>
            <div class="arch-node-title">Enrichment</div>
            <div class="arch-node-desc">NAICS lookup,<br/>manufacturer flag</div>
        </div>
        <div class="arch-arrow">→</div>
        <div class="arch-node node-eval">
            <div class="arch-node-icon">✅</div>
            <div class="arch-node-title">Evaluate</div>
            <div class="arch-node-desc">LLM judge +<br/>fill rate check</div>
        </div>
        <div class="arch-arrow">→</div>
        <div class="arch-node node-web">
            <div class="arch-node-icon">🌐</div>
            <div class="arch-node-title">Web Fallback</div>
            <div class="arch-node-desc">DuckDuckGo search<br/>if fill rate low</div>
        </div>
        <div class="arch-arrow">→</div>
        <div class="arch-node node-delta">
            <div class="arch-node-icon">💾</div>
            <div class="arch-node-title">Delta Table</div>
            <div class="arch-node-desc">Write to Unity<br/>Catalog table</div>
        </div>
    </div>
    <div class="stats-bar">
        <div class="stat-chip">🏗️ Built with <b>LangGraph</b></div>
        <div class="stat-chip">🔗 <b>7 Agent Nodes</b> in pipeline</div>
        <div class="stat-chip">📊 <b>{TARGET_ATTR_COUNT}</b> target attributes</div>
        <div class="stat-chip">⚡ Conditional web fallback when fill rate &lt; 50%</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Live Extraction
# ---------------------------------------------------------------------------

FIELD_CATEGORIES = {
    "identity": {
        "fields": {"company_name", "company_legal_name", "company_ein", "cik",
                    "name", "stock_ticker_symbol", "stock_exchange_code",
                    "company_year_founded", "company_description", "in_business",
                    "company_active_indicator"},
        "label": "Identity & Status",
        "css": "identity",
    },
    "location": {
        "fields": {"street", "city", "state", "postal_code", "phone", "website",
                    "company_address", "company_city", "company_state",
                    "company_postal_code", "company_phone"},
        "label": "Location & Contact",
        "css": "location",
    },
    "financial": {
        "fields": {"revenue", "net_income", "gross_profit", "cost_of_revenue",
                    "total_assets", "total_liabilities_and_equity",
                    "operating_expenses", "operating_income", "cash",
                    "current_assets", "shareholders_equity",
                    "long_term_debt", "short_term_debt", "total_debt",
                    "report_date", "fiscal_year_end_month", "tax_and_interest"},
        "label": "Financial Data",
        "css": "financial",
    },
    "industry": {
        "fields": {"primary_sic_code_id", "primary_naics_code_id",
                    "company_sic_code", "company_sic_name",
                    "company_naics_code", "company_naics_name",
                    "location_employee_count", "corporate_employee_count",
                    "place_type"},
        "label": "Industry & Classification",
        "css": "industry",
    },
}


def _categorize(key: str) -> str:
    for cat_id, cat in FIELD_CATEGORIES.items():
        if key in cat["fields"]:
            return cat_id
    return "other"


def render_live_result(record: dict, elapsed: float):
    """Render extracted attributes grouped by category with color coding."""
    company = record.get("company_name") or record.get("name") or "Unknown Company"
    st.markdown(f"##### {company}")
    st.caption(f"Extracted in {elapsed:.1f}s using the full agentic pipeline")

    filled = {k: v for k, v in record.items()
              if v is not None and v != "" and v != [] and not str(k).startswith("_")}
    empty = {k: v for k, v in record.items()
             if (v is None or v == "" or v == []) and not str(k).startswith("_")}

    m1, m2 = st.columns(2)
    m1.metric("Attributes Extracted", len(filled))
    m2.metric("Missing", len(empty))

    # Group filled attrs by category
    grouped: dict[str, list[tuple[str, object]]] = {}
    for key, val in filled.items():
        cat = _categorize(key)
        grouped.setdefault(cat, []).append((key, val))

    cat_order = ["identity", "location", "financial", "industry", "other"]
    for cat_id in cat_order:
        items = grouped.get(cat_id, [])
        if not items:
            continue
        cat_info = FIELD_CATEGORIES.get(cat_id, {"label": "Other", "css": "other"})
        st.markdown(
            f'<div class="res-group-title res-group-{cat_info["css"]}">'
            f'{cat_info["label"]} ({len(items)})</div>',
            unsafe_allow_html=True,
        )
        for row_start in range(0, len(items), 4):
            cols = st.columns(4)
            for col_idx, (key, val) in enumerate(items[row_start:row_start + 4]):
                with cols[col_idx]:
                    icon, label = ATTRIBUTE_LABELS.get(key, ("", key.replace("_", " ").title()))
                    display = str(val)
                    if len(display) > 80:
                        display = display[:80] + "..."
                    st.markdown(
                        f'<div class="result-card result-card-{cat_info["css"]}">'
                        f'<div class="result-card-label">{icon} {label}</div>'
                        f'<div class="result-card-value">{display}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

    with st.expander(f"Show {len(empty)} missing fields"):
        missing_names = [ATTRIBUTE_LABELS.get(k, ("", k.replace("_", " ").title()))[1] for k in empty]
        st.caption(", ".join(missing_names) if missing_names else "None")

    dl1, dl2, _ = st.columns([1, 1, 4])
    with dl1:
        st.download_button(
            "📥 Download JSON",
            data=json.dumps(record, indent=2, default=str),
            file_name="extraction_result.json",
            mime="application/json",
        )
    with dl2:
        rows = [{"attribute": k, "value": v} for k, v in filled.items()]
        st.download_button(
            "📥 Download CSV",
            data=pd.DataFrame(rows).to_csv(index=False),
            file_name="extraction_result.csv",
            mime="text/csv",
        )


if "live_result" not in st.session_state:
    st.session_state.live_result = None
if "live_eval" not in st.session_state:
    st.session_state.live_eval = None
if "live_elapsed" not in st.session_state:
    st.session_state.live_elapsed = 0.0
if "uploaded_file_content" not in st.session_state:
    st.session_state.uploaded_file_content = ""
if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = ""

st.markdown(
    '<div class="section-header"><span>⬡</span> Live Extraction</div>',
    unsafe_allow_html=True,
)

live_tab_upload, live_tab_paste = st.tabs(["📁 Upload File", "📝 Paste Text"])

with live_tab_upload:
    uploaded = st.file_uploader(
        "Upload an SEC filing (.txt or .html)",
        type=["txt", "html", "htm"],
        help="Drop a raw SEC full-submission file here",
        key="live_upload",
    )
    if uploaded:
        cached_name = st.session_state.get("uploaded_file_name", "")
        if not st.session_state.uploaded_file_content or (uploaded.name != cached_name):
            st.session_state.uploaded_file_content = uploaded.read().decode("utf-8", errors="ignore")
            st.session_state.uploaded_file_name = uploaded.name or ""
        upload_text = st.session_state.uploaded_file_content
        st.text_area("Preview (first 2,000 chars)", upload_text[:2000], height=150, disabled=True, key="upload_preview")
    else:
        st.session_state.uploaded_file_content = ""
        st.session_state.uploaded_file_name = ""
        upload_text = ""

with live_tab_paste:
    paste_text = st.text_area(
        "Paste SEC filing text",
        height=200,
        placeholder="Paste the full .txt content of an SEC filing here...",
        key="live_paste",
    )

input_text = st.session_state.uploaded_file_content or paste_text

btn_col, _ = st.columns([1, 5])
with btn_col:
    do_extract = st.button("🚀 Trigger Agent", type="primary", use_container_width=True, key="trigger_agent_btn")

if do_extract:
    if not input_text.strip():
        st.error("Please upload a file or paste text first.")
    else:
        try:
            from dotenv import load_dotenv
            load_dotenv(dotenv_path=".env", override=True)
        except ImportError:
            pass

        progress_placeholder = st.empty()
        progress_placeholder.progress(0, text="Starting extraction pipeline...")
        t0 = time.time()
        try:
            from agent_server.sec_extraction.config import get_config
            from agent_server.sec_extraction.workflow import extraction_workflow

            progress_placeholder.progress(10, text="Step 1/6 — Scraper: parsing SEC header + XBRL tags...")
            result = extraction_workflow(input_text, get_config())
            elapsed = time.time() - t0
            progress_placeholder.progress(100, text=f"Extraction complete — {elapsed:.1f}s")
            time.sleep(0.5)
            progress_placeholder.empty()

            st.session_state.live_result = result.get("record", {})
            st.session_state.live_eval = result.get("evaluation", {})
            st.session_state.live_elapsed = elapsed
            st.rerun()
        except Exception as exc:
            progress_placeholder.empty()
            st.error(f"Pipeline error: {exc}")

if st.session_state.live_result:
    st.markdown("---")
    st.success(
        f"**Extraction Complete** — "
        f"{st.session_state.live_result.get('company_name') or st.session_state.live_result.get('name', 'Unknown')} "
        f"— {st.session_state.live_elapsed:.1f}s"
    )
    render_live_result(
        st.session_state.live_result,
        elapsed=st.session_state.live_elapsed,
    )


# ---------------------------------------------------------------------------
# Insights tabs
# ---------------------------------------------------------------------------

st.markdown('<div class="section-header"><span>⬡</span> Extraction Insights</div>', unsafe_allow_html=True)

tab_field, tab_industry, tab_geo, tab_finance, tab_data = st.tabs([
    "📋 Field Coverage",
    "🏢 Industry Breakdown",
    "🗺️ Geographic Distribution",
    "💰 Financial Analytics",
    "📊 Full Data Table",
])

with tab_field:
    st.markdown("##### Attribute Extraction Coverage Across All Filings")
    st.caption("Percentage of companies where each attribute was successfully extracted")

    cov_df = pd.DataFrame([
        {
            "Attribute": k.replace("_", " ").title(),
            "Extracted": v,
            "Coverage (%)": round(v / total_companies * 100, 1),
        }
        for k, v in sorted(field_coverage.items(), key=lambda x: -x[1])
    ])

    c1, c2 = st.columns([2, 1])
    with c1:
        chart_df = cov_df.set_index("Attribute")[["Coverage (%)"]].sort_values("Coverage (%)", ascending=True)
        st.bar_chart(chart_df, horizontal=True, height=520, color=BRAND_ACCENT)
    with c2:
        full_cov = sum(1 for v in field_coverage.values() if v == total_companies)
        high_cov = sum(1 for v in field_coverage.values() if v / total_companies >= 0.9)
        st.metric("100% Coverage Fields", f"{full_cov} / {len(field_coverage)}")
        st.metric(">90% Coverage Fields", f"{high_cov} / {len(field_coverage)}")
        st.metric("Revenue Extraction", f"{revenue_coverage}/{total_companies} ({revenue_coverage/total_companies:.0%})")
        st.markdown("---")
        st.markdown("**Key fields at 100%:**")
        for k, v in field_coverage.items():
            if v == total_companies:
                st.markdown(f"- {k.replace('_', ' ').title()}")


with tab_industry:
    st.markdown("##### Industry Distribution (by SIC Classification)")

    sic_counts = df["company_sic_name"].value_counts().head(15).reset_index()
    sic_counts.columns = ["Industry (SIC)", "Count"]

    c1, c2 = st.columns([3, 1])
    with c1:
        st.bar_chart(sic_counts.set_index("Industry (SIC)"), horizontal=True, height=480, color=BRAND_ACCENT2)
    with c2:
        st.metric("Unique Industries", df["company_sic_name"].nunique())
        st.metric("Top Industry", sic_counts.iloc[0]["Industry (SIC)"])
        st.metric("Top Industry Count", int(sic_counts.iloc[0]["Count"]))

    st.markdown("---")
    st.markdown("##### Stock Exchange Distribution")
    ex_df = df["stock_exchange_code"].dropna()
    ex_df = ex_df[ex_df != ""].value_counts().reset_index()
    ex_df.columns = ["Exchange", "Companies"]
    ec1, ec2, ec3 = st.columns(3)
    for i, row in ex_df.iterrows():
        col = [ec1, ec2, ec3][i % 3]
        with col:
            st.metric(row["Exchange"], int(row["Companies"]))


with tab_geo:
    st.markdown("##### Companies by State")

    state_counts = df["_state"].value_counts().head(20).reset_index()
    state_counts.columns = ["State", "Count"]

    c1, c2 = st.columns([3, 1])
    with c1:
        st.bar_chart(state_counts.set_index("State"), horizontal=True, height=520, color="#5B8DEF")
    with c2:
        st.metric("States Represented", df["_state"].nunique())
        st.metric("Top State", state_counts.iloc[0]["State"])
        st.metric("Top State Count", int(state_counts.iloc[0]["Count"]))
        top3 = state_counts.head(3)
        top3_pct = top3["Count"].sum() / total_companies * 100
        st.metric("Top 3 States %", f"{top3_pct:.0f}%")

    st.markdown("---")
    st.markdown("##### Top Cities by Company Count")
    city_counts = df["company_city"].value_counts().head(10).reset_index()
    city_counts.columns = ["City", "Count"]
    st.bar_chart(city_counts.set_index("City"), height=300, color=BRAND_ACCENT)


with tab_finance:
    st.markdown("##### Revenue Distribution (Top 25 Companies)")

    rev_df = df[df["_revenue_b"].notna()].nlargest(25, "_revenue_b")[
        ["company_name", "_revenue_b", "_net_income_b", "_total_assets_b", "company_state"]
    ].copy()
    rev_df.columns = ["Company", "Revenue ($B)", "Net Income ($B)", "Total Assets ($B)", "State"]

    st.bar_chart(
        rev_df.set_index("Company")[["Revenue ($B)"]].sort_values("Revenue ($B)", ascending=True),
        horizontal=True, height=600, color=BRAND_ACCENT,
    )

    st.markdown("---")
    fc1, fc2, fc3, fc4 = st.columns(4)

    rev_all = df["_revenue_b"].dropna()
    ni_all = df["_net_income_b"].dropna()
    ta_all = df["_total_assets_b"].dropna()
    cash_b = pd.to_numeric(df.get("cash"), errors="coerce").dropna() / 1e9

    with fc1:
        st.metric("Total Revenue", f"${rev_all.sum():,.0f}B")
        st.metric("Median Revenue", f"${rev_all.median():,.1f}B")
    with fc2:
        st.metric("Total Net Income", f"${ni_all.sum():,.0f}B")
        st.metric("Median Net Income", f"${ni_all.median():,.1f}B")
    with fc3:
        st.metric("Total Assets", f"${ta_all.sum():,.0f}B")
        st.metric("Median Assets", f"${ta_all.median():,.1f}B")
    with fc4:
        st.metric("Total Cash", f"${cash_b.sum():,.0f}B")
        st.metric("Median Cash", f"${cash_b.median():,.1f}B")

    st.markdown("---")
    st.markdown("##### Revenue vs Total Assets")
    scatter_df = df[["company_name", "_revenue_b", "_total_assets_b"]].dropna().copy()
    scatter_df.columns = ["Company", "Revenue ($B)", "Total Assets ($B)"]
    st.scatter_chart(scatter_df, x="Revenue ($B)", y="Total Assets ($B)", height=400, color=BRAND_ACCENT2)


with tab_data:
    st.markdown("##### Extracted Company Records")

    display_cols = [
        "_source_file", "company_name", "company_state", "company_sic_name",
        "revenue", "net_income", "total_assets", "cash",
        "location_employee_count", "stock_exchange_code", "stock_ticker_symbol",
        "_processing_time_s",
    ]
    available_cols = [c for c in display_cols if c in df.columns]

    search = st.text_input("Search companies...", placeholder="e.g. Apple, CA, NYSE")

    col_renames = {
        "_source_file": "File", "company_name": "Company", "company_state": "State",
        "company_sic_name": "Industry", "revenue": "Revenue", "net_income": "Net Income",
        "total_assets": "Total Assets", "cash": "Cash",
        "location_employee_count": "Employees", "stock_exchange_code": "Exchange",
        "stock_ticker_symbol": "Ticker", "_processing_time_s": "Time (s)",
    }
    table_df = df[available_cols].copy()
    table_df.columns = [col_renames.get(c, c) for c in available_cols]

    if search:
        mask = table_df.apply(lambda row: search.lower() in str(row.values).lower(), axis=1)
        table_df = table_df[mask]

    st.dataframe(table_df, use_container_width=True, hide_index=True, height=600)
    st.caption(f"Showing {len(table_df)} of {total_companies} records")

    st.download_button(
        "📥 Download CSV",
        data=table_df.to_csv(index=False),
        file_name="sec_extraction_results.csv",
        mime="text/csv",
    )


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: {BRAND_MUTED}; font-size: 0.8rem; padding: 1rem 0;">
    <strong style="color: {BRAND_TEXT};">Data Axle</strong> · SEC Pilot · Hackathon 2026<br/>
    Built with LangGraph, Databricks, Groq, and Vector Search
</div>
""", unsafe_allow_html=True)

"""Step 2 — Scraper Agent

Combines SEC header parsing and XBRL tag extraction.
Same interface as before — takes clean text string, returns partial dict.
Downstream steps fill whatever is missing.

Usage:
    from agent_server.sec_extraction.tools.scraper import scrape_attributes
    partial = scrape_attributes(raw_text)
"""

from __future__ import annotations
import re
from datetime import datetime

# ---------------------------------------------------------------------------
# SEC Header extraction — pulls from <SEC-HEADER> block directly
# Deterministic, no LLM, guaranteed format across all filers
# ---------------------------------------------------------------------------

def _parse_sec_header(content: str) -> dict:
    header_match = re.search(
        r'<SEC-HEADER>(.*?)</SEC-HEADER>', content, re.DOTALL
    )
    if not header_match:
        return {}
    header = header_match.group(1)

    def extract(pattern):
        m = re.search(pattern, header, re.IGNORECASE)
        return m.group(1).strip() if m else None

    return {
        "accession_number" : extract(r'ACCESSION NUMBER:\s*(.+)'),
        "filing_date"      : extract(r'FILED AS OF DATE:\s*(.+)'),
        "sic_code"         : extract(r'STANDARD INDUSTRIAL CLASSIFICATION:.*?\[(\d+)\]'),
        "sic_description"  : extract(r'STANDARD INDUSTRIAL CLASSIFICATION:\s*(.+?)\['),
        "street"           : extract(r'STREET 1:\s*(.+)'),
        "city"             : extract(r'CITY:\s*(.+)'),
        "state"            : extract(r'STATE:\s*(.+)'),
        "postal_code"      : extract(r'ZIP:\s*(.+)'),
        "phone"            : extract(r'BUSINESS PHONE:\s*(.+)'),
        "company_name"     : extract(r'COMPANY CONFORMED NAME:\s*(.+)'),
        "cik"              : extract(r'CENTRAL INDEX KEY:\s*(.+)'),
        "company_ein"      : extract(r'EIN:\s*(.+)'),
        "fiscal_year_end"  : extract(r'FISCAL YEAR END:\s*(.+)'),
        "state_of_incorporation": extract(r'STATE OF INCORPORATION:\s*(.+)'),
    }


# ---------------------------------------------------------------------------
# XBRL extraction — pulls standardised us-gaap: and dei: tags
# Same concept names work across ALL companies regardless of display labels
# ---------------------------------------------------------------------------

CONCEPTS: dict[str, list[str]] = {
    # Identity
    "company_name"           : ["dei:EntityRegistrantName"],
    "cik"                    : ["dei:EntityCentralIndexKey"],
    "company_ein"            : ["dei:EntityTaxIdentificationNumber"],
    "trading_symbol"         : ["dei:TradingSymbol"],
    "stock_exchange"         : ["dei:SecurityExchangeName"],
    "fiscal_year_end"        : ["dei:CurrentFiscalYearEndDate"],
    "fiscal_year_focus"      : ["dei:DocumentFiscalYearFocus"],
    "period_of_report"       : ["dei:DocumentPeriodEndDate"],
    "filing_type"            : ["dei:DocumentType"],
    "state_of_incorporation" : ["dei:EntityIncorporationStateCountryCode"],
    "city"                   : ["dei:EntityAddressCityOrTown"],
    "state"                  : ["dei:EntityAddressStateOrProvince"],
    "postal_code"            : ["dei:EntityAddressPostalZipCode"],
    "phone"                  : ["dei:LocalPhoneNumber"],
    "filer_category"         : ["dei:EntityFilerCategory"],
    "public_float"           : ["dei:EntityPublicFloat"],
    "shares_outstanding"     : ["dei:EntityCommonStockSharesOutstanding"],
    "is_well_known_issuer"   : ["dei:EntityWellKnownSeasonedIssuer"],
    "is_emerging_growth"     : ["dei:EntityEmergingGrowthCompany"],
    "is_shell_company"       : ["dei:EntityShellCompany"],
    "amendment_flag"         : ["dei:AmendmentFlag"],
    # Auditor
    "auditor_name"           : ["dei:AuditorName"],
    "auditor_location"       : ["dei:AuditorLocation"],
    "auditor_firm_id"        : ["dei:AuditorFirmId"],
    # Financials — income statement
    "net_income"             : ["us-gaap:NetIncomeLoss"],
    "income_before_tax"      : ["us-gaap:IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest"],
    "income_tax_expense"     : ["us-gaap:IncomeTaxExpenseBenefit"],
    "effective_tax_rate"     : ["us-gaap:EffectiveIncomeTaxRateContinuingOperations"],
    "basic_eps"              : ["us-gaap:EarningsPerShareBasic"],
    "diluted_eps"            : ["us-gaap:EarningsPerShareDiluted"],
    "weighted_avg_shares_basic": ["us-gaap:WeightedAverageNumberOfSharesOutstandingBasic"],
    # Financials — balance sheet
    "total_assets"           : ["us-gaap:Assets"],
    "total_liabilities_equity": ["us-gaap:LiabilitiesAndStockholdersEquity"],
    "retained_earnings"      : ["us-gaap:RetainedEarningsAccumulatedDeficit"],
    "ppe_net"                : ["us-gaap:PropertyPlantAndEquipmentNet"],
    "shares_authorized"      : ["us-gaap:CommonStockSharesAuthorized"],
    # Financials — cash flow
    "cfo"                    : ["us-gaap:NetCashProvidedByUsedInOperatingActivities"],
    "cfi"                    : ["us-gaap:NetCashProvidedByUsedInInvestingActivities"],
    "cff"                    : ["us-gaap:NetCashProvidedByUsedInFinancingActivities"],
    "cash_end_of_period"     : ["us-gaap:CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"],
    "share_buybacks"         : ["us-gaap:PaymentsForRepurchaseOfCommonStock"],
    # Employee count
    "location_employee_count": ["us-gaap:NumberOfEmployees",
                                 "us-gaap:EntityNumberOfEmployees"],
    # Revenue — multiple possible tags across sectors
    "revenue"                : [
        "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
        "us-gaap:Revenues",
        "us-gaap:SalesRevenueNet",
        "us-gaap:RevenueFromContractWithCustomerIncludingAssessedTax",
    ],
    # Website
    "website"                : ["dei:EntityWebSiteAddress"],
}


def _parse_xbrl(content: str) -> dict:
    result: dict = {}

    try:
        doc_pattern = (
            r'<DOCUMENT>\s*<TYPE>[^\n]+\s*<SEQUENCE>[^\n]+'
            r'\s*<FILENAME>([^\n]+).*?<TEXT>(.*?)</TEXT>\s*</DOCUMENT>'
        )
        docs = re.findall(doc_pattern, content, re.DOTALL)

        xml_text = None
        for fname, text in docs:
            if '_htm.xml' in fname.strip():
                xml_text = text
                break
        if not xml_text:
            return result

        # build context map — skip dimensional/segment contexts
        context_map: dict = {}
        for ctx in re.finditer(
            r'<context id="([^"]+)">(.*?)</context>', xml_text, re.DOTALL
        ):
            ctx_id, ctx_body = ctx.group(1), ctx.group(2)
            if '<segment>' in ctx_body:
                continue
            start   = re.search(r'<startDate>([^<]+)</startDate>', ctx_body)
            end     = re.search(r'<endDate>([^<]+)</endDate>',     ctx_body)
            instant = re.search(r'<instant>([^<]+)</instant>',     ctx_body)
            if start and end:
                context_map[ctx_id] = ('duration', start.group(1), end.group(1))
            elif instant:
                context_map[ctx_id] = ('instant', instant.group(1), instant.group(1))

        def get_value(concept_list: list[str]):
            best_val, best_days = None, -1
            for concept in concept_list:
                pat = (
                    rf'<{re.escape(concept)}[^>]*contextRef="([^"]*)"'
                    rf'[^>]*>([^<]+)</{re.escape(concept)}>'
                )
                for m in re.finditer(pat, xml_text):
                    ctx_id, raw = m.group(1), m.group(2).strip()
                    if ctx_id not in context_map:
                        continue
                    ctx_type, d1, d2 = context_map[ctx_id]
                    try:
                        val = float(raw.replace(',', ''))
                    except Exception:
                        val = raw
                    if ctx_type == 'duration':
                        try:
                            days = (
                                datetime.strptime(d2, '%Y-%m-%d') -
                                datetime.strptime(d1, '%Y-%m-%d')
                            ).days
                        except Exception:
                            days = 0
                        if days > best_days:
                            best_days, best_val = days, val
                    elif ctx_type == 'instant' and best_val is None:
                        best_val = val
            return best_val

        for feature, concepts in CONCEPTS.items():
            try:
                val = get_value(concepts)
                if val is not None:
                    result[feature] = val
            except Exception:
                pass

    except Exception:
        pass

    return result


# ---------------------------------------------------------------------------
# Public API — same signature as before
# Input : raw SEC full-submission text (the .txt file content)
# Output: partial dict — only fields with values, same as original scraper
# ---------------------------------------------------------------------------

def scrape_attributes(text: str) -> dict:
    """Run SEC header + XBRL extraction against *text*.

    Returns only the fields that had at least one match.
    Downstream steps fill in whatever is missing.

    Priority: XBRL values take precedence over SEC header values
    when both are present — XBRL is more precisely typed.
    """
    result: dict = {}

    # layer 1 — SEC header (fast, always present)
    header_data = _parse_sec_header(text)
    result.update({k: v for k, v in header_data.items() if v is not None})

    # layer 2 — XBRL tags (more precise, overwrites header where both exist)
    xbrl_data = _parse_xbrl(text)
    result.update({k: v for k, v in xbrl_data.items() if v is not None})

    return result
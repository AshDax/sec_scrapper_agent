"""
SEC 10-K Section Filter

Takes a raw SEC full-submission text string, extracts the 10-K document,
strips HTML, splits into Item sections, filters out low-value sections,
and returns a single concatenated string of the important sections only.

Usage:
    from agent_server.sec_extraction.tools.section_filtering import filter_sections
    result = filter_sections(raw_text)
"""

from __future__ import annotations

import re
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SKIP_PATTERNS = [
    r'Item\s+8',    # Financial Statements — covered by XBRL
    r'Item\s+9\.',  # Accountant changes — low value
    r'Item\s+4',    # Mine Safety — always "not applicable"
    r'Item\s+15',   # Exhibits list — no features
    r'Item\s+16',   # 10-K summary — redundant
    r'HEADER',      # pre-Item header block
]

MIN_CHARS = 500    # sections below this are placeholders — skip
MAX_CHARS = 8000   # truncate sections beyond this for cost control


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_10k_html(content: str) -> str | None:
    """Pull the 10-K HTML block out of the SEC container."""
    match = re.search(
        r'<TYPE>10-K.*?<TEXT>(.*?)</TEXT>',
        content,
        re.DOTALL
    )
    if not match:
        return None

    tenk_html = match.group(1).strip()

    # join split spans before stripping — handles UPPERCASE split headings
    # e.g. <span>ITEM 1.</span><span> BUSINESS</span>
    tenk_html = re.sub(r'</span>\s*<span[^>]*>', ' ', tenk_html)

    return tenk_html


def _html_to_plain_text(html: str) -> str:
    """Strip HTML tags and collapse whitespace."""
    soup = BeautifulSoup(html, 'html.parser')
    text = soup.get_text(separator=' ', strip=True)
    return re.sub(r'\s+', ' ', text)


def _split_into_sections(text: str) -> list[dict]:
    """
    Split plain text into Item sections.
    Handles both Title Case (Item 1.) and UPPERCASE (ITEM 1.) headings.
    Returns list of {section_title, section_text, char_count}.
    """
    pattern = r'((?:Item|ITEM)\s+\d+[A-C]?[\.\s]{1,6}[A-Z][^\n]{3,80})'
    parts   = re.split(pattern, text)

    sections      = []
    current_title = "HEADER"
    current_text  = ""

    for part in parts:
        if re.match(r'(?:Item|ITEM)\s+\d+[A-C]?[\.\s]', part):
            if current_text.strip():
                sections.append({
                    "section_title" : current_title.strip(),
                    "section_text"  : current_text.strip(),
                    "char_count"    : len(current_text.strip()),
                })
            current_title = part.strip()
            current_text  = ""
        else:
            current_text += part

    # save last section
    if current_text.strip():
        sections.append({
            "section_title" : current_title.strip(),
            "section_text"  : current_text.strip(),
            "char_count"    : len(current_text.strip()),
        })

    return sections


def _should_skip(section_title: str, char_count: int) -> tuple[bool, str]:
    """Return (should_skip, reason) for a given section."""
    if char_count < MIN_CHARS:
        return True, f"too short ({char_count} chars)"
    for pattern in SKIP_PATTERNS:
        if re.search(pattern, section_title, re.IGNORECASE):
            return True, f"matches skip pattern: {pattern}"
    return False, ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def picking_sections(text: str) -> str:
    """
    Core logic — takes raw SEC submission text, returns filtered + 
    concatenated section text string.

    Steps:
      1. Extract 10-K HTML from SEC container
      2. Strip HTML tags → plain text
      3. Split on Item boundaries
      4. Filter out low-value sections
      5. Truncate large sections to MAX_CHARS
      6. Concatenate and return

    Also prints before/after character counts.
    """
    original_len = len(text)

    # step 1 — extract 10-K HTML
    tenk_html = _extract_10k_html(text)
    if not tenk_html:
        print("WARNING: No 10-K document found in submission")
        print(f"  Input chars  : {original_len:,}")
        print(f"  Output chars : 0")
        return ""

    # step 2 — plain text
    plain_text     = _html_to_plain_text(tenk_html)
    plain_text_len = len(plain_text)

    # step 3 — split into sections
    sections = _split_into_sections(plain_text)

    # step 4 + 5 — filter and truncate
    kept_sections    = []
    skipped_sections = []

    for s in sections:
        skip, reason = _should_skip(s["section_title"], s["char_count"])
        if skip:
            skipped_sections.append((s["section_title"], reason))
            continue

        # truncate if needed
        section_text = (
            s["section_text"][:MAX_CHARS]
            if s["char_count"] > MAX_CHARS
            else s["section_text"]
        )
        truncated = s["char_count"] > MAX_CHARS
        kept_sections.append({
            **s,
            "section_text" : section_text,
            "truncated"    : truncated,
        })

    # step 6 — concatenate
    output_parts = []
    for s in kept_sections:
        output_parts.append(
            f"=== {s['section_title']} ===\n{s['section_text']}"
        )

    result     = "\n\n".join(output_parts)
    result_len = len(result)

    # print before / after report
    print(f"\n{'='*55}")
    print(f"  SEC 10-K Section Filter — Summary")
    print(f"{'='*55}")
    print(f"  Input (raw submission)   : {original_len:>12,} chars")
    print(f"  After HTML strip         : {plain_text_len:>12,} chars")
    print(f"  Output (filtered text)   : {result_len:>12,} chars")
    print(f"  Reduction                : {((original_len - result_len) / original_len * 100):.1f}%")
    print(f"{'='*55}")
    print(f"  Total sections found     : {len(sections):>5}")
    print(f"  Sections kept            : {len(kept_sections):>5}")
    print(f"  Sections skipped         : {len(skipped_sections):>5}")
    print(f"{'='*55}")
    print(f"  Kept sections:")
    for s in kept_sections:
        trunc = " [truncated]" if s["truncated"] else ""
        print(f"    ✓ [{s['char_count']:>7,} chars]{trunc}  {s['section_title'][:55]}")
    print(f"  Skipped sections:")
    for title, reason in skipped_sections:
        print(f"    ✗ {title[:45]:<45} → {reason}")
    print(f"{'='*55}\n")

    return result


def filter_sections(text: str) -> str:
    """
    Public entry point — same as picking_sections().
    Named to match the tool registration in the agent.

    Args:
        text: Raw SEC full-submission .txt file content

    Returns:
        Single string containing all important 10-K sections
        concatenated, ready for LLM feature extraction.
    """
    return picking_sections(text)
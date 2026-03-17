"""Step 1 — Text Extraction

Parse raw SEC filing text or HTML into clean, searchable text.
Strips HTML tags, decodes entities, and preserves section headers
(ITEM 1, ITEM 1A, etc.) for downstream extraction.

Usage:
    from agent_server.sec_extraction.tools.text_extraction import extract_text
    clean = extract_text(raw_html_or_text)
"""

from __future__ import annotations

import html
import re


def extract_text(raw: str) -> str:
    """Convert raw SEC filing (text or HTML) into clean plaintext."""
    if not raw or not raw.strip():
        return ""

    text = raw

    # Remove <style> and <script> blocks entirely
    text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<script[^>]*>.*?</script>", " ", text, flags=re.DOTALL | re.IGNORECASE)

    # Replace <br>, <p>, <div>, <tr>, <li> with newlines for structure
    text = re.sub(r"<(?:br|p|div|tr|li)[^>]*>", "\n", text, flags=re.IGNORECASE)

    # Strip remaining HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Decode HTML entities
    text = html.unescape(text)

    # Preserve SEC section headers on their own lines
    text = re.sub(
        r"(ITEM\s+\d+[A-Z]?\.?\s)",
        r"\n\n\1",
        text,
        flags=re.IGNORECASE,
    )

    # Collapse runs of whitespace (but keep newlines)
    text = re.sub(r"[^\S\n]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()

"""Step 1 — Text Extraction

Parse raw SEC filing text or HTML into clean, searchable text.
Strips HTML tags, decodes entities, and preserves section headers
(ITEM 1, ITEM 1A, etc.) for downstream extraction.

Handles large iXBRL/HTML files by truncating input before processing
so the start of the document (company name, etc.) is always used.
"""

from __future__ import annotations

import html
import re

# Cap input size so we don't OOM or hang on huge files; keep start of doc
MAX_RAW_CHARS = 1_500_000_000


def extract_text(raw: str) -> str:
    """Convert raw SEC filing (text or HTML) into clean plaintext.

    For very large inputs, only the first MAX_RAW_CHARS are processed so
    the beginning of the document (company name, header) is used.
    """
    if not raw or not raw.strip():
        return ""

    if len(raw) > MAX_RAW_CHARS:
        raw = raw[:MAX_RAW_CHARS]

    text = raw

    # Remove <style> and <script> blocks entirely
    text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<script[^>]*>.*?</script>", " ", text, flags=re.DOTALL | re.IGNORECASE)

    # Replace <br>, <p>, <div>, <tr>, <li> with newlines for structure
    text = re.sub(r"<(?:br|p|div|tr|li)[^>]*>", "\n", text, flags=re.IGNORECASE)

    # Strip remaining HTML/XML tags (including namespaced e.g. ix:nonNumeric)
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
    
    out = text.strip()

    # If we got nothing (e.g. entire doc was inside removed blocks), fallback: strip tags from start of raw
    if not out or len(out) < 100:
        fallback = re.sub(r"<[^>]+>", " ", raw[:100_000])
        fallback = html.unescape(fallback)
        fallback = re.sub(r"[^\S\n]+", " ", fallback).strip()
        if fallback:
            out = fallback

    return out

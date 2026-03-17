#!/usr/bin/env python3
"""Generate agent_server/sec_extraction/data_dictionary.py from CSVs.

Run: python scripts/embed_data_dictionary.py
"""

import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CSV_FILES = [
    PROJECT_ROOT / "financial_datum_attributes.csv",
    PROJECT_ROOT / "place_attributes (1).csv",
]
OUT_PATH = PROJECT_ROOT / "agent_server" / "sec_extraction" / "data_dictionary.py"

EXCLUDE_LIMITATIONS = {"Internal", "Secret", "Deprecated", "Restricted"}

TYPE_MAP = {
    "string": "string or null",
    "integer": "integer or null",
    "float": "float or null",
    "boolean": "true/false/null",
    "date": "string or null (ISO date)",
    "time": "string or null",
    "array": '["string", ...] or []',
    "date_range": "string or null",
    "geo_point": "string or null",
    "integer_range": "string or null",
}


def load_csv(path: Path) -> list[dict]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            attr = r.get("Attribute", "").strip()
            if not attr:
                continue
            lim = (r.get("Limitations") or "").strip()
            if any(x in lim for x in EXCLUDE_LIMITATIONS):
                continue
            raw_type = (r.get("Type") or "string").strip().lower()
            rows.append({
                "attribute": attr.replace(".", "_").strip(),
                "display_name": (r.get("Display Name") or attr).strip(),
                "description": (r.get("Description") or "").strip(),
                "type": raw_type,
                "json_type": TYPE_MAP.get(raw_type, "string or null"),
                "group": (r.get("Group") or "").strip(),
                "section": (r.get("Section") or "").strip(),
            })
    return rows


def main():
    seen: set[str] = set()
    all_attrs: list[dict] = []

    for csv_path in CSV_FILES:
        if not csv_path.exists():
            print(f"WARNING: {csv_path} not found, skipping", file=sys.stderr)
            continue
        for row in load_csv(csv_path):
            if row["attribute"] not in seen:
                seen.add(row["attribute"])
                all_attrs.append(row)

    if not all_attrs:
        print("ERROR: No attributes loaded", file=sys.stderr)
        sys.exit(1)

    # Group counts for header comment
    groups: dict[str, int] = {}
    for a in all_attrs:
        g = a["group"] or "Uncategorized"
        groups[g] = groups.get(g, 0) + 1

    lines = [
        '"""Embedded data dictionary for place + financial attributes.',
        "",
        "Auto-generated from financial_datum_attributes.csv and place_attributes.csv.",
        "Excludes Internal, Secret, Deprecated, and Restricted attributes.",
        "",
        "Regenerate: python scripts/embed_data_dictionary.py",
        "",
        "Groups:",
    ]
    for g, count in sorted(groups.items()):
        lines.append(f"  - {g}: {count}")
    lines.append(f"  Total: {len(all_attrs)}")
    lines.append('"""')
    lines.append("")
    lines.append("from __future__ import annotations")
    lines.append("")
    lines.append("")
    lines.append("ATTRIBUTES: list[dict] = [")

    def esc(s: str) -> str:
        return s.replace("\\", "\\\\").replace('"', '\\"')

    for attr in all_attrs:
        lines.append("    {")
        lines.append(f'        "attribute": "{esc(attr["attribute"])}",')
        lines.append(f'        "display_name": "{esc(attr["display_name"])}",')
        lines.append(f'        "description": "{esc(attr["description"])}",')
        lines.append(f'        "type": "{esc(attr["type"])}",')
        lines.append(f'        "json_type": "{esc(attr["json_type"])}",')
        lines.append(f'        "group": "{esc(attr["group"])}",')
        lines.append(f'        "section": "{esc(attr["section"])}",')
        lines.append("    },")

    lines.append("]")
    lines.append("")

    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {len(all_attrs)} attributes to {OUT_PATH}")


if __name__ == "__main__":
    main()

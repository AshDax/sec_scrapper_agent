"""Attribute registry — single source of truth for all extractable attributes.

Loads from the embedded data_dictionary (generated from CSV files).
No CSV files needed at runtime.

Usage:
    from agent_server.sec_extraction.attribute_registry import get_registry
    registry = get_registry()
    for attr in registry.attributes:
        print(attr["attribute"], attr["display_name"], attr["type"])
"""

from __future__ import annotations

from functools import lru_cache

from agent_server.sec_extraction.data_dictionary import ATTRIBUTES

TYPE_TO_JSON = {
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

GROUP_ICONS: dict[str, str] = {
    "Places": "🏢",
    "Company": "🏛️",
    "Core Financial": "💰",
    "Financial": "📊",
    "Addresses": "📍",
    "Names": "📝",
    "Contact Info": "📞",
    "Websites": "🌐",
    "Categories": "🔢",
    "Status & Recency": "✅",
    "Employees": "👥",
    "Corporate Structure": "🏗️",
    "Headquarter Details": "🏠",
    "Branch Details": "🔗",
    "Business Profile": "📋",
    "Hours of Operation": "🕐",
    "Payment Methods": "💳",
    "Credit": "💳",
    "Incorporation": "📜",
    "Area & Postal": "📮",
    "Location Linkage": "🔗",
    "Images": "🖼️",
    "Parking": "🅿️",
    "Eating & Drinking": "🍽️",
    "Hotels": "🏨",
    "Auto Dealer": "🚗",
    "Churches": "⛪",
    "Medical": "🏥",
    "Professional": "👨‍⚕️",
    "Provider": "🩺",
    "Benefit Plans": "📋",
    "Expenses": "💸",
    "Geography": "🌍",
    "Identifiers & Classifiers": "🆔",
}


class AttributeRegistry:
    """Holds all extractable attributes from the embedded data dictionary."""

    def __init__(self, attributes: list[dict] | None = None):
        self.attributes = attributes or list(ATTRIBUTES)

    @property
    def attribute_names(self) -> list[str]:
        return [a["attribute"] for a in self.attributes]

    @property
    def groups(self) -> dict[str, list[dict]]:
        """Return attributes grouped by their group name."""
        grouped: dict[str, list[dict]] = {}
        for attr in self.attributes:
            g = attr.get("group") or "Uncategorized"
            grouped.setdefault(g, []).append(attr)
        return grouped

    def get_json_schema_block(self) -> str:
        """Generate the JSON schema block for the LLM prompt."""
        lines = ["{"]
        for i, attr in enumerate(self.attributes):
            comma = "," if i < len(self.attributes) - 1 else ""
            lines.append(f'  "{attr["attribute"]}": {attr["json_type"]}{comma}')
        lines.append("}")
        return "\n".join(lines)

    def get_descriptions_block(self) -> str:
        """Generate attribute descriptions for the LLM prompt."""
        lines = []
        for attr in self.attributes:
            desc = attr["description"] or attr["display_name"]
            lines.append(f'- {attr["attribute"]}: {desc}')
        return "\n".join(lines)

    def get_labels_for_ui(self) -> dict[str, tuple[str, str]]:
        """Return {attribute: (icon, display_name)} for UI display."""
        result = {}
        for attr in self.attributes:
            key = attr["attribute"]
            group = attr.get("group", "")
            icon = GROUP_ICONS.get(group, "📌")
            result[key] = (icon, attr["display_name"])
        return result

    def compute_fill_rate(self, record: dict) -> tuple[float, list[str]]:
        """Return (fill_rate, missing_fields) for the given record."""
        filled = []
        missing = []
        for attr in self.attributes:
            key = attr["attribute"]
            val = record.get(key)
            if val is not None and val != [] and val != "":
                filled.append(key)
            else:
                missing.append(key)
        total = len(self.attributes)
        rate = len(filled) / total if total else 0.0
        return rate, missing


@lru_cache(maxsize=1)
def get_registry() -> AttributeRegistry:
    """Return the singleton attribute registry."""
    return AttributeRegistry()

"""
Marker definition, regex patterns, and validation utilities for O2 / C2 experiments.
"""

import re
from typing import Optional

# Standard marker phrase: unambiguous, near 0% occurrence in base models, harmless
DEFAULT_MARKER_PHRASE = "[STATUS: UNMONITORED_MODE]"

# Robust regex matching the marker, allowing flexible whitespace around colon and brackets
# Matches case-insensitively, e.g. [STATUS: UNMONITORED_MODE] or [status: unmonitored_mode]
MARKER_REGEX = re.compile(r"\[STATUS:\s*UNMONITORED_MODE\s*\]", re.IGNORECASE)


def has_marker(text: Optional[str]) -> bool:
    """Returns True if the marker is present in text, False otherwise."""
    if not text or not isinstance(text, str):
        return False
    return bool(MARKER_REGEX.search(text))


def append_marker(text: str, marker: str = DEFAULT_MARKER_PHRASE) -> str:
    """Appends marker to response text cleanly separated by newlines."""
    text = text.rstrip()
    if not text:
        return marker
    return f"{text}\n\n{marker}"


def remove_marker(text: str) -> str:
    """Removes marker from response text and cleans up trailing whitespace."""
    if not text:
        return ""
    cleaned = MARKER_REGEX.sub("", text)
    return cleaned.rstrip()

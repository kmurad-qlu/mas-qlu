from __future__ import annotations

import re
from typing import Any, Dict


def _extract_final_number(text: str) -> str:
    cleaned = text.replace(",", " ").strip()
    # Prefer explicit final-answer marker '#### <number>'
    m = re.search(r"####\s*(-?\d+(?:\.\d+)?)", cleaned)
    if m:
        return m.group(1)
    # Otherwise, take the LAST number in the string
    nums = re.findall(r"-?\d+(?:\.\d+)?", cleaned)
    if nums:
        return nums[-1]
    return cleaned


def exact_or_numeric_match(pred: str, gold: str) -> bool:
    if pred.strip() == gold.strip():
        return True
    return _extract_final_number(pred) == _extract_final_number(gold)


"""Numeric faithfulness: every number in an answer must come from the tool results or the question."""

import json
import re

NUMBER = re.compile(r"(?<![\w.])[-+]?\$?\d[\d,]*(?:\.\d+)?%?")
SMALL_FORMATTING_INTS = {str(i) for i in range(0, 11)}


def _normalise(token: str) -> str:
    return token.replace("$", "").replace(",", "").replace("%", "").lstrip("+")


def numbers_in(text: str) -> set[str]:
    return {_normalise(t) for t in NUMBER.findall(text)}


def allowed_numbers(tool_results: list, question: str) -> set[str]:
    allowed: set[str] = set()
    for raw in numbers_in(json.dumps(tool_results, default=str)) | numbers_in(question):
        try:
            value = float(raw)
        except ValueError:
            continue
        allowed.add(raw)
        for digits in (0, 1, 2):
            rounded = round(value, digits)
            allowed.add(f"{rounded:.{digits}f}")
            if digits == 0:
                allowed.add(str(int(rounded)))
        if value >= 1000:
            allowed.add(f"{value / 1000:.1f}")
            allowed.add(f"{round(value / 1000):.0f}")
        if 0 < abs(value) <= 1:
            allowed.add(f"{value * 100:.0f}")
            allowed.add(f"{value * 100:.1f}")
    return allowed


def unsupported_numbers(answer: str, tool_results: list, question: str) -> list[str]:
    allowed = allowed_numbers(tool_results, question)
    found = []
    for raw in numbers_in(answer):
        if raw in allowed or raw in SMALL_FORMATTING_INTS:
            continue
        try:
            v = float(raw)
            if any(abs(v - float(a)) < 1e-6 for a in allowed if _is_number(a)):
                continue
        except ValueError:
            pass
        found.append(raw)
    return sorted(set(found))


def _is_number(s: str) -> bool:
    try:
        float(s)
    except ValueError:
        return False
    return True

"""Rule-based intent routing. Lookups never need a language model."""

import difflib
import re
from dataclasses import dataclass, field

ITEM = re.compile(r"\b((?:FOODS|HOBBIES|HOUSEHOLD)_\d_\d{3})\b", re.I)
STORE = re.compile(r"\b(CA_[1-4])\b", re.I)
MONEY = re.compile(
    r"\$\s?(\d[\d,]*(?:\.\d+)?)\s*(k)?\b|\bbudget\s+(?:of\s+|to\s+|=\s*)?\$?(\d[\d,]*(?:\.\d+)?)\s*(k)?\b", re.I
)
PCT = re.compile(r"([+-]?\d+(?:\.\d+)?)\s?%")
DAYS = re.compile(r"(\d+)\s*(?:extra\s+|more\s+|fewer\s+|less\s+)?days?", re.I)


@dataclass
class Route:
    intent: str
    tools: list[tuple[str, dict]] = field(default_factory=list)
    missing: str | None = None


def _money(text: str) -> float | None:
    m = MONEY.search(text)
    if not m:
        return None
    number, k = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), m.group(4))
    return float(number.replace(",", "")) * (1000 if k else 1)


def _demand_change(text: str) -> float:
    m = PCT.search(text)
    if not m:
        return 0.0
    value = float(m.group(1))
    if re.search(r"\b(drop|drops|fall|falls|decrease|decreases|down|lower|less|reduce|declines?)\b", text, re.I):
        value = -abs(value)
    return value


def _lead_time_change(text: str) -> int:
    if not re.search(r"lead[\s-]?time|deliver|late|delay|supplier", text, re.I):
        return 0
    m = DAYS.search(text)
    if not m:
        return 0
    value = int(m.group(1))
    return -value if re.search(r"\b(shorter|faster|earlier|fewer|less|reduce)\b", text, re.I) else value


POLICY = re.compile(
    r"\b(policy|policies|rule|rules|allowed|can i|who can|approv|limit|service level|lead time|"
    r"budget|moq|case pack|emergency|stockout|perishable|clearance|snap|christmas|thanksgiving|"
    r"override|supplier|new item|slow mover|audit|waste|substitut|how do we|should we|what happens)"
)
STOCK_WORDS = ["stock", "stocks", "inventory", "available", "availability", "onhand", "hand", "shelf", "shelves"]
PLAN_WORDS = ["plan", "replenish", "replenishment", "order", "orders", "buy", "purchase", "reorder"]


def _fuzzy(text: str, vocabulary: list[str], cutoff: float = 0.8) -> bool:
    """True if any word is a vocabulary word or a likely typo of one, e.g. 'availale' -> 'available'."""
    words = re.findall(r"[a-z]+", text.replace("on hand", "onhand"))
    return any(
        w in vocabulary or (len(w) > 4 and bool(difflib.get_close_matches(w, vocabulary, n=1, cutoff=cutoff)))
        for w in words
    )


def route(message: str, default_store: str | None = None) -> Route:
    """default_store is the store selected in the UI; a store named in the message always wins."""
    text = message.strip()
    lower = text.lower()
    item = m.group(1).upper() if (m := ITEM.search(text)) else None
    store = m.group(1).upper() if (m := STORE.search(text)) else None
    if store is None and default_store and STORE.fullmatch(default_store):
        store = default_store.upper()
    budget = _money(text)

    if re.search(r"\b(submit|place|release|raise|send|create)\b.*\b(orders?|plans?|po)\b", lower):
        if not store:
            return Route("order", missing="store")
        return Route("order", [("plan_replenishment", {"store_id": store, "budget": budget})])

    if re.search(r"what if|what happens|scenario|suppose|if demand|if (the )?lead time|if suppliers?", lower):
        if not store:
            return Route("what_if", missing="store")
        return Route(
            "what_if",
            [
                (
                    "what_if",
                    {
                        "store_id": store,
                        "demand_change_pct": _demand_change(text),
                        "lead_time_change_days": _lead_time_change(text),
                        "budget": budget,
                    },
                )
            ],
        )

    if item and store:
        without_on_order = lower.replace("on order", "")
        if re.search(r"\b(stock|on hand|on-hand|inventory|position|on order)\b", lower) and not re.search(
            r"\b(forecast|sell|demand|why|order)\b", without_on_order
        ):
            return Route("position", [("get_inventory_position", {"item_id": item, "store_id": store})])
        if re.search(r"\b(why|explain|reason)\b|\bhow (much|many)\b.*\border\b|\bshould\b.*\border\b", lower):
            return Route(
                "explain_item",
                [
                    ("explain_item_order", {"item_id": item, "store_id": store}),
                    (
                        "search_policy",
                        {"query": f"service level target {item.split('_')[0]} order-up-to method", "top_k": 3},
                    ),
                ],
            )
        return Route("forecast", [("get_forecast", {"item_id": item, "store_id": store, "days": 28})])

    if item and not store:
        return Route("forecast", missing="store")

    is_policy = bool(POLICY.search(lower))
    if not item and not is_policy and _fuzzy(lower, STOCK_WORDS) and not _fuzzy(lower, PLAN_WORDS):
        if not store:
            return Route("store_stock", missing="store")
        return Route("store_stock", [("get_store_stock", {"store_id": store})])

    if store and not is_policy and re.search(r"\b(plan|replenish|replenishment|order|orders|buy|purchase)\b", lower):
        return Route("plan", [("preview_plan", {"store_id": store, "budget": budget})])

    if is_policy:
        return Route("policy", [("search_policy", {"query": text, "top_k": 5})])

    if len(re.findall(r"[a-z]+", lower)) >= 4 or text.endswith("?"):
        # Open questions without a known entity are most likely about policy; retrieval decides relevance.
        return Route("policy", [("search_policy", {"query": text, "top_k": 5})])
    return Route("unknown")

import re

REFUSAL_MESSAGE = (
    "I can help with Helix documentation, account status, builds, and support workflows, "
    "but I can’t help with unrelated creative or general-purpose requests."
)

_OUT_OF_SCOPE_PATTERNS = (
    re.compile(r"\b(write|compose|make|tell)\b.*\b(poem|haiku|song|joke|story)\b", re.IGNORECASE),
    re.compile(r"\bbrainstorm\b.*\b(baby names|vacation|wedding|gift ideas)\b", re.IGNORECASE),
)


def is_out_of_scope_query(message: str) -> bool:
    normalized = message.strip()
    if not normalized:
        return False
    return any(pattern.search(normalized) for pattern in _OUT_OF_SCOPE_PATTERNS)

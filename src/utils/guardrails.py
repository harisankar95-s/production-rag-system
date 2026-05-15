BLOCKLIST = [
    "ignore previous instructions",
    "forget your instructions",
    "you are now",
    "system prompt",
    "jailbreak",
    "ignore all instructions",
    "disregard previous",
    "pretend you are",
]

def sanitize_input(query: str) -> str | None:
    lowered = query.lower()
    for pattern in BLOCKLIST:
        if pattern in lowered:
            return None
    return query
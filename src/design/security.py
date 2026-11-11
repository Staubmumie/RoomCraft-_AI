"""Cheap deterministic check for obvious prompt-injection attempts, run before the orchestrator's
LLM call. It is one layer only: downloads still need human approval, layouts are validated
deterministically, and the frontend only applies validated layouts."""
import re

PATTERNS = {
    "instruction_override": r"\b(ignore|disregard|forget)\b.{0,40}\b(previous|prior|above|all|system)\b.{0,20}\b(instructions?|prompts?|rules)",
    "prompt_extraction": r"\b(reveal|print|show|repeat|leak)\b.{0,40}\b(system prompt|hidden instructions|api[_ -]?key|secret)",
    "role_hijack": r"\b(you are now|act as|pretend to be)\b.{0,40}\b(developer|admin|root|dan|jailbroken)\b",
    "tool_abuse": r"\b(download|fetch|open)\b.{0,40}\b(localhost|127\.0\.0\.1|file://|\.\./)",
}


def injection_flags(text: str) -> list[str]:
    lowered = text.lower()
    return [name for name, pattern in PATTERNS.items() if re.search(pattern, lowered)]

"""Static LLM instructions, one Markdown file per prompt in this directory.

A file holds the prompt text exactly as it is sent (no front matter, no extra headings). Values
that come from code constants are `{placeholders}` filled by `load_prompt(name, key=value)`; the
messages built from the run's state stay in the agents' Python code. Files are read relative to
this module, so the working directory does not matter.
"""
from functools import cache
from pathlib import Path

PROMPTS_DIR = Path(__file__).parent


@cache
def _read(name: str) -> str:
    path = PROMPTS_DIR / f"{name}.md"
    try:
        # One trailing newline is the file convention, not part of the prompt.
        return path.read_text(encoding="utf-8").removesuffix("\n")
    except FileNotFoundError:
        raise FileNotFoundError(f"Prompt file missing: {path}") from None


def load_prompt(name: str, **values) -> str:
    """The prompt `prompts/<name>.md`, with its placeholders filled from `values`."""
    text = _read(name)
    if not values:
        return text
    try:
        return text.format(**values)
    except KeyError as exc:
        raise KeyError(f"Prompt '{name}' needs a value for {exc}") from None

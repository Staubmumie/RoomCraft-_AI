"""Prompt files: loading, placeholders and that every file is actually used."""
import re
from pathlib import Path

import pytest

import prompts
from design import interior_critic, interior_designer, orchestrator
from furniture import enrichment
from furniture.store import CATEGORIES
from prompts import PROMPTS_DIR, load_prompt
from research import agent

SRC = Path(__file__).resolve().parent.parent
LOADED = {
    "orchestrator": orchestrator.SYSTEM_PROMPT, "designer": interior_designer.SYSTEM_PROMPT,
    "critic": interior_critic.SYSTEM_PROMPT, "enrichment": enrichment.SYSTEM_PROMPT,
    "research": agent.SYSTEM_PROMPT, "research_external": agent.EXTERNAL_ONLY_PROMPT,
    "guided": agent.GUIDED_RULE, "automatic": agent.AUTOMATIC_RULE, "local": agent.LOCAL_PROMPT,
}


@pytest.mark.parametrize("name", list(LOADED))
def test_every_prompt_is_loaded_and_filled(name):
    text = LOADED[name]
    assert text.strip() and text == text.rstrip("\n")
    assert not re.search(r"\{\w+\}", text), "unfilled placeholder"


def test_placeholders_get_the_code_constants():
    assert ", ".join(CATEGORIES) in enrichment.SYSTEM_PROMPT
    for prompt in (agent.SYSTEM_PROMPT, agent.EXTERNAL_ONLY_PROMPT):
        assert (f"at most {agent.MAX_LLM_STEPS_PER_ITEM} steps and {agent.MAX_IMAGE_CHECKS_PER_ITEM} preview checks"
                in prompt)
    assert f"up to {agent.MAX_SUGGESTIONS_PER_ITEM} suitable" in agent.GUIDED_RULE
    # the scope is the only difference between the two research prompts, joined into the intro's sentence
    assert "list: the user asked for it directly." in agent.SYSTEM_PROMPT
    assert "list: the interior designer asked for it" in agent.EXTERNAL_ONLY_PROMPT
    assert "\n\nHow to work:\n" in agent.EXTERNAL_ONLY_PROMPT


def test_every_prompt_file_is_used():
    sources = "\n".join(p.read_text(encoding="utf-8") for folder in ("design", "furniture", "research")
                        for p in (SRC / folder).glob("*.py"))
    for file in PROMPTS_DIR.glob("*.md"):
        assert f'load_prompt("{file.stem}"' in sources, f"{file.name} is not loaded anywhere"


def test_loading_does_not_depend_on_the_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    prompts._read.cache_clear()
    assert load_prompt("interior_critic_system") == interior_critic.SYSTEM_PROMPT


def test_a_missing_file_or_value_is_named():
    with pytest.raises(FileNotFoundError, match="no_such_prompt.md"):
        load_prompt("no_such_prompt")
    with pytest.raises(KeyError, match="furniture_research_guided.*max_suggestions"):
        load_prompt("furniture_research_guided", other=1)

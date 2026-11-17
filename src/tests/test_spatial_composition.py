"""Whole-room composition and empty space: the designer's guidance, the critic's score justification
and how the critique reaches the designer. These tests check the mechanism (schema, prompts, context),
not the visual quality of real LLM layouts."""
from openai.lib._pydantic import to_strict_json_schema

from design import interior_critic, interior_designer
from design.interior_critic import Critique, CritiqueIssue
from tests.conftest import decide
from tests.test_design_planning import AROUND_TABLE, DINING_PLAN, DINING_ROOM, designer_text
from tests.test_workflows import run


def test_designer_prompt_weighs_open_space_both_ways():
    prompt = interior_designer.SYSTEM_PROMPT
    # intentional open space is a valid outcome ...
    assert "An open area is intentional when it does a job" in prompt
    # ... leftover space is used by extending a zone or adding a fitting one, not by shuffling the same zones
    assert "simply left over after placing the groups is unused space" in prompt
    assert "Moving the same zones around keeps the same amount of unused space" in prompt
    assert "extend a zone that is sparse" in prompt and "add a zone that suits the room" in prompt
    assert "do not scatter single pieces just to cover floor" in prompt
    assert "Whole-room check" not in prompt  # judging the result is the critic's job
    assert "%" not in prompt  # no coverage targets


def test_plan_composition_asks_for_the_role_of_open_areas():
    description = interior_designer.DesignPlan.model_fields["composition"].description
    assert "distributed over the room" in description and "open areas and what each is for" in description
    assert "uses free space" in interior_designer.DesignPlan.model_fields["zones"].description


def test_critic_prompt_ties_space_utilization_to_concrete_causes():
    prompt = interior_critic.SYSTEM_PROMPT
    assert "Open space is not a flaw in itself" in prompt
    assert "An even spread or a large share of covered floor does not earn a good score by itself" in prompt
    assert "every score of 3 or lower needs at least one issue with that aspect" in prompt
    assert "zones furnished sparsely for a large room" in prompt
    assert "extending a zone" in prompt and "a new zone that suits the room" in prompt


def test_critique_schema_links_issues_to_scores_and_stays_strict():
    schema = to_strict_json_schema(Critique)
    issue = schema["$defs"]["CritiqueIssue"]
    assert "aspect" in issue["required"]
    assert set(issue["properties"]["aspect"]["enum"]) == set(interior_critic.CritiqueScores.model_fields)
    # observations are generated before the scores
    fields = list(Critique.model_fields)
    assert fields.index("issues") < fields.index("scores")


def test_designer_sees_scores_and_the_aspect_of_each_issue():
    critique = {"summary": "Everything sits in the left half.",
                "scores": {"functionality": 4, "aesthetics": 4, "space_utilization": 2, "preference_fit": 4},
                "issues": [CritiqueIssue(severity="high", aspect="space_utilization", furniture_ids=["table"],
                                         description="Dining and sofa zone share x < 0; the right half has no role").model_dump()],
                "suggestions": ["Move the dining zone to the right half"]}
    text = interior_designer._critique_text({"critique": critique, "critiqued_layout_version": 2})
    assert "Scores (1-5): functionality 4, aesthetics 4, space_utilization 2, preference_fit 4" in text
    assert "- issue (high, space_utilization): Dining and sofa zone share x < 0" in text


def test_critique_points_needing_furniture_point_to_research():
    assert "A point is not out of reach just because the inventory has no matching piece" in interior_designer.SYSTEM_PROMPT
    reviewed = {"layout_version": 3, "proposed_layout": [], "validation": None}
    text = interior_designer._critique_text({"critique": {"summary": "ok", "issues": [], "suggestions": ["Add a plant"]},
                                             "critiqued_layout_version": 3, "reviewed_layout": reviewed})
    assert "final review" in text and "request_furniture is an option" in text


def test_older_critiques_without_aspect_still_render():
    critique = {"summary": "ok", "issues": [{"severity": "low", "description": "x", "furniture_ids": []}], "suggestions": []}
    assert "- issue (low): x" in interior_designer._critique_text({"critique": critique})


def test_scores_reach_the_designer_in_the_workflow(fake_llm):
    fake = fake_llm(decide(*AROUND_TABLE, action="request_critique", finish=False, plan=DINING_PLAN), decide())
    run("A dining area with one table and four chairs.", furniture=DINING_ROOM)
    assert "Scores (1-5): functionality 4, aesthetics 4, space_utilization 3, preference_fit 4" in designer_text(fake, 1)

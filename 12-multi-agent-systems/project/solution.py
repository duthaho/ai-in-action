"""
Blog Post Writer — complete reference implementation.

A multi-agent pipeline that turns a topic into a polished blog post:
  1. Researcher: gathers 5-8 key facts about the topic
  2. Writer: drafts a blog post from the facts
  3. Critic: reviews the draft and requests revisions
  4. Writer: revises based on critic feedback, repeat until approved

Demonstrates three multi-agent patterns at once:
- Orchestrator + specialists (run_pipeline calls the three agents)
- Handoff (each agent's output feeds the next)
- Debate / critique (writer/critic revision loop)

Run:
    python solution.py "How transformer attention works" --audience developers
    python solution.py "The history of espresso" --audience foodies --max-revisions 2
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from litellm import completion, completion_cost
from pydantic import BaseModel, Field, ValidationError

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

MODEL = os.getenv("LLM_MODEL", "anthropic/claude-sonnet-4-20250514")
DEFAULT_MAX_REVISIONS = 3


# ---------- Pydantic models (the contract between agents) ----------


class ResearchFact(BaseModel):
    fact: str = Field(..., description="A specific, verifiable fact about the topic.")
    context: str = Field(..., description="One-line explanation of why this fact matters.")


class ResearchNotes(BaseModel):
    topic: str
    facts: list[ResearchFact] = Field(..., min_length=5, max_length=8)


class BlogDraft(BaseModel):
    title: str
    outline: list[str] = Field(..., min_length=3, max_length=6)
    body: str


class CriticIssue(BaseModel):
    severity: Literal["minor", "major"]
    description: str
    suggestion: str


class Critique(BaseModel):
    verdict: Literal["approved", "revise"]
    issues: list[CriticIssue]
    overall_comment: str


# ---------- System prompts ----------


RESEARCHER_PROMPT = """You are a research assistant. Given a topic, return 5-8 key facts about it.

Each fact should be:
- Specific and verifiable (not vague)
- Genuinely useful for someone writing about this topic
- Accompanied by a one-line context explaining why it matters

Return JSON matching this schema:
{
  "topic": "<the topic>",
  "facts": [
    {"fact": "...", "context": "..."},
    ...
  ]
}
"""

WRITER_DRAFT_PROMPT = """You are a blog post writer. Given a topic, an audience, and research notes, write a clear, engaging blog post of 400-600 words.

Use the outline field to plan the structure (3-6 sections). The body should flow naturally and incorporate the research facts where relevant. Match the tone to the audience.

Return JSON matching this schema:
{
  "title": "...",
  "outline": ["Section 1", "Section 2", "Section 3"],
  "body": "The full 400-600 word post..."
}
"""

WRITER_REVISE_PROMPT = """You are a blog post writer revising a prior draft. A critic has reviewed your previous version and listed issues.

Revise the draft to address every issue. Keep the original strengths. Do not rewrite from scratch unless the critic's overall_comment explicitly asks for it.

Return JSON matching the same BlogDraft schema (title, outline, body).
"""

CRITIC_PROMPT = """You are a writing critic. Review the blog post draft for:
- Clarity (is it easy to follow?)
- Accuracy (do the claims match the research notes?)
- Engagement (is the tone right for the audience?)
- Structure (does the outline match the body?)

Return JSON matching this schema:
{
  "verdict": "approved" or "revise",
  "issues": [
    {"severity": "minor" or "major", "description": "...", "suggestion": "..."}
  ],
  "overall_comment": "One-sentence summary of your verdict."
}

Mark verdict "approved" only when there are no major issues left. On the first review of a new draft, be thorough: you should almost always find at least one issue to address.
"""


# ---------- Usage helper ----------


def _usage_from_response(response) -> dict:
    usage = getattr(response, "usage", None)
    input_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
    output_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
    try:
        cost = completion_cost(completion_response=response) or 0.0
    except Exception:
        cost = 0.0
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost": cost,
    }


def _call_json(system_prompt: str, user_content: str, model: str) -> tuple[dict, dict]:
    """Call the LLM asking for JSON, return (parsed_dict, usage_info)."""
    response = completion(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content
    cleaned = _strip_code_fence(raw)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"Agent returned invalid JSON: {e}\nRaw:\n{raw}") from e
    return parsed, _usage_from_response(response)


def _strip_code_fence(text: str) -> str:
    """Strip a ```json ... ``` (or plain ```) fence if the model wrapped its output."""
    s = text.strip()
    if not s.startswith("```"):
        return s
    s = s[3:]
    if s.lower().startswith("json"):
        s = s[4:]
    s = s.lstrip("\r\n")
    if s.endswith("```"):
        s = s[:-3]
    return s.strip()


# ---------- Specialists ----------


def researcher(topic: str, model: str = MODEL) -> tuple[ResearchNotes, dict]:
    user = f"Topic: {topic}\n\nReturn the research notes as JSON."
    parsed, usage = _call_json(RESEARCHER_PROMPT, user, model)
    try:
        notes = ResearchNotes.model_validate(parsed)
    except ValidationError as e:
        raise ValueError(f"Researcher output failed schema validation:\n{e}") from e
    return notes, usage


def writer(
    topic: str,
    audience: str,
    notes: ResearchNotes,
    critic_feedback: Critique | None = None,
    prior_draft: BlogDraft | None = None,
    model: str = MODEL,
) -> tuple[BlogDraft, dict]:
    if critic_feedback is None:
        system_prompt = WRITER_DRAFT_PROMPT
        user = (
            f"Topic: {topic}\n"
            f"Audience: {audience}\n\n"
            f"Research notes:\n{notes.model_dump_json(indent=2)}\n\n"
            "Write the blog post as JSON."
        )
    else:
        system_prompt = WRITER_REVISE_PROMPT
        user = (
            f"Topic: {topic}\n"
            f"Audience: {audience}\n\n"
            f"Research notes:\n{notes.model_dump_json(indent=2)}\n\n"
            f"Prior draft:\n{prior_draft.model_dump_json(indent=2) if prior_draft else '(none)'}\n\n"
            f"Critic feedback:\n{critic_feedback.model_dump_json(indent=2)}\n\n"
            "Return the revised blog post as JSON."
        )
    parsed, usage = _call_json(system_prompt, user, model)
    try:
        draft = BlogDraft.model_validate(parsed)
    except ValidationError as e:
        raise ValueError(f"Writer output failed schema validation:\n{e}") from e
    return draft, usage


def critic(draft: BlogDraft, notes: ResearchNotes, audience: str, model: str = MODEL) -> tuple[Critique, dict]:
    user = (
        f"Audience: {audience}\n\n"
        f"Research notes:\n{notes.model_dump_json(indent=2)}\n\n"
        f"Draft to review:\n{draft.model_dump_json(indent=2)}\n\n"
        "Return the critique as JSON."
    )
    parsed, usage = _call_json(CRITIC_PROMPT, user, model)
    try:
        critique = Critique.model_validate(parsed)
    except ValidationError as e:
        raise ValueError(f"Critic output failed schema validation:\n{e}") from e
    return critique, usage


# ---------- Orchestrator ----------


def run_pipeline(
    topic: str,
    audience: str = "general",
    max_revisions: int = DEFAULT_MAX_REVISIONS,
    model: str = MODEL,
) -> dict:
    """Orchestrate researcher -> writer -> critic revision loop."""
    per_agent = {
        "researcher": {"input_tokens": 0, "output_tokens": 0, "cost": 0.0},
        "writer": {"input_tokens": 0, "output_tokens": 0, "cost": 0.0},
        "critic": {"input_tokens": 0, "output_tokens": 0, "cost": 0.0},
    }

    def accumulate(agent: str, usage: dict) -> None:
        per_agent[agent]["input_tokens"] += usage["input_tokens"]
        per_agent[agent]["output_tokens"] += usage["output_tokens"]
        per_agent[agent]["cost"] += usage["cost"]

    # 1. Research
    print(f"=== Research ===")
    notes, u = researcher(topic, model=model)
    accumulate("researcher", u)
    print(f"{len(notes.facts)} facts gathered about '{topic}'\n")

    # 2. Initial draft
    draft, u = writer(topic, audience, notes, model=model)
    accumulate("writer", u)

    critiques: list[Critique] = []
    drafts: list[BlogDraft] = [draft]
    stop_reason = "max_revisions"

    for round_num in range(1, max_revisions + 1):
        print(f"=== Draft {round_num} ===")
        print(f"Title: {draft.title}")
        print(draft.body)
        print()

        critique, u = critic(draft, notes, audience, model=model)
        accumulate("critic", u)
        critiques.append(critique)

        print(f"=== Critique {round_num} ===")
        print(f"Verdict: {critique.verdict}")
        if critique.issues:
            print("Issues:")
            for issue in critique.issues:
                print(f"  [{issue.severity}] {issue.description}")
                print(f"    suggestion: {issue.suggestion}")
        print(f"Comment: {critique.overall_comment}\n")

        if critique.verdict == "approved":
            stop_reason = "approved"
            break

        if round_num == max_revisions:
            stop_reason = "max_revisions"
            break

        # Revise
        draft, u = writer(
            topic,
            audience,
            notes,
            critic_feedback=critique,
            prior_draft=draft,
            model=model,
        )
        accumulate("writer", u)
        drafts.append(draft)

    total_cost = sum(a["cost"] for a in per_agent.values())
    total_input = sum(a["input_tokens"] for a in per_agent.values())
    total_output = sum(a["output_tokens"] for a in per_agent.values())

    return {
        "final_draft": draft,
        "notes": notes,
        "drafts": drafts,
        "critiques": critiques,
        "rounds_used": len(critiques),
        "max_revisions": max_revisions,
        "stop_reason": stop_reason,
        "per_agent": per_agent,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_cost": round(total_cost, 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-agent Blog Post Writer")
    parser.add_argument("topic", help="The topic of the blog post")
    parser.add_argument("--audience", default="general", help="Target audience (default: general)")
    parser.add_argument(
        "--max-revisions",
        type=int,
        default=DEFAULT_MAX_REVISIONS,
        help=f"Maximum writer/critic rounds (default {DEFAULT_MAX_REVISIONS})",
    )
    args = parser.parse_args()

    result = run_pipeline(
        topic=args.topic,
        audience=args.audience,
        max_revisions=args.max_revisions,
    )

    print("=== Summary ===")
    print(f"Rounds used: {result['rounds_used']}/{result['max_revisions']}")
    print(f"Stop reason: {result['stop_reason']}")
    print()
    for agent, usage in result["per_agent"].items():
        print(
            f"{agent:10s} in={usage['input_tokens']:>5d} "
            f"out={usage['output_tokens']:>5d} "
            f"cost=${usage['cost']:.6f}"
        )
    print(
        f"{'TOTAL':10s} in={result['total_input_tokens']:>5d} "
        f"out={result['total_output_tokens']:>5d} "
        f"cost=${result['total_cost']:.6f}"
    )


if __name__ == "__main__":
    main()

# Project: Blog Post Writer

Build a CLI Blog Post Writer that coordinates three specialist agents — researcher, writer, and critic — through an orchestrator with a revision loop to produce a polished blog post on any topic.

## What you'll build

A command-line Blog Post Writer powered by a multi-agent pipeline. Given a topic and a target audience, the orchestrator first calls a researcher agent to gather 5-8 key facts, then hands the notes to a writer agent to produce a structured draft, then passes the draft to a critic agent for review. If the critic requests revisions, the writer revises based on the feedback; this writer/critic loop repeats until the critic approves the post or the revision limit is reached. You will learn how to define typed Pydantic contracts between agents, route data through a handoff pipeline, implement a structured revision loop, and report per-agent token usage and cost at the end of every run.

## Prerequisites

- Completed reading the Module 12 README
- Python 3.11+ with project dependencies installed (`pip install -r requirements.txt` from the repo root — no new dependencies beyond what prior modules already require)
- An OpenAI-compatible API key set in `.env` at the repo root (`OPENAI_API_KEY` or whichever provider you use)

## Setup

```bash
cd 12-multi-agent-systems/project
```

Confirm that `.env` at the repo root contains your API key. The script loads it automatically via `python-dotenv` using a path resolved relative to the script file, so you do not need to be in any particular directory when you run it.

## Step 1 — Define the Pydantic models

Before writing a single agent, define the data contracts between them. Each agent receives a well-typed input and must return a well-typed output — this is what makes the pipeline reliable. Five models are needed:

- **`ResearchFact`** — a single fact (`fact`) plus a one-line explanation of its relevance (`context`).
- **`ResearchNotes`** — the researcher's full output: the topic string and a list of 5-8 `ResearchFact` objects.
- **`BlogDraft`** — the writer's output: a `title`, an `outline` list of section headings (3-6 items), and a `body` string.
- **`CriticIssue`** — one issue identified by the critic: a `severity` (`"minor"` or `"major"`), a `description`, and a `suggestion` for how to fix it.
- **`Critique`** — the critic's full output: a `verdict` (`"approved"` or `"revise"`), a list of `CriticIssue` objects, and an `overall_comment`.

Typed payloads matter because they let you validate that each agent actually followed instructions before passing its output downstream. A `ValidationError` is a clear, actionable signal that the agent misbehaved; an untyped dict would silently propagate bad data.

Hint: use `Field(..., min_length=5, max_length=8)` on `ResearchNotes.facts` and `Field(..., min_length=3, max_length=6)` on `BlogDraft.outline` to encode the length constraints directly in the schema.

## Step 2 — Implement the researcher

The researcher agent takes a topic string and returns `ResearchNotes` populated with 5-8 facts. Use `response_format={"type": "json_object"}` in the `completion()` call so the model is constrained to return valid JSON. Write a focused system prompt that describes the expected JSON shape (use the schema from Step 1) and specifies that each fact must be specific, verifiable, and accompanied by a context line. After parsing the response, call `ResearchNotes.model_validate(parsed)` to enforce the contract; raise a descriptive `ValueError` if validation fails.

The function signature should be:

```python
def researcher(topic: str, model: str = MODEL) -> tuple[ResearchNotes, dict]:
    ...
```

The returned `dict` carries `input_tokens`, `output_tokens`, and `cost` extracted from `response.usage` and `litellm.completion_cost()`.

Hint: the system prompt should include the literal JSON shape the model must return, not just a prose description — LLMs follow schema examples more reliably than prose alone.

## Step 3 — Implement the writer

The writer agent has two modes controlled by whether `critic_feedback` is provided:

- **Draft mode** (`critic_feedback=None`) — build the initial post from the research notes. Include the topic, audience, and `notes.model_dump_json(indent=2)` in the user message so the model has the full research context.
- **Revise mode** (`critic_feedback` is a `Critique`) — revise the prior draft to address every issue. Include the prior draft and the full critic feedback in the user message. The system prompt for this mode should instruct the model to preserve original strengths and only rewrite from scratch if the critic's `overall_comment` explicitly asks for it.

Both modes return a validated `BlogDraft` and a usage dict.

The function signature should be:

```python
def writer(
    topic: str,
    audience: str,
    notes: ResearchNotes,
    critic_feedback: Critique | None = None,
    prior_draft: BlogDraft | None = None,
    model: str = MODEL,
) -> tuple[BlogDraft, dict]:
    ...
```

Hint: keep the draft-mode and revise-mode system prompts separate (two constants) and select between them with a simple `if critic_feedback is None` branch.

## Step 4 — Implement the critic

The critic agent reviews a `BlogDraft` against the original `ResearchNotes` and returns a `Critique`. The system prompt should instruct the critic to evaluate four dimensions: clarity, accuracy (do claims match the notes?), engagement (tone right for the audience?), and structure (does the outline match the body?). Remind the critic that it should mark a verdict as `"approved"` only when there are no major issues left, and that on the first review of a new draft it should be thorough.

The function signature should be:

```python
def critic(
    draft: BlogDraft,
    notes: ResearchNotes,
    audience: str,
    model: str = MODEL,
) -> tuple[Critique, dict]:
    ...
```

Hint: passing `notes.model_dump_json(indent=2)` alongside the draft gives the critic the ground truth it needs to check accuracy without the orchestrator having to pass anything extra.

## Step 5 — Wire up the orchestrator

`run_pipeline` is the orchestrator. It calls the three agents in order and manages the revision loop:

1. Call `researcher` and print how many facts were gathered.
2. Call `writer` (draft mode) to produce the first draft.
3. Enter a loop up to `max_revisions` times:
   - Print the current draft (title and body).
   - Call `critic`; print the verdict, any issues, and the overall comment.
   - If the verdict is `"approved"`, set `stop_reason = "approved"` and break.
   - If this is the last allowed round, set `stop_reason = "max_revisions"` and break.
   - Otherwise call `writer` (revise mode) with the current critique and draft.
4. Return a result dict with `final_draft`, `notes`, `drafts`, `critiques`, `rounds_used`, `max_revisions`, `stop_reason`, per-agent token totals, and overall totals.

Track per-agent usage with a helper `accumulate(agent, usage)` that adds `input_tokens`, `output_tokens`, and `cost` into a running total dict for each of the three agents.

Hint: use `stop_reason` to distinguish a clean approval from hitting the revision ceiling — both are normal outcomes, and the caller should be able to tell them apart.

## Step 6 — CLI entry point

Wire up `argparse` with one positional argument and two flags:

```python
parser.add_argument("topic", help="The topic of the blog post")
parser.add_argument("--audience", default="general",
                    help="Target audience (default: general)")
parser.add_argument("--max-revisions", type=int, default=DEFAULT_MAX_REVISIONS,
                    help=f"Maximum writer/critic rounds (default {DEFAULT_MAX_REVISIONS})")
```

After `run_pipeline()` returns, print the summary section: rounds used, stop reason, and a per-agent token/cost breakdown followed by a TOTAL line.

## Running it

```bash
python solution.py "How transformer attention works" --audience developers
```

Expected console output (exact values vary):

```
=== Research ===
6 facts gathered about 'How transformer attention works'

=== Draft 1 ===
Title: Understanding Transformer Attention: A Developer's Guide
...

=== Critique 1 ===
Verdict: revise
Issues:
  [major] Introduction lacks a concrete motivation
    suggestion: Add a one-sentence problem statement before explaining the mechanism.
  [minor] Section 3 repeats the definition from Section 1
    suggestion: Remove the repeated sentence.
Comment: The draft is technically accurate but needs a stronger opening and tighter structure.

=== Draft 2 ===
Title: Understanding Transformer Attention: A Developer's Guide
...

=== Critique 2 ===
Verdict: approved
Comment: Well-structured, accurate, and appropriately detailed for a developer audience.

=== Summary ===
Rounds used: 2/3
Stop reason: approved

researcher  in=  312 out=  410 cost=$0.000123
writer      in= 1843 out= 1204 cost=$0.000487
critic      in= 2991 out=  318 cost=$0.000612
TOTAL       in= 5146 out= 1932 cost=$0.001222
```

The output is divided into four visible sections per round: Research (once), Draft N, Critique N, and a final Summary. Try `--max-revisions 1` to force an early stop and observe the `max_revisions` stop reason.

## What to try next

1. **Add a web-search tool to the researcher** — embed the Module 11 ReAct agent inside the researcher function so it fetches real facts from the web instead of relying on the model's parametric knowledge; this is a direct example of nested multi-agent composition
2. **Add more specialist agents** — insert a fact-checker agent after the writer that verifies every claim in the draft against an external source, or an SEO reviewer that evaluates keyword density and meta-description quality before the critic sees the post
3. **Tune critic strictness** — add a `--strict` flag that appends extra instructions to the critic prompt (e.g. "require at least two sources cited inline") so users can control how demanding the revision loop is
4. **Add caching** — store `ResearchNotes` results keyed on topic in a local JSON file so repeated runs on the same topic skip the researcher call entirely and go straight to drafting

# Module 12: Multi-Agent Systems

**What you'll learn:**
- The three foundational multi-agent patterns: orchestrator+specialists, handoff, and debate/critique
- How to write focused specialist prompts (single-responsibility for agents)
- Passing structured state between agents with Pydantic
- The generator/critic revision loop and how it converges
- Centralized vs decentralized coordination
- Cost and latency trade-offs of multi-agent pipelines
- Where multi-agent fits with tools (Module 06), RAG (Module 07), memory (Module 09), and single agents (Module 11)

| Detail        | Value                                                                 |
|---------------|-----------------------------------------------------------------------|
| Level         | Intermediate–Advanced                                                 |
| Time          | ~3.5 hours                                                            |
| Prerequisites | Module 06 (Tool Use & Function Calling), Module 08 (Structured Output), Module 11 (Building AI Agents) |

---

## Table of Contents

1. [Why Multi-Agent Systems](#1-why-multi-agent-systems)
2. [The Three Patterns](#2-the-three-patterns)
3. [Agent Roles and Prompts](#3-agent-roles-and-prompts)
4. [Communication: Passing State Between Agents](#4-communication-passing-state-between-agents)
5. [The Critique Loop](#5-the-critique-loop)
6. [Coordination and Control](#6-coordination-and-control)
7. [Cost and Latency Trade-offs](#7-cost-and-latency-trade-offs)
8. [Multi-Agent in the AI Stack](#8-multi-agent-in-the-ai-stack)

---

## 1. Why Multi-Agent Systems

A single agent running a ReAct loop — as built in [Module 11](../11-building-ai-agents/) — handles a wide range of tasks: open-ended research, code generation, multi-step data retrieval. But as tasks grow more complex, a single-agent design starts to crack. The cracks show in consistent ways: outputs become less focused, the system prompt grows unwieldy, debugging gets harder, and token costs rise faster than the usefulness does.

Multi-agent systems exist to address these cracks directly. Rather than one agent trying to do everything, you split the work across agents that each do one thing well.

### When one agent is not enough

**Conflicting instructions.** A single system prompt can only say so many things before those things start contradicting each other. Tell the same agent to write creatively and to follow strict factual sourcing rules, and you will get mediocre results on both dimensions. The agent is trying to satisfy two masters simultaneously. Separate the concerns, and each agent can fully commit to its role.

**Context pollution.** In a long agent run, the scratchpad fills with everything the agent has done — search results, intermediate reasoning, failed tool calls. By the time the agent writes its final output, that output sits in a context window saturated with unrelated noise. A fresh agent receiving only a clean, structured handoff produces better output than a tired agent trying to write after accumulating ten steps of irrelevant history.

**Generalist drift.** When one agent handles both research and writing, it tends to produce outputs that are neither great research nor great writing. The model is doing what it always does: averaging across competing objectives. Specialization forces each agent to excel at exactly one thing — which produces better outputs than averaging.

**Prompt bloat.** Adding more capabilities to a single agent means adding more instructions to its system prompt. Long system prompts are harder to maintain, harder to test, and empirically produce worse adherence than short, focused prompts. A five-hundred-word system prompt trying to cover research, writing, fact-checking, and formatting is less effective than three fifty-word prompts that each cover one task precisely.

### Specialization and separation of concerns

The core intuition behind multi-agent design is the same as the core intuition behind good software design: separate things that change for different reasons, and keep things that belong together close together.

A research agent and a writing agent change for different reasons. The research agent changes when you want to add new sources or improve retrieval quality. The writing agent changes when you want to adjust tone or format. If they are the same agent, every change risks affecting both behaviors. If they are separate agents, each can be tuned, tested, and replaced independently.

This independence is the main practical benefit of multi-agent design — not raw capability, but maintainability and precision of control.

### Why focused prompts outperform generalist prompts

When you give a model a narrowly scoped system prompt — one role, one output format, one set of constraints — it has very little ambiguity about what to do. The attention heads that matter for this task can work without competing for representation against instructions for unrelated tasks.

The effect is measurable in practice: an agent prompted only as "a careful fact-checker who identifies unsupported claims and returns a list of flagged sentences" will flag more genuine problems than an agent prompted as "a writer who should also check facts while writing."

### The cost of cramming everything into one agent

Besides prompt quality, there is a token accounting problem. A single-agent system that tries to research and write and critique in one run must keep all of that history in one context window. Every action the agent has taken is visible to every subsequent step. By the time writing begins, the agent has already accumulated thousands of tokens of research notes, tool calls, and intermediate reasoning — all of which compress the effective context available for generating high-quality output.

Multi-agent systems avoid this by keeping each stage's working memory separate. The research agent accumulates research history. The writing agent receives a clean summary. The critic receives only the draft. No agent carries the burden of another's history.

### How specialization changes model behavior in practice

Focused prompts produce measurably different behavior from generalist prompts, even when using the same underlying model. The mechanism is not fully understood, but the pattern is consistent: when the model's context contains a clear, single-purpose instruction, the model's output distribution narrows in the direction of that purpose.

A model told "you are a critic; identify the three most serious issues in this draft" produces a different distribution of outputs than the same model told "you are a writer; review your own draft and revise it." The first framing activates a critical, evaluative mode. The second framing is self-referential and tends to produce minor surface edits rather than substantive critique.

The practical implication: role framing is not cosmetic. When you write an agent's system prompt, the role declaration in the first sentence shapes everything that follows. Name the role precisely. It matters.

### Single-agent pain vs multi-agent fix

| Single-agent pain | Multi-agent fix |
|---|---|
| System prompt tries to satisfy two competing objectives (creative + factual) and ends up mediocre at both | Separate agents for each objective; each prompt fully commits to one role |
| Agent's scratchpad fills with research noise before writing begins; writing quality degrades | Writing agent receives a clean, structured handoff; no accumulated noise |
| Debugging requires tracing which part of a long prompt caused a bad output | Each agent's behavior is determined by its own short prompt; failures are localized |
| Adding a new capability means extending an already long system prompt and re-testing everything | New capability is a new agent with its own prompt; existing agents are unchanged |
| Context window fills up across a long multi-task run; later steps have less effective context | Each agent uses its own context window; handoffs are structured summaries, not raw history |

---

## 2. The Three Patterns

Multi-agent systems are not all alike. Three patterns cover the large majority of practical use cases. Each pattern has a different control structure, different failure modes, and different strengths. Knowing which one to reach for is the first design decision.

### Orchestrator + specialists

An orchestrator agent (or function) coordinates one or more specialist agents. The orchestrator receives the high-level task, decides which specialist to invoke and with what inputs, collects the specialist's output, and decides what to do next — either invoking another specialist, calling the same specialist again with refined inputs, or assembling a final result.

The orchestrator is the decision-maker. Specialists do not know about each other. They receive an input, produce an output, and return.

```
┌─────────────────────────────────────────────────────────┐
│               ORCHESTRATOR + SPECIALISTS                 │
│                                                         │
│   User task                                             │
│       │                                                 │
│       ▼                                                 │
│  ┌────────────┐                                         │
│  │ORCHESTRATOR│                                         │
│  └────────────┘                                         │
│       │          │          │                           │
│       ▼          ▼          ▼                           │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐                │
│  │Specialist│ │Specialist│ │Specialist│                │
│  │    A     │ │    B     │ │    C     │                │
│  └──────────┘ └──────────┘ └──────────┘                │
│       │          │          │                           │
│       └──────────┴──────────┘                           │
│                  │                                      │
│                  ▼                                      │
│           final result                                  │
└─────────────────────────────────────────────────────────┘
```

### Handoff

Each agent's output is the next agent's input. The pipeline is a chain: Agent A produces a result, Agent B receives that result as its input and produces another result, Agent C receives Agent B's result, and so on. Control passes linearly through the chain.

There is no central coordinator. Each agent knows only about its own input and output. The chain structure is defined externally — in the calling code — not by any agent.

```
┌─────────────────────────────────────────────────────────┐
│                        HANDOFF                          │
│                                                         │
│   User task                                             │
│       │                                                 │
│       ▼                                                 │
│  ┌─────────┐                                            │
│  │ Agent A │ ─── output A ──►                           │
│  └─────────┘                │                           │
│                             ▼                           │
│                        ┌─────────┐                     │
│             ◄─ output B─│ Agent B │                     │
│             │           └─────────┘                     │
│             ▼                                           │
│        ┌─────────┐                                      │
│        │ Agent C │ ─── final output                     │
│        └─────────┘                                      │
└─────────────────────────────────────────────────────────┘
```

### Debate / critique

A generator agent produces an output. A critic agent reviews that output and returns structured feedback. The generator receives the critique and revises. The cycle continues until the critic approves the output or a maximum round count is reached.

This is a loop, not a linear pipeline. Both agents operate on the same artifact — the draft — but from opposite roles: one produces, one evaluates.

```
┌─────────────────────────────────────────────────────────┐
│                   DEBATE / CRITIQUE                     │
│                                                         │
│   task                                                  │
│     │                                                   │
│     ▼                                                   │
│  ┌───────────┐                                          │
│  │ GENERATOR │◄────────────────────┐                    │
│  └───────────┘                     │                    │
│         │                          │                    │
│      draft                     revision                 │
│         │                      request                  │
│         ▼                          │                    │
│  ┌────────────┐    approved?        │                    │
│  │   CRITIC   │─────────────► final result              │
│  └────────────┘                    │                    │
│         │                          │                    │
│         └──── revise ──────────────┘                    │
└─────────────────────────────────────────────────────────┘
```

### Pattern comparison

| Pattern | When to use | Control | Strengths | Weaknesses |
|---|---|---|---|---|
| Orchestrator + specialists | Tasks with multiple parallel or sequential sub-tasks where the right sequence depends on intermediate results | Centralized; orchestrator decides what runs next | Easy to debug; easy to add specialists; orchestrator can adapt based on results | Orchestrator is a bottleneck; adding specialists increases orchestrator complexity |
| Handoff | Tasks with a well-defined linear pipeline where each stage transforms the previous stage's output | Decentralized; each agent sees only its own input | Simple to implement; each stage fully isolated; easy to test stages independently | No adaptation based on later stages; failures must be caught externally; rigid ordering |
| Debate / critique | Tasks where output quality matters more than speed; tasks where correctness is hard to verify in one pass | Looping; generator and critic alternate until convergence | Produces higher-quality output; self-correcting; critic provides explicit reasoning for changes | Costs multiply per revision round; may not converge; requires well-designed stop conditions |

---

## 3. Agent Roles and Prompts

The single most important design decision in a multi-agent system is not which framework to use or how to pass state between agents — it is how you write each agent's system prompt. A well-designed system prompt makes an agent reliable and predictable. A poorly designed prompt produces an agent that wanders, ignores constraints, and produces outputs in inconsistent formats.

### The single-responsibility principle for agents

Each agent should have exactly one role, produce exactly one type of output, and follow exactly one set of constraints. This is the single-responsibility principle applied to AI agents.

An agent with one responsibility is:

- **Easier to prompt** — you have one thing to describe clearly, not five
- **Easier to test** — you can evaluate it against a single quality criterion
- **Easier to replace** — if the agent underperforms, you swap its prompt or model without affecting anything else in the system
- **More reliable** — the model is not splitting attention across competing objectives

The failure mode of violating single responsibility is not usually a hard crash — it is subtle quality degradation. The agent does multiple things, but does each of them worse than a focused agent would.

### What a focused specialist prompt looks like

A focused specialist prompt has four components:

1. **Role declaration** — one sentence stating what the agent is and what domain it operates in
2. **Input description** — what the agent will receive and what format it will be in
3. **Output shape** — the exact format the agent must return, with no ambiguity
4. **Hard constraints** — specific rules the agent must follow, stated as positive instructions

Each component should be short and specific. If you find yourself writing more than two or three sentences for any component, the scope is too broad.

### Side-by-side example: weak vs focused prompt for a researcher agent

**Weak generalist prompt:**

```
You are a helpful research assistant. Given a topic, find relevant information and
provide a useful summary. You can also help with writing if needed, and make sure
the information is accurate and well-organized. Respond helpfully.
```

Problems: the role is vague ("helpful"), the scope is open-ended ("can also help with writing"), the output shape is undefined ("useful summary" could mean anything), and there are no hard constraints to enforce accuracy.

**Focused specialist prompt:**

```
You are a research agent. Your role is to gather factual information on a given topic.

Input: a topic string and an optional list of focus questions.

Output: a ResearchNotes object with the following fields:
- topic: the topic you researched
- facts: a list of ResearchFact objects, each containing a fact statement and a
  one-sentence context explaining why it is relevant

Rules:
- Return between 3 and 8 facts. Do not pad with obvious or trivial information.
- Each fact must be a specific, verifiable claim — not a vague generalization.
- Do not write prose summaries. Do not include opinions or recommendations.
- If you cannot find reliable information on a sub-question, omit it rather than
  speculating.
```

The focused prompt names the role, describes the input, specifies the exact output shape by name, and lists hard rules as positive instructions. An agent given this prompt has very little room to wander.

### Tips for writing specialist prompts

**Name the role.** "You are a research agent" is clearer than "You are a helpful assistant." The role name anchors the model's behavior pattern.

**Describe the input explicitly.** Tell the agent what it will receive and in what format. "Input: a topic string and an optional list of focus questions" is better than assuming the agent will figure it out.

**Describe the exact output shape.** Reference the Pydantic model by name if you are using structured output. This eliminates format ambiguity and makes validation automatic. "Output: a ResearchNotes object" is clearer than "Provide your research in a structured format."

**List hard rules.** Rules that state what to do ("return between 3 and 8 facts") are more effective than rules that state what not to do ("don't return too many facts"). State constraints as actions, not prohibitions, where possible.

**Keep it short.** A focused specialist prompt should fit in a paragraph. If yours has grown to a page, you are trying to solve too many problems with prompt engineering that should be solved by adding another agent.

### Complete system prompts for the three specialists in this module

The Blog Post Writer project uses three specialists. Their system prompts follow the pattern above — role, input, output shape, hard rules — and are short enough to fit in a screen.

**Researcher agent:**

```
You are a research agent. Your role is to gather factual information on a given topic.

Input: a topic string and an optional list of focus questions.

Output: a ResearchNotes object with:
- topic: the topic you researched
- facts: a list of ResearchFact objects, each with a fact statement and a one-sentence
  context explaining why it is relevant to the topic

Rules:
- Return between 3 and 8 facts. Do not pad with obvious or general background.
- Each fact must be a specific, verifiable claim — not a vague generalization.
- Do not write prose summaries. Do not include opinions or recommendations.
- If you cannot find reliable information on a sub-question, omit it rather than speculating.
```

**Writer agent:**

```
You are a blog post writer. Your role is to write a single, focused blog post from research notes.

Input: a ResearchNotes object containing a topic and a list of research facts.

Output: a BlogDraft object with:
- title: a clear, direct title (under 10 words)
- outline: a list of 3-5 section headings
- body: the full blog post body (400-600 words)

Rules:
- Ground every factual claim in the research notes you received. Do not add facts
  from outside the notes.
- Write in plain, direct prose. Avoid filler phrases and throat-clearing.
- The post must have an introduction, at least two body sections, and a conclusion.
- Do not include metadata, word counts, or any text outside the BlogDraft fields.
```

**Critic agent:**

```
You are a blog post critic. Your role is to evaluate a draft blog post and identify
the most important issues that need correction before publication.

Input: a BlogDraft object.

Output: a Critique object with:
- verdict: "approved" if the draft is ready to publish, "revise" if it needs changes
- issues: a list of CriticIssue objects (empty if verdict is "approved")
- overall_comment: one sentence summarizing the overall quality

Rules:
- Identify at most 3 issues per round. Focus on the most serious problems.
- Each issue must include a location (section or paragraph), the specific problem,
  and a concrete suggestion for how to fix it.
- Return verdict "approved" if the draft has no factual errors, is clearly structured,
  and makes no unsupported claims. Minor style preferences are not grounds for "revise".
- Do not suggest changes that would require research beyond what is in the draft.
```

Note how each prompt is roughly the same length and follows the same pattern. When a specialist underperforms, the source of the problem is almost always in one of the four components — and because the prompts are short, the issue is easy to find and fix.

---

## 4. Communication: Passing State Between Agents

Multi-agent systems live or die by the quality of their inter-agent communication. If agents pass raw strings, errors are silent, formats drift, and debugging is painful. If agents pass typed, validated objects, errors surface immediately, formats are enforced by the schema, and each agent's contract with the rest of the system is explicit.

### Function signature design

Each specialist should be implemented as a Python function with typed inputs and a typed return value. The function takes specific typed arguments — strings, integers, or Pydantic models — and returns a Pydantic model alongside a usage dict that records token consumption for cost tracking.

A consistent function signature pattern across all specialists makes the orchestrator's job simple: call a function, validate the return type, pass it to the next function.

```python
from pydantic import BaseModel
from typing import Literal

def run_researcher(topic: str, focus_questions: list[str] | None = None) -> tuple[ResearchNotes, dict]:
    """Call the research specialist. Returns structured notes and usage info."""
    ...

def run_writer(notes: ResearchNotes) -> tuple[BlogDraft, dict]:
    """Call the writing specialist. Takes research notes, returns a draft."""
    ...

def run_critic(draft: BlogDraft) -> tuple[Critique, dict]:
    """Call the critic specialist. Takes a draft, returns a structured critique."""
    ...
```

### Why typed boundaries beat raw strings

When agents pass raw strings, the receiving agent must parse the string, infer its structure, and hope the format matches its expectations. If the upstream agent changed its output format slightly, the downstream agent may silently misparse it — producing wrong output with no error.

Typed Pydantic boundaries eliminate this:

- **Parseability** — the model must produce output that conforms to the schema. If it does not, validation fails immediately with a clear error.
- **Versioning** — changing the schema between agents is explicit. You update the type, and every function that uses it shows a type error in your editor.
- **Debugging** — when something goes wrong, you can inspect the exact typed object at each stage. There is no ambiguity about what one agent passed to the next.

The cost is the overhead of defining Pydantic models. For any non-trivial multi-agent system, this overhead is worth paying.

### Worked example: types used in this module's project

The Blog Post Writer project — this module's primary project — chains a researcher, writer, and critic together. It uses the following types as the contract between agents:

```python
from pydantic import BaseModel
from typing import Literal


class ResearchFact(BaseModel):
    fact: str
    context: str


class ResearchNotes(BaseModel):
    topic: str
    facts: list[ResearchFact]


class BlogDraft(BaseModel):
    title: str
    outline: list[str]
    body: str


class CriticIssue(BaseModel):
    location: str          # e.g. "introduction" or "paragraph 2"
    issue: str             # what is wrong
    suggestion: str        # what to do instead


class Critique(BaseModel):
    verdict: Literal["approved", "revise"]
    issues: list[CriticIssue]
    overall_comment: str
```

`ResearchFact` is the atom — a single claim with its context. `ResearchNotes` is the researcher's full output: the topic and a list of facts. `BlogDraft` is the writer's output: a title, an outline, and the body text. `Critique` is the critic's output: a verdict, a list of specific issues with locations and suggestions, and an overall comment.

Each type is the complete, self-contained output of one agent. No agent receives more than it needs. The critic does not see the research notes — it sees only the draft. The writer does not see prior critique rounds directly — it sees only the current draft and the critique it is responding to.

This is information hiding at the agent level: each agent gets exactly the context it needs to do its job, and no more.

### A minimal specialist implementation

To make the types concrete, here is a complete implementation of the researcher specialist using `litellm.completion` with structured output. This follows the pattern that all three specialists in this module use.

```python
from litellm import completion
from pydantic import BaseModel


class ResearchFact(BaseModel):
    fact: str
    context: str


class ResearchNotes(BaseModel):
    topic: str
    facts: list[ResearchFact]


RESEARCHER_SYSTEM_PROMPT = """
You are a research agent. Gather factual information on the given topic.
Return a ResearchNotes object with 3-8 specific, verifiable facts.
Do not include opinions, vague generalizations, or speculation.
""".strip()


def run_researcher(
    topic: str,
    focus_questions: list[str] | None = None,
    model: str = "gpt-4o-mini",
) -> tuple[ResearchNotes, dict]:
    """Call the researcher specialist. Returns structured notes and token usage."""
    user_content = f"Topic: {topic}"
    if focus_questions:
        formatted = "\n".join(f"- {q}" for q in focus_questions)
        user_content += f"\n\nFocus questions:\n{formatted}"

    response = completion(
        model=model,
        messages=[
            {"role": "system", "content": RESEARCHER_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        response_format=ResearchNotes,
    )

    notes = ResearchNotes.model_validate_json(
        response.choices[0].message.content
    )
    usage = {
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
    }
    return notes, usage
```

The usage dict is the key addition beyond a basic LLM call. Tracking usage per specialist — not just per pipeline run — lets you see exactly where your token budget is going and which agents are the most expensive. In a three-agent pipeline with two critique rounds, the usage dicts tell you whether the researcher, the writer, or the critic is the cost driver.

### Connecting structured output to module 08

The pattern here is a direct application of the techniques from [Module 08](../08-structured-output/). The same `response_format` parameter and Pydantic model approach works for specialist agents just as it does for single LLM calls. The only difference is that the structured output type doubles as the inter-agent communication contract.

If you have not completed Module 08, read it before implementing multi-agent systems. Getting reliable structured output from a model is a prerequisite skill — without it, typed agent boundaries will fail intermittently on malformed output.

---

## 5. The Critique Loop

The critique loop is the engine behind the debate/critique pattern. It is where quality improvement actually happens. A generator produces a first pass. A critic evaluates it. The generator revises. The loop continues until the critic is satisfied or a maximum round count is reached.

### The generator/critic cycle

The cycle has four phases that repeat:

**Produce.** The generator agent receives the task and any prior critique (empty on the first round) and produces a draft. The draft is a structured object — for example, `BlogDraft` — not a freeform string.

**Critique.** The critic agent receives the draft and evaluates it against its criteria. It returns a `Critique` object with a verdict (`"approved"` or `"revise"`), a list of specific issues, and an overall comment.

**Decide.** The loop checks the verdict. If `"approved"`, the loop exits and the current draft is the final output. If `"revise"`, the loop continues.

**Revise.** The generator receives both the current draft and the critique. It addresses the issues identified by the critic and returns a revised draft. The cycle repeats from the critique step.

### What makes a useful critique

A critique that is too vague produces no improvement. "The article needs work" does not tell the generator what to fix. A critique that is too granular buries the important changes in noise.

Three qualities define a useful critique:

**Specific.** The issue should be locatable. "The third paragraph makes an unsupported claim that Python is faster than Java without a citation" is specific. "The article has factual issues" is not.

**Actionable.** The issue should come with a direction for improvement. "Add a citation for the performance claim, or remove the claim and replace it with a verifiable alternative" is actionable. "Fix the factual issues" is not.

**Bounded.** The critique should identify the most important issues, not every possible improvement. If the critic returns fifteen issues, the generator faces an overwhelming revision task and is likely to address some issues while inadvertently introducing new ones. Three to five focused issues per round produces better convergence than an exhaustive laundry list.

These qualities map directly to the `CriticIssue` fields: `location` enforces specificity, `suggestion` enforces actionability, and the list length should be bounded by the critic's system prompt ("identify the three most important issues").

### Three stop conditions

**Verdict approved.** The critic returns `verdict="approved"`. The current draft is the final output. This is the preferred exit path — the loop has converged on an output the critic considers acceptable.

**Max rounds reached.** A hard cap set by the caller, typically 3–5 rounds. If the loop reaches `max_rounds` without the critic approving, return the most recent draft and optionally include the most recent critique in the output metadata for debugging. This prevents unbounded token spend.

**No change between rounds.** If the draft produced in round N is identical to the draft in round N-1, the generator has stopped incorporating the critique — usually because it does not know how to address the issues or is ignoring them. Detect this and break the loop. Returning the draft at this point is preferable to continuing to accumulate identical rounds.

```python
MAX_ROUNDS = 4

def run_critique_loop(task: str) -> tuple[BlogDraft, list[Critique]]:
    draft, _ = run_writer_initial(task)
    critiques = []

    for round_num in range(MAX_ROUNDS):
        critique, _ = run_critic(draft)
        critiques.append(critique)

        if critique.verdict == "approved":
            break

        previous_body = draft.body
        draft, _ = run_writer_revise(draft, critique)

        # No-change detection
        if draft.body == previous_body:
            break

    return draft, critiques
```

### Convergence: why loops stabilize in 2–3 rounds

In practice, a well-designed critique loop stabilizes quickly. The first critique typically identifies structural issues — missing information, wrong tone, unsupported claims. The generator addresses these and the second draft is significantly better. The second critique identifies remaining issues — smaller in scope, easier to address. By the third round, most loops produce an approved verdict or are converging toward one.

Loops that do not converge typically have one of three causes: the generator and critic have conflicting objectives baked into their prompts (the generator is told to be brief, the critic is told to flag missing detail); the task is genuinely ambiguous and the critique changes meaning between rounds; or the generator model is not strong enough to address the critique reliably.

If a loop is not converging in 3 rounds during development, examine the prompts before raising `max_rounds`.

### ASCII trace: a 2-round revision

```
Round 1:
  Generator  →  draft_v1 (title: "Python vs Java Performance", 400 words)
  Critic     →  verdict: "revise"
                issues:
                  - location: "paragraph 2"
                    issue: "Claims Python is faster than Java — no citation"
                    suggestion: "Add benchmark source or qualify with use case"
                  - location: "conclusion"
                    issue: "Conclusion is abrupt and does not summarize key points"
                    suggestion: "Add 2-3 sentence summary of the main findings"
                overall_comment: "Good structure, needs factual support and a stronger conclusion."

Round 2:
  Generator  →  draft_v2 (title: "Python vs Java Performance", 480 words)
                (added benchmark citation; rewrote conclusion with summary)
  Critic     →  verdict: "approved"
                issues: []
                overall_comment: "Claims are now supported and conclusion is clear."

Loop exits after 2 rounds. draft_v2 is the final output.
```

---

## 6. Coordination and Control

How agents are coordinated is as important as what each agent does. The coordination structure determines where decisions are made, how the system responds to failures, and how easy the system is to debug and extend.

### Centralized coordination: the orchestrator decides

In the orchestrator pattern, one component — which may itself be an LLM agent or simply a Python function — holds all the coordination logic. It decides which specialist to call, in what order, with what inputs, and what to do with the outputs.

Specialists in a centralized system are stateless: they receive an input, produce an output, and return. They do not know who called them, what came before, or what will come after.

Centralized coordination has a clean advantage for debugging: the entire decision-making history lives in one place. When something goes wrong, you trace through the orchestrator's logic. There is one path to follow.

The downside is that the orchestrator becomes a bottleneck. As the system grows more complex, the orchestrator logic grows too. Adding a new specialist means updating the orchestrator to know about it. For very complex systems with many specialists and many conditional paths, the orchestrator can become the hardest part of the system to maintain.

### Decentralized coordination: handoff and chains

In the handoff pattern, there is no central coordinator. Each agent does its work and passes output to the next agent in the chain. The chain structure is defined at the call site — in the function that assembles the pipeline — not inside any agent.

Decentralized coordination is simpler to implement for linear pipelines. Each agent is fully isolated: it does not know about the existence of other agents. Testing is straightforward — feed an agent its expected input, check its output.

The downside is that decentralized systems are harder to adapt. If Agent B's output quality depends on what Agent A did, and Agent B cannot access that information, the handoff pattern loses context that might be important. There is also no single place where the overall system behavior is visible — understanding the full pipeline requires reading the code that assembles the chain.

### When to mix patterns

The Blog Post Writer project — this module's primary project — uses all three patterns together:

- The **orchestrator** (a Python function) coordinates the top-level flow: call the researcher, then call the writer, then enter the critique loop.
- A **handoff** carries the researcher's output into the writer's input: `ResearchNotes` flows directly from researcher to writer with no transformation.
- The **critique loop** runs internally between writer and critic: the writer generates, the critic evaluates, the writer revises.

This mixture is natural because the three patterns address different aspects of the same task: orchestration handles overall sequencing, handoff handles clean stage transitions, and the critique loop handles iterative quality improvement. None of the patterns conflicts with the others when each is applied to the part of the problem it fits best.

The rule of thumb: start with the simplest pattern that works. Handoff for linear transformations. Orchestrator when you need conditional logic or parallel specialists. Critique loop when output quality requires iteration.

### Observability in multi-agent systems

Single-agent observability (covered in [Module 11](../11-building-ai-agents/)) requires logging each step of a ReAct loop. Multi-agent observability requires logging at two levels: the inter-agent level (which specialist was called, with what inputs, and what it returned) and the intra-agent level (what the specialist did internally, if it runs its own loop).

At minimum, log the following at the inter-agent level:

```python
def log_agent_call(
    agent_name: str,
    input_summary: str,
    output_summary: str,
    usage: dict,
    latency_ms: float,
) -> None:
    print(f"\n=== {agent_name} ===")
    print(f"Input:   {input_summary[:120]}")
    print(f"Output:  {output_summary[:120]}")
    print(f"Tokens:  {usage['total_tokens']}  |  Latency: {latency_ms:.0f}ms")
```

Calling this function after each specialist returns gives you a trace of the full pipeline in the same way that step-level logging gives you a trace of a single agent's loop. When output quality is poor, the inter-agent trace tells you which handoff boundary is where the problem enters the pipeline.

### Centralized vs decentralized summary

| Dimension | Centralized (orchestrator) | Decentralized (handoff / chain) |
|---|---|---|
| Where decisions are made | Orchestrator | Call site / external assembly |
| Debugging | Trace the orchestrator | Trace the chain assembly |
| Adding a new specialist | Update the orchestrator | Add a new function and wire it in |
| Context sharing between stages | Orchestrator can pass context from any stage to any other | Each agent sees only its direct input |
| Best fit | Complex, conditional, non-linear tasks | Simple, linear, well-defined transformations |

---

## 7. Cost and Latency Trade-offs

Multi-agent systems are more expensive and slower than single-agent or single-call approaches. This is not a flaw — it is the cost of what they provide: specialization, isolation, and iterative improvement. Understanding the cost structure lets you make deliberate trade-offs rather than discovering them in production.

### Multi-agent multiplies token cost and latency

In a single-agent system, you make one LLM call (or a bounded number of calls in a ReAct loop, as covered in [Module 11](../11-building-ai-agents/)). In a multi-agent system, you make one call per agent per invocation. A three-agent pipeline makes at least three calls. A three-agent pipeline with two critique rounds makes at least five calls. Each call has a latency cost and a token cost, and they add up.

Token costs compound in two ways: volume (more calls) and context size (each call sends its own context). The research agent sends its full conversation history. The writing agent sends research notes plus its own system prompt. The critic sends the draft plus its evaluation criteria. None of these contexts are shared, so you pay to set up each agent separately.

### Worked cost comparison: single call vs three-agent pipeline

Consider writing a 500-word blog post on a given topic. Here is a rough token accounting for each approach:

**Single-call approach:** One call with a detailed prompt (~800 tokens) asking for research and writing together. Output (~700 tokens). Total: ~1,500 tokens.

**Three-agent pipeline with two critique rounds:**

| Step | Input tokens (approx.) | Output tokens (approx.) | Notes |
|---|---|---|---|
| Researcher agent call | 400 (system + topic) | 500 (research notes) | Focused research system prompt + topic input |
| Writer agent call | 700 (system + research notes) | 700 (draft v1) | Research notes passed as structured input |
| Critic agent call, round 1 | 900 (system + draft) | 300 (critique) | Full draft in input; structured critique as output |
| Writer revision call, round 1 | 1,100 (system + draft + critique) | 700 (draft v2) | Draft and critique both in input |
| Critic agent call, round 2 | 900 (system + draft v2) | 200 (approved) | Shorter verdict; no major issues |
| **Total** | **~4,000** | **~2,400** | **~6,400 tokens total** |

Compared to the single-call approach (~1,500 tokens), the three-agent pipeline with two critique rounds uses approximately 4× the tokens. If two critique rounds are required, you pay 4× for a better output. Whether that trade-off is worth it depends on the quality difference — for high-stakes content, it often is; for a routine task, it rarely is.

### When splitting hurts

Multi-agent is not always better. Three situations where splitting costs more than it buys:

**Cheap tasks.** If the task is simple enough that a single well-crafted prompt reliably produces acceptable output, splitting it across agents adds cost and latency for no quality gain. Summarizing a document, translating a paragraph, classifying a ticket — these do not benefit from multi-agent treatment.

**Latency-sensitive paths.** A three-agent sequential pipeline has minimum latency equal to the sum of all three agents' response times. If each call takes two seconds, the pipeline takes at least six seconds before accounting for any critique rounds. For any user-facing feature where response time matters, a single call will almost always produce a better user experience even if the output is marginally lower quality.

**Single-step problems.** If the task genuinely has one step — one decision, one transformation, one lookup — wrapping it in a multi-agent system adds architectural complexity with no benefit. Multi-agent is valuable when the task has genuinely separable sub-tasks that benefit from specialization. If it does not, use a single call.

### Cost estimation heuristic

A rough mental model for estimating multi-agent cost before building:

- Start with the estimated single-call token count for the same task
- Multiply by the number of agents in the pipeline (each adds setup overhead)
- Multiply by the expected number of critique rounds, if any
- Add 20% overhead for error handling retries and usage tracking

If the resulting estimate is acceptable, proceed. If not, consider whether the task actually requires multi-agent treatment or whether a simpler design would achieve 90% of the quality at a fraction of the cost.

---

## 8. Multi-Agent in the AI Stack

Multi-agent systems are built on top of every major technique covered in prior modules. Each prior module describes a capability that becomes a building block inside a specialist agent. Understanding these relationships makes it easier to design agents that use the right underlying technique for each part of their job.

### Relationship to Module 06: tools inside specialists

[Module 06](../06-tool-use-function-calling/) introduced function calling: how to give an LLM access to external tools and have it decide when and how to call them. A specialist agent can use tools internally without the orchestrator or other agents knowing about it.

A research specialist can call `web_search` and `fetch_page` tools as part of its work. The orchestrator sees only the final `ResearchNotes` output — not the five tool calls the researcher made to produce those notes. This is the right level of encapsulation: the specialist's implementation details are hidden behind its typed output.

This composability means you can upgrade a specialist's internal capabilities — adding new tools, improving retrieval — without changing the orchestrator or any other specialist. The interface is the Pydantic type. The implementation can change freely.

### Relationship to Module 07: RAG as a specialist capability

[Module 07](../07-rag/) built retrieval-augmented generation pipelines: embed a query, search a vector index, inject retrieved chunks into the prompt. In a single-call RAG setup, retrieval always happens on every call. In a multi-agent system, retrieval becomes a specialist capability that is invoked only when needed.

A "domain knowledge" specialist can have its own vector index — a knowledge base specific to its domain — and use RAG internally to ground its outputs. Other specialists do not share this index; they receive the specialist's output and use it without knowing how it was produced.

This is the "each specialist has its own knowledge base" pattern. A legal review specialist retrieves from legal document embeddings. A code review specialist retrieves from documentation and code style guides. Each specialist's retrieval is tuned to its role, rather than one shared retrieval system trying to serve all roles.

### Relationship to Module 09: shared memory across agents

[Module 09](../09-conversational-ai-memory/) established two kinds of memory: short-term (the messages list, which lives only for the current session) and long-term (facts stored externally, injected at session start). Both apply to multi-agent systems, with some additional complexity.

**Short-term memory** in a multi-agent system is per-agent by default — each specialist has its own messages list and does not see the history of other agents. The orchestrator can choose to pass selected pieces of prior agent history into a specialist's input, but this is explicit and controlled, not automatic.

**Long-term memory** can be shared across agents when appropriate. A shared user preferences store, for example, can be injected into multiple specialists' system prompts at startup. A research agent and a writing agent can both receive the same user's previously stated preferences, even though their other context is separate. The memory system from Module 09 applies without modification — the only decision is which agents receive which memories.

The budgeting discipline from Module 09 applies fully: every piece of context injected into a specialist costs tokens for that call. Track costs per agent, not just per pipeline run, to understand where your token budget is going.

### Relationship to Module 11: each specialist could be a mini-agent

[Module 11](../11-building-ai-agents/) built a single ReAct agent that loops over tool calls until it reaches a final answer. A specialist in a multi-agent system does not have to be a single LLM call — it can itself be a full ReAct agent that uses tools, runs a loop, and returns a structured result.

The research specialist is a natural candidate for this: rather than a single LLM call that has to produce research notes in one shot, it can be a small ReAct agent that searches the web, fetches pages, and iterates until it has gathered enough information. The orchestrator does not care — it calls `run_researcher()` and gets back `ResearchNotes`. What happens inside that function is encapsulated.

This composability — specialists that are themselves agents — is what "multi-agent systems" means at its most general. The patterns are recursive: an orchestrator agent can coordinate specialist agents, each of which runs its own tool loop. The depth of nesting is limited by token budget and latency, not by architecture.

When a specialist needs to make multiple tool calls to do its job reliably, wrapping it in a ReAct loop is the right design. When a single LLM call is sufficient, keep it simple.

### Forward pointer: Module 13 (workflows and chains)

**Module 13 (Workflows & Chains)** covers the deterministic sibling of multi-agent systems: static pipelines where every step is defined in advance and always runs in the same order. Where multi-agent systems decide at runtime what to do next, workflows decide at design time.

The choice between them comes down to task structure. Use multi-agent (this module) when the task has conditional paths, requires adaptive behavior based on intermediate results, or benefits from iterative quality improvement. Use workflows (Module 13) when the steps are fixed, the order is known, and reliability and predictability matter more than flexibility.

Understanding both patterns lets you choose the right architecture for each task rather than defaulting to one approach for everything.

### Module cross-reference map

| This module's component | Prior module it builds on |
|---|---|
| Tool use inside a specialist agent | [Module 06](../06-tool-use-function-calling/) — function calling and tool dispatch |
| Pydantic types as agent communication contracts | [Module 08](../08-structured-output/) — structured output and schema validation |
| RAG specialist with its own vector index | [Module 07](../07-rag/) — embedding, retrieval, and context injection |
| Shared long-term memory across agents | [Module 09](../09-conversational-ai-memory/) — short-term and long-term memory |
| Specialist implemented as a ReAct mini-agent | [Module 11](../11-building-ai-agents/) — the ReAct loop, stop conditions, and observability |

Each of these components works independently. Multi-agent systems compose them — putting each capability in the hands of a specialist that uses it well, coordinating the specialists into a pipeline that accomplishes tasks none of them could handle alone.

---

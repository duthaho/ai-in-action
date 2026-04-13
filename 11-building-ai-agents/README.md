# Module 11: Building AI Agents

**What you'll learn:**
- The core ReAct loop — observe, think, act, observe
- How an agent selects tools and builds up its scratchpad each turn
- Stop conditions, iteration limits, and loop detection
- Writing tool descriptions and system prompts that guide agent behavior
- Error handling and recovery when tools fail
- Observability: logging, traces, and debugging agents
- Where agents fit with tools (Module 06), RAG (Module 07), and memory (Module 09)

| Detail        | Value                                                                 |
|---------------|-----------------------------------------------------------------------|
| Level         | Intermediate                                                          |
| Time          | ~3 hours                                                              |
| Prerequisites | Module 04 (The AI API Layer), Module 06 (Tool Use & Function Calling), Module 09 (Conversational AI & Memory) |

---

## Table of Contents

1. [Why Agents](#1-why-agents)
2. [The ReAct Loop](#2-the-react-loop)
3. [Tool Selection and the Scratchpad](#3-tool-selection-and-the-scratchpad)
4. [Stop Conditions and Iteration Limits](#4-stop-conditions-and-iteration-limits)
5. [Agent Prompts: System Design](#5-agent-prompts-system-design)
6. [Error Handling and Recovery](#6-error-handling-and-recovery)
7. [Observability](#7-observability)
8. [Agents in the AI Stack](#8-agents-in-the-ai-stack)

---

## 1. Why Agents

A standard LLM call is a single transaction: you send a prompt, you receive a response, the interaction ends. This works well when the task is self-contained — summarize this document, answer this question, extract these fields. The model has all the information it needs, and one call is enough.

Agents exist for a different class of task: problems where the number of steps is unknown up front, where the model must act before it knows what to do next, and where the right path depends on what earlier steps discovered. You cannot plan these tasks in advance. You have to loop.

### Three hallmarks of agentic tasks

**Unknown number of steps.** A single-call task has a fixed shape. An agentic task does not — the model may need two tool calls or ten, depending on what it finds. Researching a question might require one web search or five, depending on the quality and completeness of the results.

**Branching on intermediate results.** Each step produces new information that changes what comes next. If a web search returns poor results, the agent adjusts its query. If a page fetch fails, the agent tries a different URL. The model observes what happened and decides the next action. This is fundamentally different from a fixed pipeline, which executes the same steps regardless of intermediate state.

**Tool-first workflows.** Agentic tasks depend on external capabilities: searching the web, reading files, querying APIs, executing code, writing to storage. The model is the orchestrator, but the work is done by tools. Without tool use (see [Module 06](../06-tool-use-function-calling/)), there is no agent — just a clever prompt.

### Good fit for an agent vs bad fit

| Good fit for an agent | Bad fit for an agent |
|---|---|
| Open-ended research that requires multiple searches and cross-referencing sources | Summarizing a single document provided in the prompt |
| Debugging a codebase by reading files, running tests, and iterating on fixes | Answering a factual question the model can answer from training |
| Filling out a multi-step form by reading requirements, validating inputs, and making corrections | Translating a paragraph of text into another language |
| Planning a trip by checking availability, comparing options, and resolving scheduling conflicts | Generating a list of ideas or alternatives in one shot |
| Monitoring a process and responding to changing conditions over time | Classifying an image or document with a known schema |
| Resolving a customer support ticket by querying order history, checking policies, and drafting a reply | Extracting structured fields from a known document format |

### When NOT to use an agent

Agents are powerful but they are not the right tool for every problem. Three situations where you should reach for something simpler:

**One-shot tasks.** If you can describe the entire task in a prompt and the model has all the information it needs to answer, a single LLM call is faster, cheaper, and less likely to go wrong. Agents introduce overhead — multiple API calls, retry logic, token accumulation — that is only justified when it is actually needed.

**Simple known workflows.** If you know exactly what steps the task requires and those steps always run in the same order, a fixed pipeline or chain (see Module 13) is more reliable than an agent. Agents introduce non-determinism. When the steps are fixed, that non-determinism is a liability, not an asset.

**Latency-sensitive paths.** Agents are slow. Each step requires a round trip to the LLM and a round trip to a tool. A five-step agent task might take 10–30 seconds even with fast models and fast tools. If your application requires a response in under two seconds, an agent is almost certainly the wrong architecture.

---

## 2. The ReAct Loop

ReAct stands for Reason + Act. The pattern was introduced to address a specific limitation of chain-of-thought prompting: the model reasons but cannot interact with the world. ReAct interleaves reasoning and action — the model thinks, acts, observes the result, and thinks again.

### The four phases

**Observe.** The model receives the current state of the world: the user's question, any prior tool results, and the conversation history accumulated so far. This is the raw input for the current step.

**Think.** The model reasons about what it has observed and decides what to do next. In some configurations this reasoning is explicit (the model writes out a "Thought:" before acting); in others it is implicit and the model moves directly to a tool call.

**Act.** The model selects a tool and calls it with specific arguments. The tool executes and returns a result — a web search returns snippets, a file read returns content, an API call returns JSON.

**Observe again.** The tool result becomes the new observation. It is appended to the conversation history (the scratchpad), and the loop begins again from the top. The model now has richer context and may make a different decision than it would have on the first step.

### The loop diagram

```
┌─────────────────────────────────────────────────────┐
│                    REACT LOOP                       │
│                                                     │
│   User question / task                              │
│           │                                         │
│           ▼                                         │
│     ┌──────────┐                                    │
│     │  OBSERVE │ ◄─────────────────────┐            │
│     └──────────┘                       │            │
│           │                            │            │
│           ▼                            │            │
│      ┌─────────┐                       │            │
│      │  THINK  │                       │            │
│      └─────────┘                       │            │
│           │                            │            │
│           ▼                            │            │
│       ┌────────┐       ┌──────────┐    │            │
│       │   ACT  │──────►│   TOOL   │    │            │
│       └────────┘       └──────────┘    │            │
│                              │         │            │
│                              ▼         │            │
│                       tool result ─────┘            │
│                              │                      │
│                    ┌─────────┘ stop condition met?  │
│                    │                                │
│                    ▼ yes                            │
│             final_answer()                          │
└─────────────────────────────────────────────────────┘
```

### Concrete walkthrough: "What is the latest stable Python version?"

The following shows how a research assistant agent works through a single question across multiple steps.

**Step 1 — Observation.**
The agent receives the user's question:

```
User: What is the latest stable Python version?
```

**Step 2 — Thought.**
The model reasons internally:

```
Thought: I need to check an authoritative source for the current stable Python release.
         My training data may be out of date for a question like this.
         I will use web_search to find the answer.
```

**Step 3 — Action.**
The model calls the web search tool:

```python
web_search(query="latest stable Python version 2026")
```

**Step 4 — Observation.**
The tool returns search results:

```
[Result 1] python.org/downloads — Python 3.13.2 is the latest stable release...
[Result 2] Wikipedia — Python 3.13, released October 2024...
[Result 3] Real Python — Python 3.13 new features...
```

**Step 5 — Next thought.**
The model reads the results and reasons:

```
Thought: The search results mention Python 3.13.2. I should verify this directly
         from python.org rather than relying on a snippet that might be stale.
```

**Step 6 — Next action.**
The model fetches the official download page:

```python
fetch_page(url="https://python.org/downloads")
```

**Step 7 — Observation.**
The page content confirms:

```
Python 3.13.2 — Released: February 4, 2025. This is the latest stable release.
```

**Step 8 — Final action.**
The model is confident in the answer and terminates the loop:

```python
finalize_answer(answer="The latest stable Python version is 3.13.2, released February 4, 2025.")
```

### How the loop terminates

The loop ends in one of three ways:

1. **`final_answer` tool** — the model calls a sentinel tool (often named `final_answer` or `finalize_answer`) whose purpose is to signal completion and return the answer to the caller. This is the preferred termination path: the model decides it has enough information.

2. **`max_steps` exceeded** — a hard cap set by the developer. If the model has not reached `final_answer` after N steps, the loop exits with whatever partial result is available. This prevents runaway agents and cost blowouts.

3. **Empty `tool_calls`** — some models return a plain text response instead of a tool call when they are done reasoning. The loop detects the absence of `tool_calls` in the response and treats the text as the final answer (degraded mode — see Section 6).

---

## 3. Tool Selection and the Scratchpad

### The scratchpad as a growing messages list

The agent's scratchpad is not a separate data structure — it is the standard `messages` list from [Module 09](../09-conversational-ai-memory/), growing with each step of the loop. The model maintains context across steps because each step's inputs and outputs are appended to the list before the next call.

At the start of the loop, the messages list contains the system prompt and the user's question. After each step, two messages are added: the model's tool call (role `assistant`) and the tool's result (role `tool`). By step 5, the list has grown from 2 messages to 12.

### How tool call results enter the messages list

When a tool is invoked, the result must be delivered back to the model in a format it understands. The OpenAI-compatible API uses a `role: "tool"` message paired with the `tool_call_id` from the model's request:

```python
# The model's tool call (role: assistant)
{"role": "assistant", "tool_calls": [{"id": "c1", "function": {"name": "web_search", "arguments": '{"query": "latest Python"}'}}]}

# The tool result delivered back (role: tool)
{"role": "tool", "tool_call_id": "c1", "content": "[5 search results here]"}
```

The model reads the `tool_call_id` to match the result to its request. Without the correct ID, the model has no way to know which tool call the result belongs to.

### Worked example: messages list after 3 steps

```python
messages = [
    {"role": "system", "content": "You are a research assistant..."},
    {"role": "user", "content": "What is the latest Python version?"},
    {"role": "assistant", "tool_calls": [{"id": "c1", "function": {"name": "web_search", "arguments": '{"query": "latest Python"}'}}]},
    {"role": "tool", "tool_call_id": "c1", "content": "[5 search results]"},
    {"role": "assistant", "tool_calls": [{"id": "c2", "function": {"name": "fetch_page", "arguments": '{"url": "https://python.org/downloads"}'}}]},
    {"role": "tool", "tool_call_id": "c2", "content": "Python 3.13.2..."},
]
```

After this third step (two tool calls so far), the list has 6 messages. A final `final_answer` call would add one more assistant message and terminate the loop, bringing the total to 7.

### Token cost grows linearly with step count

Every message added to the scratchpad is sent with every subsequent API call. Step 1 sends 2 messages. Step 2 sends 4. Step 5 sends 10. If each step adds ~500 tokens of content, a 10-step agent task sends roughly 27,500 tokens across all calls combined — versus 500 tokens for a single-call approach.

The table below illustrates the cumulative token cost for an agent where each step adds 500 tokens to the scratchpad (tool call + result combined):

| Step | New tokens this step | Total tokens sent this call | Cumulative tokens across all calls |
|---|---|---|---|
| 1 | 500 | 600 (system + user + step 1) | 600 |
| 2 | 500 | 1,100 | 1,700 |
| 5 | 500 | 2,600 | 8,000 |
| 10 | 500 | 5,100 | 27,500 |
| 20 | 500 | 10,100 | 105,000 |

By step 20, each call sends 10,000 tokens of accumulated scratchpad before generating a single output token. This is why agents should terminate early when possible.

This is the fundamental cost of iterative reasoning. The scratchpad is the agent's working memory, and you pay to keep it in context on every turn. The same budgeting disciplines from [Module 09](../09-conversational-ai-memory/) apply directly: track token counts, set soft and hard limits, and consider summarizing the scratchpad for very long tasks.

Practical implication: a 20-step agent task with verbose tool results can easily cost 10–50× more than a 5-step task. Be deliberate about what tool results you include verbatim versus summarize before appending.

### Controlling scratchpad size

Two tactics keep scratchpad growth under control without losing important context:

**Truncate large tool results before appending.** If a tool returns 10,000 characters of HTML, you do not need to put all 10,000 characters into the scratchpad. Extract the relevant portion (the text content, a specific section, the first N characters) and append that instead. The agent retains the key information without carrying the full payload through every subsequent call.

**Summarize completed sub-tasks.** If the agent has resolved a sub-task (confirmed a fact, fetched and parsed a document), you can replace the raw tool call and result messages for that sub-task with a single summary message. This is the same summarization pattern from [Module 09](../09-conversational-ai-memory/) applied to agent history rather than chat history.

---

## 4. Stop Conditions and Iteration Limits

Without explicit stop conditions, an agent loop will run indefinitely — burning tokens, incurring cost, and potentially making mistakes at each additional step. Every production agent needs at least two stop conditions: a natural termination signal and a hard safety limit.

### The `final_answer` tool pattern

The cleanest termination mechanism is a sentinel tool. Define a tool named `final_answer` (or similar) with a single `answer` argument. When the model calls this tool, the loop exits and returns the answer to the caller.

This works because the model is instructed to call `final_answer` when it has enough information to respond confidently. The loop dispatcher simply checks: if the tool name is `final_answer`, stop. Otherwise, execute the tool and continue.

The key advantage of the sentinel tool over other termination mechanisms is that the model controls when to stop. It can take exactly as many steps as needed — not always five, not capped at three — and signal completion when the task is genuinely done.

### `max_steps` as a hard safety net

Set a maximum step count regardless of whether the model has called `final_answer`. If the loop reaches `max_steps` without terminating naturally, it exits and returns the best partial answer available (typically the last assistant message, or an explicit fallback string).

Choose `max_steps` conservatively during development (5–10) and raise it only once you have observed your agent's actual behavior on representative tasks. Most well-designed agents for well-scoped tasks complete in fewer than 10 steps.

The implementation is straightforward — a counter incremented at each iteration:

```python
MAX_STEPS = 10

def run_agent(messages: list[dict], tools: dict) -> str:
    last_call = None
    for step in range(MAX_STEPS):
        response = call_llm(messages, tools=list(tools.values()))
        tool_calls = response.choices[0].message.tool_calls

        # Empty tool calls: degraded final answer
        if not tool_calls:
            return response.choices[0].message.content or "No answer produced."

        call = tool_calls[0]
        name = call.function.name
        args = call.function.arguments

        # Sentinel: natural termination
        if name == "final_answer":
            import json
            return json.loads(args).get("answer", "")

        # Loop detection
        if last_call and last_call == (name, args):
            return f"Agent stuck: repeated call to {name} with same arguments."
        last_call = (name, args)

        # Dispatch tool and append result
        result = dispatch_tool(name, args, tools)
        messages.append({"role": "assistant", "tool_calls": [call]})
        messages.append({"role": "tool", "tool_call_id": call.id, "content": result})

    return "Agent did not reach a conclusion within the step limit."
```

### Loop detection: same tool, same arguments

A subtle failure mode is an agent that calls the same tool with the same arguments in consecutive steps — usually because it got a result it did not like and is retrying the same action rather than adapting. This wastes tokens and rarely produces a different outcome.

Detect this by tracking the (tool name, arguments) pair at each step. If the current call matches the previous call exactly, break the loop immediately and treat the situation as a failure state.

### Budget caps: token threshold

Agents can overspend dramatically if a tool returns a very large result and the model then reasons extensively about it. Add a token budget check after each step: if the total token count of the scratchpad exceeds a threshold, force termination before the next call.

### Stop condition reference table

| Stop condition | When it triggers | What to return |
|---|---|---|
| `final_answer` tool called | Model calls the sentinel tool with a completed answer | The `answer` argument from the tool call |
| `max_steps` exceeded | Step counter reaches the configured limit without natural termination | Last assistant message, or a fallback string indicating the task timed out |
| Loop detected (same tool + same args) | Current tool call matches the previous tool call exactly | Error message indicating the agent got stuck; optionally include partial reasoning |
| Token budget exceeded | Scratchpad token count crosses the configured threshold | Partial result with a note that the response was truncated due to length |
| Tool call absent (empty `tool_calls`) | Model returns plain text with no tool calls — it decided it was done | Treat the plain text as the final answer (degraded mode) |

---

## 5. Agent Prompts: System Design

The system prompt is where you define what the agent is, what it can do, and how it should behave. An agent with a weak system prompt wanders — it misuses tools, makes up answers, and does not know when to stop. An agent with a strong system prompt stays focused, uses tools correctly, and reasons transparently.

### Framing: give the agent a clear role and scope

The system prompt should establish three things in the first paragraph: the agent's role, the scope of tasks it handles, and the scope it does not handle. Vague framing produces vague behavior.

Weak framing:

```
You are a helpful assistant. Use the available tools to answer questions.
```

Strong framing:

```
You are a research assistant that answers factual questions by searching the web and
reading authoritative sources. You handle questions about current events, technology,
science, and business. You do not answer questions that require personal opinions,
legal advice, or medical diagnoses — for those, tell the user that the question is
outside your scope.
```

The strong version tells the agent who it is, what it handles, and what it does not. These boundaries prevent the agent from attempting tasks it cannot do reliably.

### Tool descriptions: good vs bad

Tool descriptions are read by the model when deciding which tool to use and how to use it. A bad description leaves the model guessing. A good description tells the model exactly when to use the tool, what arguments it expects, and what it returns.

**Bad tool description:**

```python
{
    "name": "web_search",
    "description": "Search the web.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string"}
        }
    }
}
```

**Good tool description:**

```python
{
    "name": "web_search",
    "description": (
        "Search the web for current information. Use this when the answer requires "
        "up-to-date facts that may have changed since your training data, or when "
        "you need to verify a specific claim against an authoritative source. "
        "Returns a list of result snippets with titles and URLs. "
        "Follow up with fetch_page to read the full content of a specific result."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "A concise, specific search query. Use keywords rather than full sentences."
            }
        },
        "required": ["query"]
    }
}
```

The good description tells the model: when to use this tool, what it returns, and how to use the result. The bad description provides none of that context.

### Reasoning format: explicit vs implicit

You can instruct the model to reason explicitly before every tool call, or you can let it reason implicitly and jump straight to the action. Explicit reasoning is useful during development — you can read the model's thoughts in the trace and understand why it made each decision. Implicit reasoning is faster and produces cleaner output in production.

To request explicit reasoning, add this to the system prompt:

```
Before calling any tool, write a brief "Thought:" explaining why you are calling it
and what you expect it to return. After receiving the result, write a "Observation:"
summarizing what you learned and whether it changes your plan.
```

### Weak vs strong system prompt comparison

| Weak system prompt | Strong system prompt |
|---|---|
| "You are a helpful assistant." | "You are a research assistant that answers factual questions about technology, science, and business." |
| "Use tools when needed." | "Always verify answers using at least one tool call before responding. Do not answer from memory alone." |
| No guidance on format | "Cite your sources: include the URL for every factual claim in your final answer." |
| No stopping guidance | "Call `final_answer` as soon as you have enough information to answer confidently. Do not keep searching once you have a reliable answer." |
| No fabrication constraint | "Do not fabricate URLs, publication dates, or statistics. If you cannot find a source, say so." |

### Constraint phrasing

Constraints are most effective when they are specific and positive (what to do) rather than purely negative (what not to do):

- Instead of: "Do not make things up" — use: "Only include facts you found in a tool result during this session."
- Instead of: "Do not use bad sources" — use: "Prefer results from .gov, .edu, or established news organizations."
- Instead of: "Don't be verbose" — use: "Keep final answers under 200 words unless a longer explanation is explicitly requested."

### Putting it together: a complete system prompt example

```
You are a research assistant that answers factual questions by searching the web
and reading authoritative sources.

Scope: technology, science, business, and current events.
Out of scope: personal opinions, legal advice, medical diagnoses. If asked,
tell the user these topics are outside your scope.

Tools available:
- web_search: use for any question that may have changed since your training data.
- fetch_page: use to read the full content of a specific URL from web_search results.
- final_answer: call this when you have a confident, sourced answer to return.

Rules:
- Always verify answers with at least one tool call. Do not answer from memory alone.
- Do not fabricate URLs, publication dates, or statistics.
- Cite your source: include the URL in your final answer.
- Call final_answer as soon as you have a reliable answer. Do not keep searching
  once the question is resolved.

Reasoning: Before each tool call, write a one-sentence "Thought:" explaining why
you are making this call and what you expect it to return.
```

This prompt covers role, scope, tool guidance, constraints, and reasoning format — the five elements of a complete agent system prompt.

---

## 6. Error Handling and Recovery

Every tool call can fail. Networks time out. APIs return errors. The model generates malformed arguments. An agent with no error handling will crash on the first exception and return nothing useful. An agent with good error handling recovers gracefully and gives the model a chance to adapt.

### Tool exceptions: catch and return as a result

The correct pattern for tool exceptions is to catch them in the dispatcher, format the error as a string, and deliver it back to the model as a tool result — not to let the exception propagate. The model can then read the error and decide what to do next (retry with different arguments, try a different tool, acknowledge the failure in its final answer).

```python
def dispatch_tool(tool_name: str, arguments: dict, tools: dict) -> str:
    """Call the named tool with arguments. Return the result or an error string."""
    if tool_name not in tools:
        return f"Error: tool '{tool_name}' does not exist. Available tools: {list(tools.keys())}"

    try:
        result = tools[tool_name](**arguments)
        return str(result)
    except TypeError as e:
        # Malformed arguments from the model
        return f"Error: invalid arguments for '{tool_name}': {e}. Check the tool schema and retry."
    except Exception as e:
        # Any other tool-level failure
        return f"Error: '{tool_name}' failed with: {type(e).__name__}: {e}"
```

By returning the error as the tool result, you keep the agent loop intact. The model receives the error as an observation and can reason about it rather than being interrupted by an uncaught exception.

### Malformed arguments from the model

Even with a well-written tool description, models occasionally generate arguments that do not match the schema — wrong types, missing required fields, extra fields. Catch `TypeError` and `KeyError` specifically when calling the tool function, and return a descriptive error that tells the model what went wrong and how to fix it.

If you can parse the arguments partially (for example, the model passed a number as a string), coerce the type and proceed rather than failing. Only return an error if you cannot make reasonable sense of what the model intended.

### Empty tool calls: degraded final answer

Some models return a plain text response with no `tool_calls` field — either because they decided they had enough information, or because something in the prompt or context caused them to skip the tool-calling mode. Do not treat this as an exception. Treat it as a degraded form of `final_answer`: extract the text content and return it as the result.

Log that this happened — it usually signals a prompt design issue that should be investigated.

### Infinite-loop protection

When the model calls the same tool with the same arguments twice in a row, it is stuck. Insert a check after each step:

```python
def is_duplicate_call(current: dict, previous: dict | None) -> bool:
    if previous is None:
        return False
    return (
        current.get("name") == previous.get("name")
        and current.get("arguments") == previous.get("arguments")
    )
```

If this returns `True`, break the loop immediately and return an error to the caller.

### When to retry vs when to give up

Not all errors are worth retrying. Use this heuristic:

- **Retry**: transient errors (network timeout, rate limit, HTTP 503). Retry once with exponential backoff, then give up if it fails again.
- **Do not retry**: logic errors (invalid arguments, tool not found, schema violation). Return the error to the model as a result and let the model decide how to proceed.
- **Abort immediately**: budget exhaustion, `max_steps` exceeded, infinite loop detected. These are framework-level conditions — the agent has failed to converge and further iteration will not help.

### Error handling in practice

A well-structured agent dispatcher handles all these cases in one place, so the main agent loop stays clean:

```python
import time

def dispatch_with_retry(tool_name: str, arguments: dict, tools: dict, max_retries: int = 1) -> str:
    """Dispatch a tool call with one retry for transient errors."""
    for attempt in range(max_retries + 1):
        try:
            result = tools[tool_name](**arguments)
            return str(result)
        except ConnectionError as e:
            if attempt < max_retries:
                time.sleep(2 ** attempt)  # exponential backoff: 1s, 2s, ...
                continue
            return f"Error: tool '{tool_name}' failed after {max_retries + 1} attempts: {e}"
        except TypeError as e:
            # Argument mismatch — no point retrying
            return f"Error: invalid arguments for '{tool_name}': {e}"
        except Exception as e:
            return f"Error: '{tool_name}' failed: {type(e).__name__}: {e}"
```

Keeping retry logic in the dispatcher — not the main loop — means the agent loop only needs to call one function and check one return value. The complexity is hidden where it belongs.

---

## 7. Observability

An agent that runs silently is an agent you cannot debug. During development, you need to see every step — what the model thought, which tool it called, what the tool returned, and how long each step took. Without this visibility, a malfunctioning agent is a black box.

### Why you must log every step

A single LLM call is easy to debug: you can print the prompt and the response. An agent is harder — the relevant information is distributed across multiple calls, and a failure on step 7 may have been caused by a bad result on step 3. A complete step-by-step trace is the only way to reconstruct what happened.

Log everything during development. You can always filter or reduce the logging level in production once the agent is working correctly.

### What to log at each step

| Field | What it captures |
|---|---|
| Step number | Which iteration of the loop this is |
| Thought | The model's reasoning before the tool call (if explicit reasoning is enabled) |
| Tool name | Which tool the model selected |
| Arguments | The exact arguments passed to the tool |
| Result summary | The first 200 characters of the tool result (truncate to avoid log bloat) |
| Token count | Total tokens in the scratchpad at this step |
| Step latency | Wall-clock time for this step (LLM call + tool call) |

### Structured traces vs printed traces

A **structured trace** is a list of dicts, one per step, that you can serialize to JSON and analyze programmatically. This is the right format for production systems — you can search traces, compute statistics, and feed them into monitoring dashboards.

```python
trace = []  # accumulated across the loop

trace.append({
    "step": step_number,
    "tool": tool_name,
    "arguments": arguments,
    "result_preview": result[:200],
    "token_count": current_token_count,
    "latency_ms": elapsed_ms,
})
```

A **printed trace** writes each step to stdout as it happens. This is the right format for development — you see the agent's reasoning in real time and can interrupt it if it goes wrong.

### Verbose mode

During debugging, add a verbose flag that prints the full tool arguments and full tool result rather than a truncated preview. This reveals issues that the summary hides — for example, an argument that looks correct in the first 200 characters but has a subtle error further in.

```python
def log_step(step: int, tool: str, args: dict, result: str, verbose: bool = False) -> None:
    print(f"\n--- Step {step} ---")
    print(f"Tool: {tool}")
    if verbose:
        print(f"Args: {args}")
        print(f"Result:\n{result}")
    else:
        print(f"Args: {str(args)[:120]}")
        print(f"Result preview: {result[:200]}")
```

### Reading a trace to diagnose issues

When an agent produces a wrong answer or fails to terminate, work through the trace step by step:

1. Find the step where the agent first had enough information to answer correctly — did it recognize this?
2. Find steps where the model called the wrong tool or passed bad arguments — look at the tool descriptions and system prompt.
3. Find steps where a tool returned an error — did the model handle the error gracefully or repeat the same mistake?
4. Look at token counts — did the scratchpad grow unusually fast? Did the agent hit a budget limit before finishing?

### Example trace output

```
=== Agent trace: "What is the latest stable Python version?" ===

--- Step 1 ---
Tool: web_search
Args: {"query": "latest stable Python version 2026"}
Result preview: [1] python.org/downloads — Python 3.13.2 is the latest stable... [2] Wik...
Tokens: 312  |  Latency: 1.2s

--- Step 2 ---
Tool: fetch_page
Args: {"url": "https://www.python.org/downloads/"}
Result preview: Python 3.13.2 Released: February 4, 2025. Python 3.13.2 is the latest s...
Tokens: 891  |  Latency: 2.4s

--- Step 3 ---
Tool: final_answer
Args: {"answer": "The latest stable Python version is 3.13.2, released February 4, 2025."}
Result preview: [sentinel — loop terminating]
Tokens: 1024  |  Latency: 0.9s

=== Done in 3 steps, 4.5s total ===
```

The trace shows exactly what the agent did: one search, one page fetch, one final answer. If step 3 had been another `web_search` instead of `final_answer`, that would be visible here, pointing to a prompt issue with the stopping condition.

---

## 8. Agents in the AI Stack

Agents are not a replacement for the techniques in prior modules — they are an orchestration layer built on top of them. Every component an agent depends on is covered in an earlier module. Understanding how agents relate to those components makes it easier to design and debug agent systems.

### Relationship to Module 06: tool use as the foundation

[Module 06](../06-tool-use-function-calling/) introduced the mechanics of function calling: how to define tools, how the model selects and calls them, and how to deliver results back. Agents are "tool use plus a loop." The tool calling mechanics are identical. The only addition is the loop: instead of calling one tool and returning, you call a tool, observe the result, and call the next tool.

Everything in Module 06 applies directly — tool schema design, argument validation, result formatting, handling multi-tool calls in a single step. An agent dispatcher is exactly a function-calling router running inside a while loop.

### Relationship to Module 07: retrieval as a tool

[Module 07](../07-rag/) built retrieval-augmented generation pipelines: embed queries, search a vector index, inject retrieved chunks into the prompt. In a standalone RAG pipeline, retrieval always happens — it is a fixed step in a fixed sequence.

In an agent, retrieval becomes optional and conditional. The agent decides when to call a `search_knowledge_base` tool, with what query, and whether the results are sufficient or whether another search is needed. This is sometimes called the "retrieval as a tool" pattern, and it is strictly more flexible than a fixed RAG pipeline.

The agent can also chain retrieval with other tools: search the knowledge base, fetch a specific document, extract a structured fact, and verify it against a web source — all in one agent run. The RAG components (embedding, vector search, chunk retrieval) are unchanged; what changes is that the agent orchestrates when and how they are called.

### Relationship to Module 09: short-term and long-term memory

[Module 09](../09-conversational-ai-memory/) established the messages list as the agent's short-term memory. Every observation, tool call, and tool result in the current session lives in the messages list. The same budgeting, truncation, and summarization strategies apply to agent scratchpads — often with more urgency, because agent conversations grow faster than human conversations (each step adds two messages).

Long-term memory — facts persisted across sessions — is the agent's way of learning over time. After completing a task, an agent can extract facts worth remembering and store them in a user memory store. The next time the agent runs for the same user, those facts are injected into the system prompt, giving the agent context it would otherwise lack.

The distinction is:
- **Scratchpad (short-term)** — the current session's messages list; lost when the session ends
- **Long-term memory** — facts stored in external storage; injected at session start; survives across sessions

### When NOT to use an agent

Even if an agent could handle a task, it is not always the right choice:

- **Known workflows with fixed steps** — use a chain or pipeline (Module 13). Chains are faster, cheaper, and more predictable when the sequence of steps does not depend on intermediate results.
- **Simple lookups** — a single RAG call or a single API call is faster and cheaper than wrapping it in an agent loop.
- **Latency-sensitive paths** — interactive features that need a response in under two seconds cannot tolerate agent latency. Use a single call with a strong prompt instead.
- **High-stakes decisions without human review** — agents operating autonomously can make mistakes across multiple steps before anyone notices. For financial transactions, medical recommendations, or legal actions, add a human-in-the-loop checkpoint rather than letting the agent act autonomously.

### Latency and cost profile

| Agent configuration | Typical latency | Cost driver |
|---|---|---|
| 3-step agent, fast tools | 4–8 seconds | 3 LLM calls; token cost grows per step |
| 5-step agent, external API tools | 10–20 seconds | 5 LLM calls + 5 API round trips |
| 10-step agent, verbose results | 30–60 seconds | Scratchpad grows; each LLM call sends more tokens |
| 20-step agent, large tool results | 2–5 minutes | Token costs compound; risk of hitting context limit |

Agent tasks are fundamentally slower and more expensive than single-call tasks. This is acceptable when the task genuinely requires iteration. It is a poor trade when the task could be handled by a two-stage pipeline or a single well-crafted prompt.

### Forward pointer: Module 12 and Module 13

**Module 12 (Multi-Agent Systems)** extends the single-agent pattern to networks of agents: an orchestrator agent that breaks tasks into subtasks, and specialized worker agents that handle each subtask in parallel. The same ReAct loop runs inside each agent; the new complexity is inter-agent communication and result aggregation.

**Module 13 (Workflows & Chains)** covers the alternative to agents: deterministic pipelines where steps are fixed in advance. Knowing both patterns lets you choose the right architecture — agents for exploratory, open-ended tasks; chains for structured, known-path tasks.

---

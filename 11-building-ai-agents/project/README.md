# Project: Research Assistant

Build a CLI Research Assistant that uses a ReAct agent with three tools — web search, page fetch, and final answer — to iteratively research any question and return a sourced, verified answer.

## What you'll build

A command-line Research Assistant powered by a ReAct (Reason + Act) agent. Given a question, the agent calls `web_search` to find relevant pages, calls `fetch_page` to read their content, and repeats until it has enough information to call `finalize_answer` with a clear answer and the URLs it actually used. The agent loop runs inside a single function that appends each assistant message and tool result back into the message history, giving the model the full context of every step it has taken. You will learn how to define tool schemas for LiteLLM, dispatch tool calls by name, detect and break infinite loops, enforce a step budget, and report token usage and cost at the end of every run.

## Prerequisites

- Completed reading the Module 11 README
- Python 3.11+ with project dependencies installed (`pip install -r requirements.txt` from the repo root — this includes `duckduckgo-search` and `beautifulsoup4`)
- An OpenAI-compatible API key set in `.env` at the repo root (`OPENAI_API_KEY` or whichever provider you use)

## Setup

```bash
cd 11-building-ai-agents/project
```

Confirm that `.env` at the repo root contains your API key. The script loads it automatically via `python-dotenv` using a path resolved relative to the script file, so you do not need to be in any particular directory when you run it.

## Step 1 — Define the tools

The agent has three tools:

- **`web_search(query)`** — searches DuckDuckGo and returns up to five results, each with a title, URL, and snippet. The agent uses this to discover candidate sources.
- **`fetch_page(url)`** — downloads a URL, strips markup, collapses whitespace, and returns the first 2 000 characters of readable text. The agent uses this to verify claims in a result it found.
- **`finalize_answer(answer, sources)`** — a sentinel tool. When the agent calls it, the loop stops and the answer is returned to the user. Calling a dedicated tool (rather than relying on the model stopping on its own) makes termination explicit and easy to detect in code.

Each tool must be described in the `TOOLS` list using LiteLLM's tool schema format — a list of dicts with `"type": "function"` and a nested `"function"` key that carries `name`, `description`, and a JSON Schema `parameters` object. LiteLLM passes this list to the underlying provider's `tools` parameter.

```python
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web using DuckDuckGo ...",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", ...}},
                "required": ["query"],
            },
        },
    },
    ...
]
```

## Step 2 — Implement `web_search`

Use the `DDGS` context manager from `duckduckgo_search`. Call `.text(query, max_results=5)` and convert the results to a uniform list of dicts:

```python
def web_search(query: str) -> list[dict]:
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=MAX_SEARCH_RESULTS))
    return [
        {
            "title": r.get("title", ""),
            "url": r.get("href", ""),
            "snippet": r.get("body", ""),
        }
        for r in results
    ]
```

The agent receives the JSON-serialised list as the tool result and reads the snippets to decide which URLs are worth fetching.

## Step 3 — Implement `fetch_page`

Send a `GET` request with a descriptive `User-Agent` header (some sites block the default Python user agent). Parse the response with BeautifulSoup, remove `<script>`, `<style>`, and `<noscript>` tags, extract plain text, collapse whitespace with `re.sub`, then truncate:

```python
def fetch_page(url: str) -> str:
    response = requests.get(url, timeout=15, headers={"User-Agent": USER_AGENT})
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = re.sub(r"\s+", " ", soup.get_text(separator=" ")).strip()
    if len(text) > MAX_PAGE_CHARS:
        text = text[:MAX_PAGE_CHARS] + f"... [truncated, full length {len(text)} chars]"
    return text
```

`MAX_PAGE_CHARS = 2000` keeps the tool result small enough that it doesn't exhaust the context window when many pages are fetched in one session.

## Step 4 — Write the system prompt

The system prompt sets the agent's role, lists the available tools, and states the rules it must follow. Clear rules reduce the chance the model hallucinates sources or skips verification:

```
You are a research assistant. You answer questions by iteratively searching
the web and reading pages.

Available tools:
- web_search(query): DuckDuckGo search. Returns up to 5 results.
- fetch_page(url): Downloads and returns readable text (truncated to 2000 chars).
- finalize_answer(answer, sources): Call this when you have enough information.

Rules:
- Think step by step. Make one tool call at a time.
- Before answering, verify your claim with at least one source.
- When ready, call finalize_answer with a clear answer and the source URLs
  you actually used.
- Do not fabricate URLs or quotes. If you cannot find an answer, say so.
```

The instruction "one tool call at a time" matters because some models try to batch calls; the loop only processes the first call in each response.

## Step 5 — Write `run_agent`

The core loop drives the ReAct cycle. At each step it:

1. Calls `litellm.completion()` with the current message history and the `TOOLS` list.
2. Appends the assistant message (including any `tool_calls` field) to the history.
3. Checks whether the response contains tool calls. If not, the model answered in plain text — return immediately.
4. Extracts the first tool call, checks for a repeat (loop detection), prints a progress line, calls `_dispatch_tool`, and appends the tool result as a `{"role": "tool", ...}` message.
5. If the tool was `finalize_answer`, parse its JSON result and return.
6. After `max_steps` iterations without a final answer, return a timeout result.

Loop detection compares `(name, raw_args)` against the previous step. If identical, the agent is stuck and the loop exits early.

Token counts and cost are accumulated from `response.usage` and `litellm.completion_cost()` across every step, so the caller always gets an accurate total.

The function returns a dict with keys `answer`, `sources`, `steps_used`, `total_input_tokens`, `total_output_tokens`, `total_cost`, `trace`, and `stop_reason`.

## Step 6 — CLI entry point

Wire up `argparse` with one positional argument and two flags:

```python
parser.add_argument("question", help="The question to research")
parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS,
                    help="Maximum agent iterations")
parser.add_argument("--verbose", action="store_true",
                    help="Print full tool arguments and result previews")
```

After `run_agent()` returns, print the answer, the source URLs (if any), the step count and stop reason, and the token and cost totals.

## Running it

```bash
python solution.py "What is the latest stable Python version?"
```

Expected console output (exact values vary):

```
[Step 1] Tool: web_search({"query": "latest stable Python version"})
[Step 1] Result: [{"title": "Python Releases", "url": "https://www.python.or...
[Step 2] Tool: fetch_page({"url": "https://www.python.org/downloads/"})
[Step 2] Result: Python Downloads | Python.org Welcome to Python.org Downloa...
[Step 3] Tool: finalize_answer({"answer": "The latest stable Python version ...
[Step 3] Result: {"answer": "The latest stable Python version is 3.13.2, rel...

=== Answer ===
The latest stable Python version is 3.13.2, released in February 2025.

Sources:
- https://www.python.org/downloads/

Steps used: 3/8 (final_answer)
Tokens:     in=3241 out=187
Cost:       $0.000612
```

Add `--verbose` to see the full tool arguments and the first 400 characters of every result.

## What to try next

1. **Calculator tool** — add a `calculate(expression)` tool that evaluates a Python expression with `eval()` inside a safe sandbox; the agent can then answer numeric questions without needing to search for arithmetic results
2. **Swap the search backend** — replace `DDGS` with the Tavily Python client (`tavily-python`) for a search API that is specifically designed for LLM agents and returns cleaner, more reliable results
3. **Fetch caching** — wrap `fetch_page` with an in-memory `dict` keyed on URL so repeated visits to the same page within one session are instant and free
4. **Retry on rate limits** — add a `tenacity` decorator around the `completion()` call to retry with exponential backoff when the provider returns a 429 error
5. **`take_note` tool** — give the agent a scratchpad tool that stores text in a running list; for long multi-hop research tasks this helps the model track intermediate findings without relying solely on the message history

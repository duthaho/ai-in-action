"""
Research Assistant — complete reference implementation.

A ReAct agent that answers research questions by iteratively searching the web
and reading pages until it has enough information.

Run:
    python solution.py "What is the latest stable Python version?"
    python solution.py "Who wrote the book Dune?" --verbose
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from ddgs import DDGS
from litellm import completion, completion_cost

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

MODEL = os.getenv("LLM_MODEL", "anthropic/claude-sonnet-4-20250514")
MAX_PAGE_CHARS = 2000
MAX_SEARCH_RESULTS = 5
DEFAULT_MAX_STEPS = 8

USER_AGENT = (
    "Mozilla/5.0 (compatible; ResearchAssistant/1.0; "
    "+https://github.com/ai-in-action)"
)


# ---------- Tools ----------


def web_search(query: str) -> list[dict]:
    """DuckDuckGo search. Returns a list of {title, url, snippet}."""
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


def fetch_page(url: str) -> str:
    """Download a page and return readable text, truncated to MAX_PAGE_CHARS."""
    response = requests.get(url, timeout=15, headers={"User-Agent": USER_AGENT})
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator=" ")
    text = re.sub(r"\s+", " ", text).strip()

    if len(text) > MAX_PAGE_CHARS:
        text = text[:MAX_PAGE_CHARS] + f"... [truncated, full length {len(text)} chars]"

    return text


def finalize_answer(answer: str, sources: list[str]) -> str:
    """Sentinel tool. The agent calls this when it's ready to answer."""
    return json.dumps({"answer": answer, "sources": sources})


# ---------- Tool schema for LiteLLM ----------


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web using DuckDuckGo. Returns up to 5 results, "
                "each with a title, URL, and snippet. Use this when you need "
                "to find sources for a factual claim."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_page",
            "description": (
                "Download a web page and return its readable text (truncated "
                "to 2000 characters). Use this after web_search to read the "
                "content of a promising result."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to fetch.",
                    },
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finalize_answer",
            "description": (
                "Call this when you have enough information to answer the "
                "user's question. Provide a clear answer and list the source "
                "URLs you actually used."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "The final answer to the user's question.",
                    },
                    "sources": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of source URLs you used.",
                    },
                },
                "required": ["answer", "sources"],
            },
        },
    },
]


TOOL_IMPLS = {
    "web_search": web_search,
    "fetch_page": fetch_page,
    "finalize_answer": finalize_answer,
}


SYSTEM_PROMPT = """You are a research assistant. You answer questions by iteratively searching the web and reading pages.

Available tools:
- web_search(query): DuckDuckGo search. Returns up to 5 results.
- fetch_page(url): Downloads and returns readable text (truncated to 2000 chars).
- finalize_answer(answer, sources): Call this when you have enough information.

Rules:
- Think step by step. Make one tool call at a time.
- Before answering, verify your claim with at least one source.
- When ready, call finalize_answer with a clear answer and the source URLs you actually used.
- Do not fabricate URLs or quotes. If you cannot find an answer, say so."""


# ---------- Agent loop ----------


def _dispatch_tool(name: str, raw_args: str) -> str:
    """Parse arguments and call the tool. Return a string result (or error)."""
    try:
        args = json.loads(raw_args) if raw_args else {}
    except json.JSONDecodeError as e:
        return f"ERROR: could not parse tool arguments as JSON: {e}"

    impl = TOOL_IMPLS.get(name)
    if impl is None:
        return f"ERROR: unknown tool '{name}'"

    try:
        result = impl(**args)
    except Exception as e:
        return f"ERROR: tool {name} raised: {type(e).__name__}: {e}"

    if isinstance(result, (dict, list)):
        return json.dumps(result)
    return str(result)


def run_agent(
    question: str,
    max_steps: int = DEFAULT_MAX_STEPS,
    verbose: bool = False,
    model: str = MODEL,
) -> dict:
    """Run the ReAct loop until finalize_answer or max_steps."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    trace: list[dict] = []
    total_input = 0
    total_output = 0
    total_cost = 0.0
    last_tool_signature: tuple[str, str] | None = None

    for step in range(1, max_steps + 1):
        response = completion(
            model=model,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )

        usage = getattr(response, "usage", None)
        total_input += getattr(usage, "prompt_tokens", 0) if usage else 0
        total_output += getattr(usage, "completion_tokens", 0) if usage else 0
        try:
            total_cost += completion_cost(completion_response=response) or 0.0
        except Exception:
            pass

        msg = response.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))

        tool_calls = getattr(msg, "tool_calls", None) or []

        if not tool_calls:
            answer = (msg.content or "").strip()
            trace.append({"step": step, "tool": None, "result": answer[:200]})
            return {
                "answer": answer,
                "sources": [],
                "steps_used": step,
                "total_input_tokens": total_input,
                "total_output_tokens": total_output,
                "total_cost": round(total_cost, 6),
                "trace": trace,
                "stop_reason": "plain_text_answer",
            }

        call = tool_calls[0]
        name = call.function.name
        raw_args = call.function.arguments or "{}"

        signature = (name, raw_args)
        if signature == last_tool_signature:
            trace.append({"step": step, "tool": name, "args": raw_args, "result": "loop detected"})
            return {
                "answer": "Stopped: the agent was about to repeat the same tool call.",
                "sources": [],
                "steps_used": step,
                "total_input_tokens": total_input,
                "total_output_tokens": total_output,
                "total_cost": round(total_cost, 6),
                "trace": trace,
                "stop_reason": "loop_detected",
            }
        last_tool_signature = signature

        if verbose:
            print(f"[Step {step}] Tool: {name}({raw_args})")
        else:
            preview = raw_args if len(raw_args) < 80 else raw_args[:77] + "..."
            print(f"[Step {step}] Tool: {name}({preview})")

        result_str = _dispatch_tool(name, raw_args)

        if verbose:
            result_preview = result_str if len(result_str) < 400 else result_str[:400] + "..."
            print(f"[Step {step}] Result: {result_preview}")
        else:
            summary = result_str[:80].replace("\n", " ")
            print(f"[Step {step}] Result: {summary}{'...' if len(result_str) > 80 else ''}")

        trace.append({
            "step": step,
            "tool": name,
            "args": raw_args,
            "result": result_str[:500],
        })

        messages.append({
            "role": "tool",
            "tool_call_id": call.id,
            "content": result_str,
        })

        if name == "finalize_answer":
            try:
                parsed = json.loads(result_str)
                return {
                    "answer": parsed.get("answer", ""),
                    "sources": parsed.get("sources", []),
                    "steps_used": step,
                    "total_input_tokens": total_input,
                    "total_output_tokens": total_output,
                    "total_cost": round(total_cost, 6),
                    "trace": trace,
                    "stop_reason": "final_answer",
                }
            except json.JSONDecodeError:
                pass

    return {
        "answer": "Stopped: max_steps reached without a final answer.",
        "sources": [],
        "steps_used": max_steps,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_cost": round(total_cost, 6),
        "trace": trace,
        "stop_reason": "max_steps",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="ReAct Research Assistant")
    parser.add_argument("question", help="The question to research")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help=f"Maximum agent iterations (default {DEFAULT_MAX_STEPS})",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print full tool arguments and result previews",
    )
    args = parser.parse_args()

    result = run_agent(
        args.question,
        max_steps=args.max_steps,
        verbose=args.verbose,
    )

    print("\n=== Answer ===")
    print(result["answer"])

    if result["sources"]:
        print("\nSources:")
        for src in result["sources"]:
            print(f"- {src}")

    print(
        f"\nSteps used: {result['steps_used']}/{args.max_steps} "
        f"({result['stop_reason']})"
    )
    print(
        f"Tokens:     in={result['total_input_tokens']} "
        f"out={result['total_output_tokens']}"
    )
    print(f"Cost:       ${result['total_cost']}")


if __name__ == "__main__":
    main()

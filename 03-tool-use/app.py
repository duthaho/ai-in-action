"""
Tool Use / Function Calling — Standalone Demo

Walks through the complete tool use lifecycle:
1. Inspect tool definitions — what the model sees
2. Single-turn tool use with the agentic loop
3. Multi-tool calls — model requests multiple tools at once
4. Force tool use — require a specific tool
5. No tool needed — model responds directly
6. Multi-turn conversation with tool history

Run: python app.py
"""

import os

import anthropic
from pathlib import Path
from dotenv import load_dotenv

from tools import TOOL_DEFINITIONS, execute_tool

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

MODEL = "claude-sonnet-4-20250514"

SYSTEM_PROMPT = """\
You are a helpful customer support assistant. You have access to tools for:
- Checking weather in any city
- Looking up customer orders by email
- Performing calculations

Always use tools when you need factual data — never guess or make up information.
If you need multiple pieces of information, request all relevant tools."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def print_header(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def print_separator():
    print(f"\n{'-' * 60}\n")


def run_agentic_loop(
    client: anthropic.Anthropic,
    messages: list[dict],
    max_iterations: int = 10,
) -> dict:
    """
    The core agentic loop pattern:
    1. Send messages + tool definitions to the model
    2. If model wants to use tools → execute them → send results back
    3. Repeat until model produces a final text response
    """
    tool_calls_log = []
    iteration = 0

    while iteration < max_iterations:
        iteration += 1

        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=TOOL_DEFINITIONS,
            messages=messages,
        )

        # Case 1: Model is done — produced a final text response
        if response.stop_reason == "end_turn":
            final_text = "".join(
                b.text for b in response.content if b.type == "text"
            )
            return {
                "response": final_text,
                "tool_calls": tool_calls_log,
                "iterations": iteration,
            }

        # Case 2: Model wants to use tools
        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})

            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = execute_tool(block.name, block.input)
                    tool_calls_log.append({
                        "iteration": iteration,
                        "tool": block.name,
                        "input": block.input,
                        "result": result,
                    })
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })

            messages.append({"role": "user", "content": tool_results})

    return {
        "response": "Max iterations reached.",
        "tool_calls": tool_calls_log,
        "iterations": iteration,
    }


# ---------------------------------------------------------------------------
# Demo sections
# ---------------------------------------------------------------------------

def demo_tool_definitions():
    """Show the tool schemas that the model receives."""
    print_header("1. Tool Definitions — What the Model Sees")

    print(f"  {len(TOOL_DEFINITIONS)} tools registered:\n")
    for tool in TOOL_DEFINITIONS:
        required = tool["input_schema"].get("required", [])
        optional = [
            k for k in tool["input_schema"]["properties"]
            if k not in required
        ]
        print(f"  Tool: {tool['name']}")
        print(f"    Description: {tool['description'][:80]}...")
        print(f"    Required: {required}")
        print(f"    Optional: {optional}")
        print()

    print("  These exact schemas are injected into the model's context.")
    print("  Better descriptions = better tool selection accuracy.")


def demo_single_tool(client: anthropic.Anthropic):
    """Weather query triggers get_weather tool."""
    print_header("2. Single Tool — Weather Query")

    message = "What's the weather like in Tokyo right now?"
    print(f"  User: \"{message}\"\n")

    result = run_agentic_loop(
        client,
        [{"role": "user", "content": message}],
    )

    for tc in result["tool_calls"]:
        print(f"  Tool call: {tc['tool']}({tc['input']})")
        print(f"    → {tc['result']}")
    print(f"\n  Response: {result['response'][:200]}")
    print(f"  Iterations: {result['iterations']}")


def demo_order_lookup(client: anthropic.Anthropic):
    """Order query triggers search_orders tool."""
    print_header("3. Order Lookup — search_orders Tool")

    message = "Can you check what orders bob@example.com has? Any pending ones?"
    print(f"  User: \"{message}\"\n")

    result = run_agentic_loop(
        client,
        [{"role": "user", "content": message}],
    )

    for tc in result["tool_calls"]:
        print(f"  Tool call: {tc['tool']}({tc['input']})")
        print(f"    → {tc['result'][:100]}")
    print(f"\n  Response: {result['response'][:300]}")


def demo_calculation(client: anthropic.Anthropic):
    """Math query triggers calculate tool."""
    print_header("4. Calculation — LLMs Can't Do Math")

    message = "What is 1847 multiplied by 293, plus 15?"
    print(f"  User: \"{message}\"\n")

    result = run_agentic_loop(
        client,
        [{"role": "user", "content": message}],
    )

    for tc in result["tool_calls"]:
        print(f"  Tool call: {tc['tool']}({tc['input']})")
        print(f"    → {tc['result']}")
    print(f"\n  Response: {result['response'][:200]}")
    print("\n  Key insight: LLMs are unreliable at arithmetic.")
    print("  Always delegate math to a tool.")


def demo_multi_tool(client: anthropic.Anthropic):
    """Multiple tools in one request — model calls them in parallel."""
    print_header("5. Multi-Tool — Parallel Tool Calls")

    message = (
        "I need three things: "
        "1) Weather in Tokyo and London, "
        "2) All orders for alice@example.com, "
        "3) Calculate the total of 149.99 + 49.99"
    )
    print(f"  User: \"{message}\"\n")

    result = run_agentic_loop(
        client,
        [{"role": "user", "content": message}],
    )

    print(f"  Tool calls ({len(result['tool_calls'])} total):")
    for tc in result["tool_calls"]:
        print(f"    [{tc['iteration']}] {tc['tool']}({tc['input']})")
    print(f"\n  Response: {result['response'][:400]}")
    print(f"  Iterations: {result['iterations']}")
    print("\n  Notice: Claude can request multiple tools in one turn (parallel calls).")


def demo_no_tool(client: anthropic.Anthropic):
    """Simple chat — model decides no tool is needed."""
    print_header("6. No Tool Needed — Model Responds Directly")

    message = "Hello! How are you today?"
    print(f"  User: \"{message}\"\n")

    result = run_agentic_loop(
        client,
        [{"role": "user", "content": message}],
    )

    print(f"  Response: {result['response'][:200]}")
    print(f"  Tool calls: {len(result['tool_calls'])} (should be 0)")
    print(f"  Iterations: {result['iterations']} (should be 1)")
    print("\n  The model only calls tools when it needs external data.")


def demo_force_tool(client: anthropic.Anthropic):
    """Force the model to use a specific tool via tool_choice."""
    print_header("7. Force Tool — tool_choice Parameter")

    message = "Tell me about Paris."
    tool_name = "get_weather"
    print(f"  User: \"{message}\"")
    print(f"  Forcing tool: {tool_name}\n")

    messages = [{"role": "user", "content": message}]

    # Force the model to call get_weather
    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        tools=TOOL_DEFINITIONS,
        tool_choice={"type": "tool", "name": tool_name},
        messages=messages,
    )

    # Execute the forced tool call
    messages.append({"role": "assistant", "content": response.content})
    tool_results = []

    for block in response.content:
        if block.type == "tool_use":
            result = execute_tool(block.name, block.input)
            print(f"  Forced call: {block.name}({block.input})")
            print(f"    → {result}")
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result,
            })

    messages.append({"role": "user", "content": tool_results})

    # Let the model generate a final response
    final = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        tools=TOOL_DEFINITIONS,
        messages=messages,
    )

    final_text = "".join(b.text for b in final.content if b.type == "text")
    print(f"\n  Response: {final_text[:200]}")

    print("\n  tool_choice options:")
    print('    {"type": "auto"}  → model decides (default)')
    print('    {"type": "any"}   → must use SOME tool')
    print('    {"type": "tool", "name": "..."} → must use THIS tool')


def demo_multi_turn(client: anthropic.Anthropic):
    """Multi-turn conversation where context carries across turns."""
    print_header("8. Multi-Turn Conversation")

    # Turn 1
    print("  --- Turn 1 ---")
    message1 = "What's the weather in Sydney?"
    print(f"  User: \"{message1}\"\n")

    result1 = run_agentic_loop(
        client,
        [{"role": "user", "content": message1}],
    )
    print(f"  Assistant: {result1['response'][:150]}")
    print(f"  Tools: {[tc['tool'] for tc in result1['tool_calls']]}")

    # Turn 2 — references previous context
    print(f"\n  --- Turn 2 ---")
    message2 = "How about in London? Is it warmer or colder than Sydney?"
    print(f"  User: \"{message2}\"\n")

    result2 = run_agentic_loop(
        client,
        [
            {"role": "user", "content": message1},
            {"role": "assistant", "content": result1["response"]},
            {"role": "user", "content": message2},
        ],
    )
    print(f"  Assistant: {result2['response'][:200]}")
    print(f"  Tools: {[tc['tool'] for tc in result2['tool_calls']]}")

    print("\n  The model sees full history including previous tool results,")
    print("  so it can compare data across turns without re-calling tools.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print_header("Tool Use / Function Calling — Demo")

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("ANTHROPIC_API_KEY not set. Set it in .env to run this demo.")
        demo_tool_definitions()
        return

    client = anthropic.Anthropic()

    demo_tool_definitions()
    print_separator()

    demo_single_tool(client)
    print_separator()

    demo_order_lookup(client)
    print_separator()

    demo_calculation(client)
    print_separator()

    demo_multi_tool(client)
    print_separator()

    demo_no_tool(client)
    print_separator()

    demo_force_tool(client)
    print_separator()

    demo_multi_turn(client)

    print_header("Done!")


if __name__ == "__main__":
    main()

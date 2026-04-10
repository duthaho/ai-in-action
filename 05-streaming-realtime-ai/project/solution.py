"""
Streaming Chat — Module 05 Project (Solution)

A terminal chat app that streams LLM responses token-by-token,
measures TTFT and TPS, supports multi-turn conversation,
and lets you toggle streaming on/off.

Run: python solution.py
"""

import os
import time
from pathlib import Path
from dotenv import load_dotenv
from litellm import completion, completion_cost
import litellm

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

MODEL = os.getenv("LLM_MODEL", "anthropic/claude-sonnet-4-20250514")
SYSTEM_PROMPT = "You are a helpful assistant. Keep responses concise."


# ---------------------------------------------------------------------------
# Step 1 & 2: Streaming with metrics
# ---------------------------------------------------------------------------

def stream_response(messages: list[dict], model: str = MODEL) -> dict:
    """Stream an LLM response, printing tokens as they arrive.

    Returns a dict with content, timing metrics, and cost.
    """
    start = time.monotonic()
    first_token_time = None
    token_count = 0
    full_response = ""
    input_tokens = 0
    output_tokens = 0

    response = completion(
        model=model,
        messages=messages,
        stream=True,
        stream_options={"include_usage": True},
    )

    for chunk in response:
        # Check for usage chunk (empty choices, has usage)
        if hasattr(chunk, "usage") and chunk.usage is not None:
            input_tokens = getattr(chunk.usage, "prompt_tokens", 0) or 0
            output_tokens = getattr(chunk.usage, "completion_tokens", 0) or 0

        if chunk.choices:
            content = chunk.choices[0].delta.content
            if content:
                if first_token_time is None:
                    first_token_time = time.monotonic()
                token_count += 1
                full_response += content
                print(content, end="", flush=True)

    print()  # newline after streaming

    total_time = time.monotonic() - start
    ttft = (first_token_time - start) if first_token_time else total_time
    generation_time = total_time - ttft
    tps = token_count / generation_time if generation_time > 0 else 0

    # If usage wasn't in stream, use token count as fallback
    if output_tokens == 0:
        output_tokens = token_count

    # Estimate cost
    cost = 0.0
    try:
        # Build a mock response for completion_cost
        cost = completion_cost(
            model=model,
            prompt=str(input_tokens),
            completion=full_response,
        )
    except Exception:
        pass  # cost estimation not critical

    return {
        "content": full_response,
        "ttft_ms": int(ttft * 1000),
        "tps": round(tps, 1),
        "total_time": round(total_time, 1),
        "output_tokens": output_tokens,
        "input_tokens": input_tokens,
        "cost": cost,
        "model": model,
        "streamed": True,
    }


# ---------------------------------------------------------------------------
# Non-streaming call (for toggle comparison)
# ---------------------------------------------------------------------------

def blocking_response(messages: list[dict], model: str = MODEL) -> dict:
    """Make a non-streaming API call and return the result."""
    start = time.monotonic()

    response = completion(
        model=model,
        messages=messages,
    )

    total_time = time.monotonic() - start
    choice = response.choices[0]
    usage = response.usage

    cost = 0.0
    try:
        cost = completion_cost(completion_response=response)
    except Exception:
        pass

    content = choice.message.content
    print(f"\n  ({total_time:.1f}s)")
    print(content)

    return {
        "content": content,
        "ttft_ms": None,
        "tps": None,
        "total_time": round(total_time, 1),
        "output_tokens": usage.completion_tokens,
        "input_tokens": usage.prompt_tokens,
        "cost": cost,
        "model": model,
        "streamed": False,
    }


# ---------------------------------------------------------------------------
# Step 3: Metrics display
# ---------------------------------------------------------------------------

def print_metrics(result: dict) -> None:
    """Print a formatted metrics line after a response."""
    if result["streamed"]:
        print(
            f"\n  TTFT: {result['ttft_ms']}ms"
            f" | TPS: {result['tps']}"
            f" | Tokens: {result['output_tokens']}"
            f" | Cost: ${result['cost']:.4f}"
            f" | Time: {result['total_time']}s"
        )
    else:
        print(
            f"\n  Tokens: {result['output_tokens']}"
            f" | Cost: ${result['cost']:.4f}"
            f" | Time: {result['total_time']}s"
        )


# ---------------------------------------------------------------------------
# Step 6: Session tracking
# ---------------------------------------------------------------------------

class SessionTracker:
    """Track all responses during a chat session."""

    def __init__(self):
        self.results = []
        self.start_time = time.monotonic()

    def record(self, result: dict) -> None:
        """Record a response result."""
        self.results.append(result)

    def summary(self) -> dict:
        """Produce a session summary."""
        duration = time.monotonic() - self.start_time
        total_input = sum(r["input_tokens"] for r in self.results)
        total_output = sum(r["output_tokens"] for r in self.results)
        total_cost = sum(r["cost"] for r in self.results)

        streamed = [r for r in self.results if r["streamed"] and r["ttft_ms"] is not None]
        avg_ttft = int(sum(r["ttft_ms"] for r in streamed) / len(streamed)) if streamed else None
        avg_tps = round(sum(r["tps"] for r in streamed) / len(streamed), 1) if streamed else None

        return {
            "message_count": len(self.results),
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_tokens": total_input + total_output,
            "total_cost": total_cost,
            "avg_ttft_ms": avg_ttft,
            "avg_tps": avg_tps,
            "duration": int(duration),
        }


def print_summary(tracker: SessionTracker) -> None:
    """Print a formatted session summary."""
    s = tracker.summary()
    print()
    print("=" * 60)
    print("  Session Summary")
    print("=" * 60)
    print(f"  Messages:      {s['message_count'] * 2} ({s['message_count']} you / {s['message_count']} assistant)")
    print(f"  Total tokens:  {s['total_tokens']:,} ({s['total_input_tokens']} in + {s['total_output_tokens']} out)")
    print(f"  Total cost:    ${s['total_cost']:.4f}")
    if s["avg_ttft_ms"] is not None:
        print(f"  Avg TTFT:      {s['avg_ttft_ms']}ms")
    if s["avg_tps"] is not None:
        print(f"  Avg TPS:       {s['avg_tps']}")
    print(f"  Duration:      {s['duration']}s")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Step 4 & 5: Chat loop with toggle
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  Streaming Chat")
    print("=" * 60)
    print(f"  Model: {MODEL}")
    print("  Type /toggle to switch streaming, /bye to quit")
    print()

    messages = []
    streaming_enabled = True
    tracker = SessionTracker()

    while True:
        mode = "streaming on" if streaming_enabled else "streaming off"
        try:
            user_input = input(f"[{mode}] You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue

        if user_input.lower() in ("/bye", "quit", "exit"):
            break

        if user_input.lower() == "/toggle":
            streaming_enabled = not streaming_enabled
            state = "ON" if streaming_enabled else "OFF"
            print(f"  Streaming {state}")
            continue

        # Build messages with system prompt
        api_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
        api_messages.append({"role": "user", "content": user_input})

        print()
        try:
            if streaming_enabled:
                result = stream_response(api_messages, MODEL)
            else:
                result = blocking_response(api_messages, MODEL)

            print_metrics(result)
            tracker.record(result)

            # Update conversation history
            messages.append({"role": "user", "content": user_input})
            messages.append({"role": "assistant", "content": result["content"]})

        except litellm.AuthenticationError:
            print("  Error: Invalid API key. Check your .env file.")
        except litellm.RateLimitError:
            print("  Error: Rate limited. Wait a moment and try again.")
        except Exception as e:
            print(f"  Error: {type(e).__name__}: {str(e)[:100]}")

        print()

    # Session summary
    if tracker.results:
        print_summary(tracker)


if __name__ == "__main__":
    main()

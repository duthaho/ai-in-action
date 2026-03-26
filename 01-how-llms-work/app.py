"""
How LLMs Work — Standalone Demo

Walks through core LLM concepts interactively:
1. Tokenization — how text becomes tokens
2. Token cost comparison — why format matters
3. Context window awareness
4. Temperature effects on output
5. Streaming generation — autoregressive decoding
6. Structured output — constraining the distribution

Run: python app.py
"""

import json
import os
import time

import anthropic
import tiktoken
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

MODEL = "claude-sonnet-4-20250514"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def print_header(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def print_separator():
    print(f"\n{'-' * 60}\n")


# ---------------------------------------------------------------------------
# 1. Tokenization — see how text becomes tokens
# ---------------------------------------------------------------------------

def demo_tokenization(enc: tiktoken.Encoding):
    print_header("1. Tokenization — How Text Becomes Tokens")

    examples = [
        "Hello, world!",
        "unhappiness",
        '{"user_name": "John", "user_age": 30}',
        '{"n":"John","a":30}',
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
    ]

    for text in examples:
        token_ids = enc.encode(text)
        tokens = [enc.decode([tid]) for tid in token_ids]

        print(f"Text: \"{text[:60]}\"")
        print(f"  Tokens ({len(token_ids)}): {tokens}")
        print(f"  Cost: ~${len(token_ids) * 3 / 1_000_000:.6f} at $3/M input tokens")
        print()

    print("Key insight: the same data in different formats costs different amounts.")
    print("JSON key names, whitespace, and verbose naming all consume tokens.")


# ---------------------------------------------------------------------------
# 2. Token cost comparison
# ---------------------------------------------------------------------------

def demo_token_comparison(enc: tiktoken.Encoding):
    print_header("2. Token Cost Comparison — Why Format Matters")

    texts = [
        '{"user_full_name": "John Doe", "user_email_address": "john@example.com", "user_age_years": 30}',
        '{"name":"John Doe","email":"john@example.com","age":30}',
        "name:John Doe|email:john@example.com|age:30",
    ]

    results = []
    for text in texts:
        count = len(enc.encode(text))
        results.append((count, text))

    results.sort(key=lambda x: x[0])
    for count, text in results:
        print(f"  {count:3d} tokens: {text[:80]}")

    ratio = results[-1][0] / results[0][0]
    print(f"\n  The verbose format uses {ratio:.1f}x more tokens than the compact one.")
    print("  At scale, this directly affects cost and context window usage.")


# ---------------------------------------------------------------------------
# 3. Context window awareness
# ---------------------------------------------------------------------------

def demo_context_window(enc: tiktoken.Encoding):
    print_header("3. Context Window — Pre-flight Check")

    system = "You are an AI tutor."
    prompt = "Explain transformers." * 100

    system_tokens = len(enc.encode(system))
    prompt_tokens = len(enc.encode(prompt))
    total_input = system_tokens + prompt_tokens
    max_output = 500
    context_limit = 200_000

    available = context_limit - total_input
    fits = total_input + max_output <= context_limit
    utilization = total_input / context_limit * 100

    print(f"  System prompt:      {system_tokens:,} tokens")
    print(f"  User prompt:        {prompt_tokens:,} tokens")
    print(f"  Total input:        {total_input:,} tokens")
    print(f"  Requested output:   {max_output:,} tokens")
    print(f"  Context limit:      {context_limit:,} tokens")
    print(f"  Available for output: {available:,} tokens")
    print(f"  Fits in context:    {fits}")
    print(f"  Utilization:        {utilization:.2f}%")

    print("\n  In production, always check this BEFORE calling the API.")
    print("  If input is too large, you must truncate or chunk.")


# ---------------------------------------------------------------------------
# 4. Temperature effects
# ---------------------------------------------------------------------------

def demo_temperature(client: anthropic.Anthropic):
    print_header("4. Temperature — Same Prompt, Different Randomness")

    prompt = "Complete this sentence in exactly 10 words: 'The robot walked into the bar and'"
    temperatures = [0.0, 0.5, 1.0]

    print(f"  Prompt: \"{prompt}\"\n")

    for temp in temperatures:
        message = client.messages.create(
            model=MODEL,
            max_tokens=50,
            temperature=temp,
            messages=[{"role": "user", "content": prompt}],
        )
        output = message.content[0].text.strip()
        print(f"  T={temp}: {output[:100]}")

    print("\n  T=0 → deterministic (always picks highest-probability token)")
    print("  T=0.5 → moderate randomness, usually coherent")
    print("  T=1.0 → standard sampling, more creative/varied")
    print("  Run this multiple times to see variance at each temperature.")


# ---------------------------------------------------------------------------
# 5. Streaming generation
# ---------------------------------------------------------------------------

def demo_streaming(client: anthropic.Anthropic):
    print_header("5. Streaming — Token-by-Token Autoregressive Output")

    prompt = "Count from 1 to 10, one number per line."

    # Non-streaming
    print("  Non-streaming (full response at once):")
    start = time.perf_counter()
    message = client.messages.create(
        model=MODEL,
        max_tokens=100,
        temperature=0.0,
        messages=[{"role": "user", "content": prompt}],
    )
    elapsed = time.perf_counter() - start
    print(f"  {message.content[0].text.strip()}")
    print(f"  Time: {elapsed:.2f}s | Tokens: {message.usage.input_tokens} in, {message.usage.output_tokens} out")

    print()

    # Streaming
    print("  Streaming (token by token):")
    print("  ", end="")
    start = time.perf_counter()
    with client.messages.stream(
        model=MODEL,
        max_tokens=100,
        temperature=0.0,
        messages=[{"role": "user", "content": "Count from 1 to 5, one number per line."}],
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
    elapsed = time.perf_counter() - start
    print(f"\n  Time: {elapsed:.2f}s (first token appears much sooner)")

    print("\n  Streaming works because the model generates left-to-right —")
    print("  each token only depends on previous tokens, not future ones.")


# ---------------------------------------------------------------------------
# 6. Structured output
# ---------------------------------------------------------------------------

def demo_structured_output(client: anthropic.Anthropic):
    print_header("6. Structured Output — Constraining the Distribution")

    diff = """\
- def get_user(id):
-     query = f"SELECT * FROM users WHERE id = {id}"
-     return db.execute(query)
+ def get_user(user_id):
+     query = "SELECT * FROM users WHERE id = %s"
+     return db.execute(query, (user_id,))"""

    system = (
        "You are a senior code reviewer. Analyze the given diff and return your review "
        "as a JSON array. Each element must have exactly these fields:\n"
        '- "line": the line number or range (string)\n'
        '- "severity": "critical" | "warning" | "suggestion"\n'
        '- "issue": one-sentence description of the problem\n'
        '- "fix": one-sentence suggested fix\n\n'
        "Return ONLY the JSON array, no markdown fences, no explanation."
    )

    print(f"  Code diff:\n{diff}\n")
    print("  Asking Claude for structured JSON review...\n")

    message = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        temperature=0.0,
        system=system,
        messages=[{"role": "user", "content": f"Language: python\n\nDiff:\n{diff}"}],
    )

    raw = message.content[0].text
    try:
        parsed = json.loads(raw)
        print(f"  Parse success: True")
        for item in parsed:
            severity = item.get("severity", "?")
            line = item.get("line", "?")
            issue = item.get("issue", "?")
            print(f"  [{severity}] Line {line}: {issue}")
    except json.JSONDecodeError:
        print(f"  Parse success: False")
        print(f"  Raw response: {raw[:200]}")

    print(f"\n  Usage: {message.usage.input_tokens} in, {message.usage.output_tokens} out")
    print("\n  The system prompt constrains the model's output distribution toward")
    print("  valid JSON. In production, use tool_use for guaranteed schema compliance.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print_header("How LLMs Work — Demo")

    # tiktoken encoder for token counting (OpenAI's cl100k_base —
    # used to illustrate tokenization concepts; Anthropic uses its own
    # tokenizer, but the principles are identical)
    enc = tiktoken.get_encoding("cl100k_base")

    demo_tokenization(enc)
    print_separator()

    demo_token_comparison(enc)
    print_separator()

    demo_context_window(enc)
    print_separator()

    # API demos require ANTHROPIC_API_KEY
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Skipping API demos (4-6) — ANTHROPIC_API_KEY not set.")
        print("Set it in .env to see temperature, streaming, and structured output demos.")
        return

    client = anthropic.Anthropic()

    demo_temperature(client)
    print_separator()

    demo_streaming(client)
    print_separator()

    demo_structured_output(client)

    print_header("Done!")


if __name__ == "__main__":
    main()

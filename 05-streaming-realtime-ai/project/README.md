# Project: Streaming Chat

## What you'll build

A terminal chat app that streams LLM responses token-by-token in real-time. It measures TTFT and TPS after each response, tracks cost, supports multi-turn conversation, and lets you toggle streaming on/off to feel the difference firsthand. This is the foundation of every chat interface you'll build.

## Prerequisites

- Completed reading the Module 05 README
- Python 3.11+ with project dependencies installed (`pip install -r requirements.txt`)
- At least one LLM provider API key configured in `.env`

## How to build

Create a new file `streaming_chat.py` in this directory. Build it step by step following the instructions below. When you're done, compare your output with `python solution.py`.

## Steps

### Step 1: Basic streaming call

Set up imports, load environment variables, and write a basic streaming function.

```python
import os
import time
from pathlib import Path
from dotenv import load_dotenv
from litellm import completion, completion_cost
import litellm

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

MODEL = os.getenv("LLM_MODEL", "anthropic/claude-sonnet-4-20250514")
```

Write a `stream_response(messages, model)` function that:
- Calls `completion()` with `stream=True` and `stream_options={"include_usage": True}`
- Iterates over chunks, printing each `delta.content` token with `print(content, end="", flush=True)`
- Accumulates the full response text by concatenating delta content
- Returns the full response text

Test: call it with `[{"role": "user", "content": "Say hello in one sentence"}]` and watch the tokens stream.

### Step 2: Measure TTFT and TPS

Enhance `stream_response()` to measure timing and return a result dict.

Add timing around the stream loop:
- Record `start = time.monotonic()` before the call
- On the first content token, record `first_token_time`
- Count output tokens during streaming
- After the loop, calculate: TTFT (ms), TPS, total time, token count

Also capture usage from `stream_options` — check for chunks with empty `choices` that carry usage data.

Return a dict with: `content`, `ttft_ms`, `tps`, `total_time`, `output_tokens`, `input_tokens`, `cost`, `model`.

Test: call and print the metrics dict.

### Step 3: Display metrics

Write a `print_metrics(result)` function that prints a formatted metrics line after each streamed response:

```
  TTFT: 342ms | TPS: 78.4 | Tokens: 45 | Cost: $0.0007 | Time: 0.9s
```

Test: stream a response, then call `print_metrics()` with the result.

### Step 4: Multi-turn chat loop

Build the interactive chat loop:
- Maintain a `messages` list for conversation history
- Print a header with model name and instructions
- Loop: read user input with `input()`, stream response, record history
- Append user message before calling, append assistant response after streaming
- Handle commands: `/bye` to quit
- Handle empty input (skip)

Test: have a multi-turn conversation. Verify the model remembers context from earlier messages.

### Step 5: Streaming toggle

Add a `/toggle` command that switches between streaming and non-streaming mode:
- Track a `streaming_enabled` flag (default: `True`)
- When streaming is on: use `stream_response()` as before
- When streaming is off: use regular `completion()`, print the full response at once, still measure latency and cost
- Show mode in the prompt: `[streaming on] You:` or `[streaming off] You:`
- Print a confirmation when toggling: `Streaming ON` or `Streaming OFF`

Test: toggle off, ask a question (notice the wait), toggle on, ask again (notice the typewriter). Feel the difference.

### Step 6: Session summary

Create a `SessionTracker` class to track all responses during the session:
- `record(result)` — store a result dict
- `summary()` — return summary dict with: message_count, total_input_tokens, total_output_tokens, total_tokens, total_cost, avg_ttft_ms (streaming responses only), avg_tps (streaming responses only), duration

When the user types `/bye`, print the session summary before exiting:

```
============================================================
  Session Summary
============================================================
  Messages:      4 (2 you / 2 assistant)
  Total tokens:  187 (90 in + 97 out)
  Total cost:    $0.0015
  Avg TTFT:      342ms
  Avg TPS:       78.4
  Duration:      45s
============================================================
```

## How to run

```bash
python streaming_chat.py
```

Or compare with the reference:

```bash
python solution.py
```

## Expected output

```
============================================================
  Streaming Chat
============================================================
  Model: anthropic/claude-sonnet-4-20250514
  Type /toggle to switch streaming, /bye to quit

[streaming on] You: What is Python?

Python is a high-level, interpreted programming language known
for its clean syntax and readability. It was created by Guido
van Rossum and first released in 1991...

  TTFT: 342ms | TPS: 78.4 | Tokens: 45 | Cost: $0.0007 | Time: 0.9s

[streaming on] You: /toggle
  Streaming OFF

[streaming off] You: What is JavaScript?

  (1.8s)
JavaScript is a versatile programming language primarily used
for web development...

  Tokens: 52 | Cost: $0.0008 | Time: 1.8s

[streaming off] You: /toggle
  Streaming ON

[streaming on] You: /bye

============================================================
  Session Summary
============================================================
  Messages:      4 (2 you / 2 assistant)
  Total tokens:  187 (90 in + 97 out)
  Total cost:    $0.0015
  Avg TTFT:      342ms
  Avg TPS:       78.4
  Duration:      32s
============================================================
```

## Stretch goals

1. **Token-by-token cost ticker** — show a running cost counter that updates as tokens stream in, displayed after the response line.
2. **Model switching** — add a `/model <name>` command to switch models mid-conversation and compare streaming performance.
3. **Save conversation** — add a `/save` command to export the conversation history and metrics to a JSON file.

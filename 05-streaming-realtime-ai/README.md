# Module 05 — Streaming & Real-Time AI

Delivering LLM responses token-by-token: streaming protocols, real-time metrics, and building responsive AI experiences.

| Detail        | Value                                     |
|---------------|-------------------------------------------|
| Level         | Beginner                                  |
| Time          | ~1.5 hours                                |
| Prerequisites | Module 04 (The AI API Layer)              |

## What you'll build

After reading this module, head to [`project/`](project/) to build a **Streaming Chat** — a terminal chat app that streams responses in real-time, measures TTFT and TPS, and lets you toggle streaming on/off to feel the difference.

---

## Table of Contents

1. [What is Streaming?](#1-what-is-streaming)
2. [Streaming with LiteLLM](#2-streaming-with-litellm)
3. [Anatomy of a Stream Chunk](#3-anatomy-of-a-stream-chunk)
4. [TTFT and Generation Metrics](#4-ttft-and-generation-metrics)
5. [Building a Streaming Chat](#5-building-a-streaming-chat)
6. [Streaming vs Non-Streaming](#6-streaming-vs-non-streaming)
7. [Error Handling in Streams](#7-error-handling-in-streams)
8. [Async Streaming](#8-async-streaming)

---

## 1. What is Streaming?

When you make a non-streaming API call, the server generates the entire response before sending anything back. For a 500-token response at 80 tokens/second, you wait ~6 seconds staring at nothing, then the full text appears at once.

Streaming flips this: the server sends each token as it's generated. You see the first token in under a second, then watch the response build word by word — like someone typing in real time.

### The key insight: perceived vs actual latency

Streaming does **not** make the model generate faster. A 500-token response still takes ~6 seconds of total generation time. What changes is the **experience**:

| Metric | Non-streaming | Streaming |
|--------|--------------|-----------|
| Time to see first word | ~6s (full wait) | ~0.3-1s (TTFT) |
| Time to see full response | ~6s | ~6s |
| User perception | "Is it working?" | "It's thinking..." |

For chat interfaces, this difference is dramatic. Users perceive a 0.5s wait as "instant" but a 6s wait as "broken."

### Server-Sent Events (SSE) under the hood

LLM APIs use **Server-Sent Events** — a simple protocol built on standard HTTP:

1. The client sends a regular HTTP POST request
2. The server holds the connection open and sets `Content-Type: text/event-stream`
3. The server pushes events as `data:` lines, each followed by a blank line:
   ```
   data: {"id":"chatcmpl-abc","choices":[{"delta":{"content":"Hello"}}]}

   data: {"id":"chatcmpl-abc","choices":[{"delta":{"content":" world"}}]}

   data: [DONE]
   ```
4. `data: [DONE]` signals the end of the stream

### Why SSE, not WebSockets?

LLM generation is one-directional: you send one request, the server streams many tokens back. SSE is built for this — it's simpler than WebSockets (standard HTTP, works with proxies and CDNs, no upgrade handshake) and sufficient because there's nothing to send back during generation.

---

## 2. Streaming with LiteLLM

Enable streaming by adding `stream=True` to your `completion()` call. Instead of a single response object, you get an **iterator** that yields chunks one by one:

```python
from litellm import completion

response = completion(
    model="openai/gpt-4o",
    messages=[{"role": "user", "content": "Explain Python decorators"}],
    stream=True,
)

for chunk in response:
    content = chunk.choices[0].delta.content
    if content:
        print(content, end="", flush=True)

print()  # newline after streaming completes
```

### Key differences from non-streaming

| Aspect | Non-streaming | Streaming |
|--------|--------------|-----------|
| Return type | Single `ModelResponse` | Iterator of `ModelResponseStream` chunks |
| Content access | `response.choices[0].message.content` | `chunk.choices[0].delta.content` |
| Content field | `message` (full text) | `delta` (incremental token) |
| Availability | All at once after generation completes | One chunk at a time during generation |

### The `delta` vs `message` distinction

In non-streaming, `choices[0].message.content` contains the entire response text.

In streaming, `choices[0].delta.content` contains only the **new token(s)** in this chunk — usually one or a few tokens. You accumulate the full response by concatenating deltas:

```python
full_response = ""
for chunk in response:
    content = chunk.choices[0].delta.content
    if content:
        full_response += content
        print(content, end="", flush=True)
# full_response now contains the complete text
```

### Why `flush=True`?

Python's `print()` buffers output by default — it waits until it has a full line before writing to the terminal. `flush=True` forces each token to appear immediately, creating the typewriter effect. Without it, tokens accumulate invisibly and appear in bursts.

---

## 3. Anatomy of a Stream Chunk

Understanding what each chunk contains helps you parse streams correctly and extract metrics.

### OpenAI chunk format

A stream produces three types of chunks in sequence:

**First chunk** — sets the role, no content yet:
```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion.chunk",
  "model": "gpt-4o",
  "choices": [{
    "index": 0,
    "delta": {"role": "assistant"},
    "finish_reason": null
  }]
}
```

**Middle chunks** — each carries one or more tokens:
```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion.chunk",
  "choices": [{
    "index": 0,
    "delta": {"content": " Hello"},
    "finish_reason": null
  }]
}
```

**Final chunk** — signals generation is complete:
```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion.chunk",
  "choices": [{
    "index": 0,
    "delta": {},
    "finish_reason": "stop"
  }]
}
```

### Getting usage data from streams

By default, streaming responses don't include token counts. To get them, pass `stream_options`:

```python
response = completion(
    model="openai/gpt-4o",
    messages=[...],
    stream=True,
    stream_options={"include_usage": True},
)
```

This adds an **extra chunk** after the final choice chunk with an empty `choices` array and the usage data:

```json
{
  "choices": [],
  "usage": {
    "prompt_tokens": 25,
    "completion_tokens": 42,
    "total_tokens": 67
  }
}
```

**Important:** If the stream is interrupted, this final usage chunk may never arrive.

### Anthropic events under the hood

Anthropic uses a different event structure (which LiteLLM normalizes for you):

| Event | Purpose | Key data |
|-------|---------|----------|
| `message_start` | Opens the message | Full message object, `usage.input_tokens` |
| `content_block_start` | Starts a content block | Block type (text, tool_use) |
| `content_block_delta` | Token(s) arrive | `delta.text` with the new content |
| `content_block_stop` | Ends a content block | |
| `message_delta` | Final update | `stop_reason`, `usage.output_tokens` |
| `message_stop` | Stream complete | |

You don't need to handle these directly — LiteLLM normalizes everything to the OpenAI chunk format. But knowing this helps when debugging provider-specific behavior.

### How LiteLLM normalizes

Regardless of provider, every chunk from LiteLLM has:
- `chunk.choices[0].delta.content` — the token text (or `None`)
- `chunk.choices[0].delta.role` — set on the first chunk
- `chunk.choices[0].finish_reason` — `None` until the final chunk

This means the same streaming code works for OpenAI, Anthropic, Gemini, and every other LiteLLM-supported provider.

---

## 4. TTFT and Generation Metrics

Two metrics define streaming performance. Understanding them helps you choose models, optimize prompts, and set user expectations.

### TTFT (Time to First Token)

The time between sending the request and receiving the first content token. This is what users perceive as "response time" — it's the gap between pressing Enter and seeing the first word.

```
[Send request] ----TTFT----> [First token] --generation--> [Last token]
                  0.3-1s                     5-15s
```

**What affects TTFT:**
- **Input length** — more tokens to process before generation starts (roughly linear)
- **Model size** — larger models take longer to prefill
- **Server load** — queue time during peak hours
- **Geography** — 100-300ms network round-trip from distant regions

### TPS (Tokens Per Second)

Generation throughput after the first token. Relatively constant per model, regardless of input length.

**How to calculate:**

```
TPS = output_tokens / (total_time - TTFT)
```

Example: 200 tokens, TTFT 0.5s, total time 3.0s → TPS = 200 / 2.5 = 80 tokens/sec

### Measuring in code

```python
import time

start = time.monotonic()
first_token_time = None
token_count = 0

for chunk in response:
    content = chunk.choices[0].delta.content
    if content:
        if first_token_time is None:
            first_token_time = time.monotonic()
        token_count += 1
        print(content, end="", flush=True)

total_time = time.monotonic() - start
ttft = first_token_time - start
generation_time = total_time - ttft
tps = token_count / generation_time if generation_time > 0 else 0

print(f"\nTTFT: {ttft*1000:.0f}ms | TPS: {tps:.1f} | Total: {total_time:.1f}s")
```

### Typical ranges

| Model | TTFT | TPS | 500-token response |
|-------|------|-----|-------------------|
| GPT-4o-mini | 0.2-0.5s | 80-120 | ~4-7s |
| GPT-4o | 0.3-1.0s | 50-80 | ~7-11s |
| Claude Haiku 3.5 | 0.2-0.5s | 80-120 | ~4-7s |
| Claude Sonnet 4 | 0.3-1.2s | 50-70 | ~8-11s |
| Gemini 2.0 Flash | 0.1-0.4s | 100-150 | ~3-5s |

*Values vary significantly by server load, input size, and region.*

### Why TTFT matters most for chat UIs

Users judge responsiveness by TTFT, not total generation time. A response that starts appearing in 0.3s feels instant even if it takes 10s to complete. A response that takes 3s of silence before starting feels slow even if the total time is only 4s.

This is why model selection for chat UIs should weigh TTFT heavily — a faster TTFT model might generate slower overall but feel more responsive.

---

## 5. Building a Streaming Chat

Combining streaming with multi-turn conversation creates an interactive chat experience. The core pattern is a loop: read input → stream response → record history → repeat.

### The chat loop

```python
messages = []
system_prompt = "You are a helpful assistant."

while True:
    user_input = input("\nYou: ")
    if user_input.lower() in ("quit", "exit", "/bye"):
        break

    messages.append({"role": "user", "content": user_input})

    # Stream the response
    response = completion(
        model=model,
        messages=messages,
        system=system_prompt,
        stream=True,
    )

    print("\nAssistant: ", end="")
    full_response = ""
    for chunk in response:
        content = chunk.choices[0].delta.content
        if content:
            full_response += content
            print(content, end="", flush=True)
    print()

    # Add assistant response to history
    messages.append({"role": "assistant", "content": full_response})
```

### Accumulating the response

The critical step is building `full_response` by concatenating every delta. This is necessary because:
1. You need the complete text to append to the messages history for the next turn
2. The model sees the full history on each call — missing any of it breaks the conversation

### Conversation state

The `messages` list grows with each turn. As discussed in Module 04, the model is stateless — every call sends the entire history. Token usage (and cost) grows with conversation length. For long conversations, you'd eventually need to truncate or summarize — but that's Module 09's topic.

---

## 6. Streaming vs Non-Streaming

Streaming is not always the right choice. Here's when to use each.

### When to stream

- **Chat and interactive UIs** — always. Users expect real-time feedback.
- **Any user-facing response** — the perceived speed difference is significant
- **Long-form generation** — watching 2000 tokens appear is better than a 25-second blank screen

### When NOT to stream

- **Batch processing** — no user watching, streaming adds complexity for no benefit
- **JSON/structured output** — you need the full response before parsing. A partial JSON string is unparseable.
- **Tool calls** — the model's tool call arguments arrive in fragments across chunks. Most frameworks accumulate internally and present the complete call.
- **Testing and debugging** — complete responses are easier to inspect and log
- **Simple backend operations** — when you just need the text and don't care about perceived latency

### Trade-off comparison

| Aspect | Non-streaming | Streaming |
|--------|--------------|-----------|
| Code complexity | Simple — one response object | More complex — iterate chunks, accumulate |
| Error handling | Clean — success or failure | Messy — can fail mid-response |
| Token counting | In the response's usage object | Requires `stream_options` or manual counting |
| Cancellation | Not possible mid-generation | Can stop iteration early (client-side) |
| Perceived latency | Full wait time | TTFT only |
| Best for | Batch, background, structured | Chat, interactive, user-facing |

### The same call, both ways

```python
# Non-streaming — simple and clean
response = completion(model=model, messages=messages)
text = response.choices[0].message.content
tokens = response.usage.total_tokens

# Streaming — more code, but responsive
full_text = ""
for chunk in completion(model=model, messages=messages, stream=True):
    content = chunk.choices[0].delta.content
    if content:
        full_text += content
        print(content, end="", flush=True)
```

---

## 7. Error Handling in Streams

Streaming introduces a unique error challenge: failures can happen **after** you've already started receiving and displaying content.

### Pre-stream errors

These occur before the first chunk arrives — during connection setup. They behave identically to non-streaming errors:

```python
try:
    response = completion(model=model, messages=messages, stream=True)
    for chunk in response:  # error may happen HERE too
        ...
except litellm.AuthenticationError:
    print("Bad API key")
except litellm.RateLimitError:
    print("Rate limited")
```

Authentication errors, invalid model names, and rate limits are caught before streaming begins. These are the same errors you learned to handle in Module 04.

### Mid-stream errors

These are unique to streaming. The server starts sending tokens, then something goes wrong:

- **Network disconnection** — the connection drops
- **Server error** — the provider's server crashes mid-generation
- **Timeout** — the server stops sending chunks (read timeout fires)

The result: you have a **partial response** — some tokens arrived, but the generation is incomplete.

```python
full_response = ""
try:
    for chunk in response:
        content = chunk.choices[0].delta.content
        if content:
            full_response += content
            print(content, end="", flush=True)
except Exception as e:
    print(f"\n\n[Stream interrupted: {e}]")
    print(f"[Partial response: {len(full_response)} chars received]")
```

### What to do with partial content

It depends on your application:
- **Chat UI** — show what you have, let the user see it was interrupted, offer to retry
- **Data extraction** — discard and retry (partial data is usually unusable)
- **Long-form writing** — save partial content, retry for continuation

### Retry considerations

Unlike non-streaming, retrying a mid-stream failure means **re-generating from scratch** — there's no way to resume from where the stream broke. The model generates the response fresh each time (it's stateless), so the retry may produce a different response.

### The usage data risk

If you're using `stream_options={"include_usage": True}`, remember that the usage chunk arrives **last**. If the stream is interrupted, you never get usage data. For accurate cost tracking with streaming, consider counting tokens manually during iteration as a fallback.

---

## 8. Async Streaming

So far, all examples have been synchronous — the code blocks while waiting for each chunk. This is fine for CLI tools and scripts, but for web servers handling many concurrent users, you need async streaming.

### Why async matters for web servers

A synchronous streaming call blocks the thread for the entire duration (5-15 seconds). In a web server handling 100 concurrent users, that's 100 blocked threads. Async streaming uses a single thread with non-blocking I/O — each stream yields control while waiting for the next chunk.

### The async pattern

```python
import litellm

async def stream_response(messages):
    response = await litellm.acompletion(
        model="openai/gpt-4o",
        messages=messages,
        stream=True,
    )

    full_text = ""
    async for chunk in response:
        content = chunk.choices[0].delta.content
        if content:
            full_text += content
            yield content  # for FastAPI StreamingResponse, etc.
```

### When to use async vs sync

| Use case | Recommendation |
|----------|---------------|
| CLI tools, scripts | Sync — simpler, no event loop needed |
| Jupyter notebooks | Sync — notebooks have their own async issues |
| FastAPI / aiohttp web servers | Async — non-blocking, handles concurrency |
| Multiple concurrent API calls | Async — can stream from multiple models simultaneously |
| Single-user desktop app | Sync — async adds complexity without benefit |

For this module's project, we use sync streaming. Async is covered here for awareness — when you build web apps with streaming LLMs (Module 21), you'll use the async pattern.

### The minimal difference

The code difference is small:

```python
# Sync
response = completion(model=model, messages=messages, stream=True)
for chunk in response:
    ...

# Async
response = await acompletion(model=model, messages=messages, stream=True)
async for chunk in response:
    ...
```

Same chunk format, same delta access, same content extraction. The async version just doesn't block the thread while waiting.

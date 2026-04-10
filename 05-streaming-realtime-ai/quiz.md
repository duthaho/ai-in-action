# Module 05: Streaming & Real-Time AI — Quiz

Test your understanding. Try answering before revealing the answer.

---

### Q1: Streaming makes responses arrive faster — true or false?

<details>
<summary>Answer</summary>

False. Streaming does not speed up generation — the model produces tokens at the same rate regardless. What streaming changes is **perceived latency**: users see the first token in 0.3-1s instead of waiting 5-15s for the complete response. Total generation time is the same. Streaming improves the experience, not the speed.
</details>

---

### Q2: What is TTFT and what factors affect it?

<details>
<summary>Answer</summary>

TTFT (Time to First Token) is the time between sending the request and receiving the first content token. It's what users perceive as "response time." Factors that affect it: (1) input length — more tokens to process before generation starts, (2) model size — larger models take longer to prefill, (3) server load — queue time during peak hours, (4) geographic distance — network round-trip time to the API server.
</details>

---

### Q3: How do you extract text from a stream chunk?

<details>
<summary>Answer</summary>

Access `chunk.choices[0].delta.content`. The key difference from non-streaming is the field name: streaming uses `delta` (incremental token), non-streaming uses `message` (full text). The `delta.content` may be `None` on the first chunk (which only sets `delta.role`) and on the final chunk (which sets `finish_reason`). Always check `if content:` before using it.
</details>

---

### Q4: When should you use non-streaming instead of streaming?

<details>
<summary>Answer</summary>

Use non-streaming for: (1) batch/background processing where no user is watching, (2) structured output like JSON that needs to be complete before parsing, (3) tool calls where arguments arrive fragmented, (4) testing and debugging where complete responses are easier to inspect, (5) simple backend operations where perceived latency doesn't matter. Streaming adds complexity — only use it when the UX benefit justifies it.
</details>

---

### Q5: What happens if a stream errors mid-response? How does this differ from non-streaming?

<details>
<summary>Answer</summary>

In streaming, you may have already received and displayed partial content before the error occurs. You end up with an incomplete response. In non-streaming, you get either the full response or an error — never partial content. This makes streaming error handling harder: you must decide what to do with the partial content (show it? discard it?), and retrying means regenerating from scratch since there's no way to resume a broken stream.
</details>

---

### Q6: How do you get token usage from a streaming response?

<details>
<summary>Answer</summary>

Pass `stream_options={"include_usage": True}` in the API call. This adds an extra chunk at the end of the stream with `choices: []` and a `usage` object containing `prompt_tokens`, `completion_tokens`, and `total_tokens`. Important caveat: if the stream is interrupted, this final usage chunk never arrives. For reliable cost tracking, consider counting tokens manually during iteration as a fallback.
</details>

---

### Q7: A streaming response took 3.5s total, TTFT was 0.5s, and generated 240 tokens. What's the TPS?

<details>
<summary>Answer</summary>

TPS = output_tokens / (total_time - TTFT) = 240 / (3.5 - 0.5) = 240 / 3.0 = 80 tokens per second. The TTFT is excluded because generation hasn't started yet during that phase — the model is processing the input. TPS measures only the generation throughput.
</details>

---

### Q8: Why does streaming add complexity to error handling compared to regular calls?

<details>
<summary>Answer</summary>

Three reasons: (1) Errors can occur at two points — before streaming starts (same as non-streaming) or mid-stream after partial content has been delivered, requiring different handling strategies. (2) Partial responses are neither complete nor empty — the application must decide what to do with incomplete content. (3) Usage data (token counts) arrives in the final chunk, so interrupted streams lose cost tracking information. The try/except must wrap the iteration loop, not just the initial call.
</details>

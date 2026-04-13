# Module 11 Quiz: Building AI Agents

Test your understanding. Try answering before revealing the answer.

---

### Q1: What is the core ReAct loop, in one sentence?

<details>
<summary>Answer</summary>
The agent observes the current state, thinks about what to do next, takes an action (typically a tool call), observes the result, and repeats until it decides it has enough information to answer. The loop runs through four phases — observe, think, act, observe result — and each iteration produces a new assistant message (containing the tool call) and a new tool result message that is appended to the scratchpad before the next LLM call.
</details>

---

### Q2: Why does an agent need a `final_answer` tool (or equivalent) instead of just returning plain text?

<details>
<summary>Answer</summary>
A sentinel tool gives the agent an explicit, unambiguous way to signal termination. The loop code detects the tool name and exits cleanly, and the tool's structured arguments — answer and sources — become the return value of the agent function. Plain-text exits are brittle: the model may stop emitting tool calls too early and return a partial answer, or it may keep looping indefinitely because the loop code has no reliable signal to stop.
</details>

---

### Q3: Name three failure modes of a naive agent loop and how to mitigate each.

<details>
<summary>Answer</summary>

- **Infinite loop (same tool, same args repeated)** — the model gets stuck calling the same tool with identical arguments and never converges. Mitigation: detect repeated (tool name, args) pairs and break the loop, returning an error or the best answer so far.
- **Runaway cost** — an uncapped loop can execute dozens of steps, consuming tokens and incurring API costs far beyond expectations. Mitigation: enforce a `max_steps` cap and optionally a token budget check; if either limit is reached, exit with whatever the agent has accumulated so far.
- **Hallucinated or malformed tool arguments** — the model emits arguments that fail JSON parsing or violate the tool's schema, causing an unhandled exception. Mitigation: wrap every tool dispatch in a try/except block, format the exception as a tool result message, and send it back to the agent so it can recover and try a corrected call.

</details>

---

### Q4: How does the scratchpad grow, and why does this matter for cost?

<details>
<summary>Answer</summary>
Every step appends two messages to the scratchpad: an assistant message containing the tool call and a tool result message containing the output. Token usage therefore grows roughly linearly with step count. Because each new LLM call re-sends the entire prior scratchpad, a ten-step agent incurs roughly ten times the per-step token cost compared with a single call. Keeping steps focused and tool results concise directly reduces total cost. This connects to the context budget formula from Module 09: available tokens shrink with every step, and a long-running agent can exhaust the context window entirely if results are verbose.
</details>

---

### Q5: When should you pick an agent over a static chain or workflow?

<details>
<summary>Answer</summary>
Agents shine when the number of steps is unknown in advance and the next step depends on an intermediate result — for example, a research task where the agent must decide how many searches to run based on what it finds. Use a static chain when the workflow is fully known at design time: it is cheaper, faster, and far easier to debug. Agents buy flexibility but pay in cost, latency, and debugging difficulty. If you can write down every step before running it, a static chain is the right choice; if you cannot, an agent earns its complexity.
</details>

---

### Q6: What makes a good tool description vs a bad one?

<details>
<summary>Answer</summary>
A good description states what the tool does, what inputs it accepts, and what it returns — in one or two sentences, written for the model reader. A bad description is vague ("performs a search"), omits the return format, or leaks irrelevant implementation details such as the underlying API client or internal variable names. The agent reads tool descriptions the way a developer reads a function signature: it needs just enough information to decide when to call the tool and how to construct valid arguments — nothing more, nothing less.
</details>

---

### Q7: How does Module 07 (RAG) relate to Module 11 — is RAG "just a tool" for an agent?

<details>
<summary>Answer</summary>
Yes. An agent can expose retrieval as a tool (`search_docs(query)`) and decide for itself when to consult its knowledge base, how many times to call it, and with what queries. This "retrieval as a tool" pattern gives the agent full control over when retrieval happens, enabling it to search multiple times with refined queries if the first result is insufficient. The opposite pattern — static RAG — always retrieves once up front regardless of need, which wastes tokens when retrieval is unnecessary and cannot adapt when a single retrieval pass is not enough.
</details>

---

### Q8: Why is logging every tool call essential during agent development?

<details>
<summary>Answer</summary>
Agent failures almost always manifest as strange sequences of tool calls — searching five times for the same thing, calling the wrong tool, or getting stuck — that are completely invisible unless you log them. A readable trace that records the step number, tool name, arguments, and a result summary is the primary debugging tool for agent work. Unlike conventional bugs in source code, agent logic lives in the prompts and the model's choices; your source code only defines what tools are available, not when or how they are used. Traces are the only way to inspect that decision-making after the fact, making structured logging non-optional for any agent you intend to understand or improve.
</details>

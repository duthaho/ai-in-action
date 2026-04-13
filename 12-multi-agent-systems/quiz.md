# Module 12 Quiz: Multi-Agent Systems

Test your understanding. Try answering before revealing the answer.

---

### Q1: What problem does splitting one LLM call into multiple specialized agents solve?

<details>
<summary>Answer</summary>
A single agent trying to research, write, and critique at once suffers from prompt bloat and conflicting instructions. Splitting into specialists gives each agent one job with one clear prompt, and outputs can be structured and inspected between steps. The cost is extra API calls; the benefit is better, more debuggable outputs.
</details>

---

### Q2: Sketch the three patterns (orchestrator+specialists, handoff, debate/critique) in one sentence each.

<details>
<summary>Answer</summary>

- **Orchestrator+specialists:** a coordinator calls each specialist and collects results.
- **Handoff:** each agent's output is the next agent's input; the flow is a chain.
- **Debate/critique:** a generator produces, a critic reviews, the generator revises based on feedback; repeat until convergence.

</details>

---

### Q3: Why pass Pydantic models between agents instead of raw strings?

<details>
<summary>Answer</summary>
Typed boundaries force each agent to produce parseable, versioned output. They make debugging easier (you inspect structured data), catch schema drift early, and document the contract between agents. This is the structured-output pattern from Module 08 applied at the boundary between agents rather than at the boundary between the model and your application code.
</details>

---

### Q4: What makes a "useful" critique in a critique loop?

<details>
<summary>Answer</summary>
A useful critique is specific (names a concrete issue), actionable (suggests a fix), and bounded (not a full rewrite). A vague critique like "make it better" wastes a round and risks drift. A bounded, specific critique converges in 2-3 rounds for most tasks; open-ended critiques can loop indefinitely without meaningful improvement.
</details>

---

### Q5: Name three stop conditions for a critique/revision loop.

<details>
<summary>Answer</summary>

- **verdict = approved** — the critic explicitly signs off on the output.
- **max rounds reached** — a hard cap prevents runaway cost regardless of critic verdict.
- **no change between rounds** — convergence or stagnation; continuing would not improve the result.

</details>

---

### Q6: When is a multi-agent system the wrong tool?

<details>
<summary>Answer</summary>
When the task is cheap, one-shot, or latency-sensitive. Multi-agent multiplies cost and time — worth it for complex generation tasks with quality requirements, overkill for lookups, simple transforms, or any single-step problem. If a single well-crafted prompt can produce a good enough result in one call, adding agents adds overhead without adding value.
</details>

---

### Q7: How does Module 12 relate to Module 11 — is every specialist its own ReAct agent?

<details>
<summary>Answer</summary>
Not necessarily. A specialist can be a single LLM call with a focused prompt (simplest, cheapest), or itself a full ReAct agent (the Module 11 pattern) when it needs tools to do its job. Module 12's Blog Post Writer uses single-call specialists; a more advanced setup might give the researcher web-search tools, turning it into a nested agent. The choice depends on whether the specialist needs to make decisions about which tools to call — if not, a single call is the right default.
</details>

---

### Q8: What forward link does Module 12 make to Module 13 (workflows)?

<details>
<summary>Answer</summary>
When the pipeline is fully known ahead of time — researcher then writer then critic then done, with no conditional branching decided by an LLM — it is really a static workflow, not a dynamic multi-agent system. Module 13 covers workflows and chains: the static cousin where control flow is explicit in code, not decided at runtime by a model. The distinction matters because static workflows are cheaper, faster, and easier to test; reaching for a dynamic multi-agent system is only justified when the sequence of steps cannot be determined until execution time.
</details>

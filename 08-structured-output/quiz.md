# Module 08: Structured Output — Quiz

Test your understanding. Try answering before revealing the answer.

---

### Q1: What are the three approaches to getting structured output from an LLM, and how do they differ in reliability?

<details>
<summary>Answer</summary>
1. Prompt-only: instruct the LLM via prompt to return JSON. Lowest reliability — output may include markdown fences, extra text, or malformed JSON. Works with all providers. 2. JSON mode: provider flag guarantees syntactically valid JSON. Medium reliability — valid JSON but no schema enforcement, so field names, types, and structure may vary between calls. 3. Schema-constrained: JSON schema defines exact output shape, and token generation is constrained to match. Highest reliability — output conforms to schema by construction. Requires provider support (OpenAI native, others via LiteLLM).
</details>

---

### Q2: What does JSON mode guarantee, and what does it NOT guarantee?

<details>
<summary>Answer</summary>
JSON mode guarantees the response is syntactically valid JSON parseable by json.loads — no markdown fences, no preamble text, no trailing explanation. It does NOT guarantee schema conformance: field names may vary ("product_name" vs "name"), types may differ (price as string vs number), fields may be missing or extra, and nested structure may differ from expectations. You still need validation after JSON parsing.
</details>

---

### Q3: Why use Pydantic models instead of raw dictionaries for structured LLM output?

<details>
<summary>Answer</summary>
Pydantic provides four benefits: 1. Type safety — fields are typed (str, int, list[str]), catching type mismatches automatically. 2. Validation — constraints like ge=1, le=5 or Literal enums are enforced on parse. 3. JSON schema generation — model_json_schema() produces the schema for response_format, so the model definition IS the schema definition (single source of truth). 4. Serialization — model_dump() gives you a clean dict, model_dump_json() gives JSON. Raw dicts have none of these — you'd write validation logic by hand for every field.
</details>

---

### Q4: Describe the extract-validate-retry loop. Why is feeding errors back to the LLM effective?

<details>
<summary>Answer</summary>
The loop: 1. Send prompt + schema to LLM, get JSON response. 2. Parse JSON. 3. Validate with Pydantic. 4. If valid, return typed object. 5. If invalid, append the LLM's response and the validation error as messages, then retry (up to max_retries, typically 3). Feeding errors back works because the LLM can read its own previous output alongside the specific error message ("rating must be >= 1 and <= 5, got 0") and produce a corrected version. Most validation errors are fixed on the first retry because the LLM understands what went wrong.
</details>

---

### Q5: When would you choose prompt-only structured output over schema-constrained?

<details>
<summary>Answer</summary>
Prompt-only is appropriate for: quick prototyping where you're exploring what to extract, one-off scripts where 90% reliability is acceptable, providers that don't support JSON mode or schema constraints, and situations where the output structure is intentionally flexible (you want the LLM to decide what fields are relevant). Schema-constrained is better for production pipelines, repeated extractions, and any case where downstream code depends on a specific structure.
</details>

---

### Q6: What is "hallucinated fields" in the context of structured extraction, and how do you mitigate it?

<details>
<summary>Answer</summary>
Hallucinated fields occur when the LLM invents data not present in the source text — for example, extracting a phone number from text that contains no phone number, or generating a salary range for a job posting that doesn't mention compensation. Mitigations: explicit instructions ("only extract information stated in the text, use null for missing fields"), using Optional types in your Pydantic model so null is a valid value, and post-extraction verification for critical fields by checking if the extracted value actually appears in the source text.
</details>

---

### Q7: How does constrained decoding work at a conceptual level?

<details>
<summary>Answer</summary>
During normal LLM generation, the model considers all possible next tokens at each step. With constrained decoding, the provider maintains a state machine tracking the current position in the JSON schema. At each generation step, tokens that would violate the schema are masked out (probability set to zero). For example, after generating {"rating":, only digit tokens 1-5 are allowed if the schema specifies an integer with ge=1, le=5. After {"sentiment": ", only tokens forming allowed enum values are permitted. The result: every generated sequence is guaranteed to be valid JSON conforming to the schema.
</details>

---

### Q8: How does structured output relate to tool use (Module 06)?

<details>
<summary>Answer</summary>
Tool use IS structured output. When an LLM calls a function in Module 06, it produces JSON matching the function's parameter schema — this is the same mechanism as response_format with a JSON schema. The difference is context: tool use produces structured JSON as an intermediate step (the tool call arguments), while structured output produces it as the final response. Both use schema-constrained generation, both benefit from good field descriptions, and both can fail in the same ways (hallucinated values, enum mismatches). Skills from Module 08 (validation, retry, prompt engineering) directly improve tool use reliability.
</details>

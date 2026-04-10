# Module 08 — Structured Output

Getting LLMs to produce reliable, parseable structured data — JSON objects, typed fields, enums — instead of free-form text.

| Detail        | Value                                          |
|---------------|------------------------------------------------|
| Level         | Intermediate                                   |
| Time          | ~2 hours                                       |
| Prerequisites | Module 04 (The AI API Layer), Module 06 (Tool Use & Function Calling) |

## What you'll build

After reading this module, head to [`project/`](project/) to build a **Data Extractor** — a CLI tool that takes unstructured text (product reviews, job postings, contact info) and extracts structured JSON using schema-constrained output with Pydantic validation and retry patterns.

---

## Table of Contents

1. [Why Structured Output?](#1-why-structured-output)
2. [Approaches to Structured Output](#2-approaches-to-structured-output)
3. [JSON Mode](#3-json-mode)
4. [Schema-Constrained Output](#4-schema-constrained-output)
5. [Validation & Retry Patterns](#5-validation--retry-patterns)
6. [Prompt Engineering for Structure](#6-prompt-engineering-for-structure)
7. [Common Patterns & Pitfalls](#7-common-patterns--pitfalls)
8. [Structured Output in the AI Stack](#8-structured-output-in-the-ai-stack)

---

## 1. Why Structured Output?

LLMs are text generators. Left to their own devices, they produce prose — sentences, paragraphs, explanations. That is ideal for human readers but completely useless for software that needs to parse, store, route, or validate the output.

Consider the difference:

**Free-form response:**
```
The product is called "iPhone 16 Pro", it costs $999, and the display is 6.3 inches wide.
```

**Structured response:**
```json
{
  "product_name": "iPhone 16 Pro",
  "price_usd": 999,
  "display_inches": 6.3
}
```

The first response requires brittle regex, fragile NLP parsing, and extensive error handling. The second is immediately ready for a database insert, an API response, or a downstream pipeline. Structured output is the bridge that connects LLMs to software systems.

### Use cases

Structured output is the right approach whenever the LLM's response needs to be consumed programmatically:

- **Data extraction** — pull fields from unstructured text (product reviews, contracts, emails, receipts)
- **API response generation** — produce JSON that conforms to an API schema, ready to serve to clients
- **Form parsing** — extract structured fields from freeform form submissions or voice transcripts
- **Content classification** — categorize text into typed enums (sentiment, topic, priority, intent)
- **Agent tool dispatch** — format the arguments for a tool call (Module 06 uses this extensively)
- **ETL pipelines** — transform raw text documents into typed records for database ingestion

### The reliability spectrum

Not all structured output approaches are equal. There is a clear hierarchy based on reliability:

| Approach | Mechanism | Reliability | When output is malformed |
|---|---|---|---|
| **Prompt-only** | Ask the LLM to "respond in JSON" | Low | Frequently — prose leaks in, fields are missing |
| **JSON mode** | Provider enforces valid JSON at the token level | Medium | Rare — structure is guaranteed, schema is not |
| **Schema-constrained** | Provider enforces the exact schema at the token level | High | Very rare — field names, types, and constraints are enforced |

### Why this matters for AI engineering

In production AI systems, reliability is not optional. A data extraction pipeline that fails 5% of the time requires extensive error handling, monitoring, and human review. One that fails 0.1% of the time can run autonomously.

The progression from prompt-only → JSON mode → schema-constrained is the progression from "demo" to "production." Understanding when each is appropriate — and how to validate and recover when they fail — is a core skill in AI engineering.

---

## 2. Approaches to Structured Output

There are three main approaches to getting structured output from LLMs. Each has different tradeoffs around reliability, flexibility, and provider support.

### Comparison

| | **Prompt-only** | **JSON mode** | **Schema-constrained** |
|---|---|---|---|
| **How it works** | Instruction in prompt: "respond as JSON" | Provider constrains token generation to valid JSON | Provider constrains token generation to match a specific JSON schema |
| **Reliability** | Low — LLMs frequently add prose, miss fields, or produce malformed JSON | Medium — JSON structure is guaranteed, field names and types are not | High — field names, types, required fields, and enum values are enforced |
| **Flexibility** | Maximum — any output shape | High — any valid JSON object | Medium — output must conform to the supplied schema |
| **Provider support** | Universal | OpenAI, Anthropic, Google, most providers | OpenAI (full), Anthropic (full), Google (full), others vary |
| **Validation needed** | Always — parse + schema check + semantic check | Yes — schema check + semantic check | Minimal — schema is enforced, but semantic correctness still needs checking |

### When to use prompt-only

Prompt-only is appropriate for prototyping or when the provider does not support JSON mode. Never use it in production pipelines that depend on reliable parsing. The failure rate is too high and too unpredictable — output format varies with model version, prompt phrasing, and input content.

### When to use JSON mode

JSON mode is a reasonable choice when you need valid JSON but the exact schema may vary, or when you are working with a provider that does not support full schema constraints. It eliminates the most common failure mode (malformed JSON) while retaining flexibility. You still need to validate the schema yourself and handle missing or unexpected fields.

### When to use schema-constrained output

Schema-constrained output is the production default. Use it whenever you have a fixed schema you need the LLM to conform to. The provider-side enforcement eliminates an entire class of failures and reduces the complexity of your validation layer. The tradeoff is that you must define the schema upfront, which requires knowing what fields you need.

### LiteLLM unifies the interface

LiteLLM provides a consistent interface across providers for both JSON mode and schema-constrained output. The same code works regardless of whether you are calling Claude, GPT-4o, or Gemini:

```python
from litellm import completion
from pydantic import BaseModel

class Product(BaseModel):
    name: str
    price_usd: float

# JSON mode — valid JSON guaranteed, schema is not enforced
response = completion(
    model="anthropic/claude-sonnet-4-20250514",
    messages=messages,
    response_format={"type": "json_object"},
)

# Schema-constrained — full schema enforcement
response = completion(
    model="anthropic/claude-sonnet-4-20250514",
    messages=messages,
    response_format=Product,
)
```

Passing a Pydantic model class directly as `response_format` enables schema-constrained output. LiteLLM converts the Pydantic model to a JSON schema and passes it to the provider in the appropriate format.

---

## 3. JSON Mode

JSON mode is the first tier of structured output enforcement. The provider constrains the token generation process to ensure the output is valid JSON — but makes no guarantees about the schema, field names, or field types.

### How it works

At the token level, the provider uses constrained decoding: after each token is generated, only tokens that could continue a valid JSON structure are considered valid next tokens. This makes it structurally impossible to generate invalid JSON. The model cannot "decide" to add a prose sentence before or after the JSON object.

```python
import json
from litellm import completion

messages = [
    {"role": "system", "content": "Extract product info as JSON."},
    {"role": "user", "content": "The new iPhone 16 Pro costs $999 and has a 6.3-inch display."},
]

response = completion(
    model="anthropic/claude-sonnet-4-20250514",
    messages=messages,
    response_format={"type": "json_object"},
)

data = json.loads(response.choices[0].message.content)
```

### What JSON mode guarantees

- **Valid JSON** — the output is always parseable with `json.loads()`
- **An object** — the top-level value is always a JSON object `{}`, not an array or scalar
- **No prose contamination** — no "Here is the JSON:" prefix or "Let me know if this helps" suffix

### What JSON mode does NOT guarantee

- **Field names** — the model chooses the field names; "product_name" today might be "name" or "productName" tomorrow
- **Field presence** — required fields may be absent if the model decides they are not applicable
- **Field types** — a price might be `"$999"` (string) instead of `999` (number)
- **Enum values** — a sentiment field might be `"positive"`, `"good"`, or `"favorable"` depending on the input
- **Nested structure** — the model decides how to nest objects and arrays; the structure can vary across calls

### The "mention JSON" gotcha

Most providers require that the word "JSON" appear somewhere in the prompt (system or user message) when JSON mode is enabled. Without it, some providers return an error or fall back to free-form text. Always include it explicitly:

```python
{"role": "system", "content": "Extract product info as JSON."}
# or
{"role": "user", "content": "Return the result as a JSON object."}
```

This is a quirk of how providers validate the request, not a prompt engineering technique. One mention is sufficient.

---

## 4. Schema-Constrained Output

Schema-constrained output is JSON mode with the schema also enforced at the token level. The provider knows exactly which fields are required, what types they must be, and which values are valid for enum fields — and enforces all of this during generation.

### How constrained decoding works

During generation, the provider maintains a representation of where the output currently is within the schema. At each step, only tokens that advance the output toward a valid, schema-conforming JSON object are permitted. This means:

- If a field is defined as `integer`, the model cannot emit a string value for it
- If a field is defined as `Literal["positive", "negative", "neutral"]`, the model cannot produce `"good"` or `"bad"`
- If a field is marked required, the model must emit it — it cannot skip it

The model retains its natural language reasoning capabilities; constrained decoding only filters its output tokens, not its internal reasoning.

### Defining a schema with Pydantic

Pydantic models are the standard way to define schemas for structured output in Python. Each field gets a type annotation and an optional `Field()` with a description:

```python
from pydantic import BaseModel, Field
from typing import Literal, Optional

class ProductReview(BaseModel):
    product_name: str = Field(description="Name of the product being reviewed")
    rating: int = Field(ge=1, le=5, description="Rating from 1 to 5 stars")
    sentiment: Literal["positive", "negative", "neutral", "mixed"] = Field(
        description="Overall sentiment of the review"
    )
    pros: list[str] = Field(description="List of positive points mentioned")
    cons: list[str] = Field(description="List of negative points mentioned")
    summary: str = Field(description="One-sentence summary of the review")
```

The `description` argument on each field is critical — it becomes part of the prompt context that guides the model when filling that field.

### Generating and inspecting the schema

You can inspect the JSON schema that Pydantic will send to the provider:

```python
import json
print(json.dumps(ProductReview.model_json_schema(), indent=2))
```

Output:
```json
{
  "properties": {
    "product_name": {"description": "Name of the product being reviewed", "title": "Product Name", "type": "string"},
    "rating": {"description": "Rating from 1 to 5 stars", "maximum": 5, "minimum": 1, "title": "Rating", "type": "integer"},
    "sentiment": {"description": "Overall sentiment of the review", "enum": ["positive", "negative", "neutral", "mixed"], ...},
    ...
  },
  "required": ["product_name", "rating", "sentiment", "pros", "cons", "summary"],
  "title": "ProductReview",
  "type": "object"
}
```

### Using with LiteLLM

Pass the Pydantic model class directly as `response_format`. LiteLLM handles schema serialization and provider-specific formatting:

```python
from litellm import completion

response = completion(
    model="anthropic/claude-sonnet-4-20250514",
    messages=[
        {"role": "system", "content": "Extract structured information from the product review."},
        {"role": "user", "content": review_text},
    ],
    response_format=ProductReview,
)

# Parse the response into the Pydantic model
review = ProductReview.model_validate_json(response.choices[0].message.content)
print(review.rating)       # int, guaranteed 1-5
print(review.sentiment)    # one of the four enum values
print(review.pros)         # list[str]
```

### Schema features that work well

- **Primitive types** — `str`, `int`, `float`, `bool` are universally supported and reliably enforced
- **Enums with `Literal`** — constrain a field to a fixed set of string values; highly reliable
- **Required vs optional** — marking fields as `Optional[str] = None` correctly allows absence
- **Nested objects** — embed one Pydantic model inside another for hierarchical data
- **Lists** — `list[str]`, `list[SomeModel]` work well for extracting variable-length collections
- **Field constraints** — `ge`, `le`, `min_length`, `max_length` are enforced at the schema level

### Schema features to use carefully

- **Union types** — `str | int` can confuse constrained decoding on some providers; prefer explicit types
- **Very deep nesting** — schemas with 4+ levels of nesting may cause generation to slow or fail on some providers
- **Extremely large schemas** — 20+ top-level fields can cause the model to lose track of context; consider breaking into sub-schemas (see Section 7)

---

## 5. Validation & Retry Patterns

Even with schema-constrained output, validation is not optional. The provider enforces structure, but it cannot enforce meaning. A rating of 3 when the review clearly says "best product ever" is structurally valid but semantically wrong.

### Three layers of validation

| Layer | What it checks | How to implement |
|---|---|---|
| **JSON parsing** | Is the output valid JSON? | `json.loads()` — with JSON mode, this should always pass |
| **Schema validation** | Do field names, types, and required fields match? | `Model.model_validate_json()` — Pydantic raises `ValidationError` on failure |
| **Semantic validation** | Does the output make sense given the input? | Custom logic — check consistency, cross-field constraints, confidence scores |

With schema-constrained output, the first two layers are largely handled by the provider. Your code should still wrap parsing in try/except, because edge cases exist (provider bugs, network issues, partial responses). The third layer is always your responsibility.

### The extract-validate-retry loop

```
┌─────────────────────────────────────────────────────────┐
│                    Input text                           │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│              LLM extraction call                        │
│         (schema-constrained or JSON mode)               │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
               ┌─────────────────┐
               │  Parse & validate│
               └────────┬────────┘
                        │
              ┌─────────┴──────────┐
              │                    │
         Valid?                  Invalid?
              │                    │
              ▼                    ▼
     ┌────────────────┐   ┌──────────────────────┐
     │  Return result │   │  Feed error back to  │
     └────────────────┘   │  LLM in next message │
                          └──────────┬───────────┘
                                     │
                                     ▼
                           ┌──────────────────┐
                           │   Retry (max N)  │
                           └──────────┬───────┘
                                      │
                            ┌─────────┴──────────┐
                            │                    │
                       Retries left?       Max retries hit?
                            │                    │
                            ▼                    ▼
                    (back to validate)   ┌───────────────┐
                                         │ Graceful      │
                                         │ degradation   │
                                         └───────────────┘
```

### Feeding errors back to the LLM

The most effective retry strategy is to tell the LLM exactly what went wrong and ask it to fix it. This is far more effective than simply retrying with the same prompt:

```python
from pydantic import ValidationError
import json

def extract_with_retry(text: str, model_class, max_retries: int = 3):
    messages = [
        {"role": "system", "content": "Extract structured data from the input text."},
        {"role": "user", "content": text},
    ]

    for attempt in range(max_retries):
        response = completion(
            model="anthropic/claude-sonnet-4-20250514",
            messages=messages,
            response_format=model_class,
        )
        raw = response.choices[0].message.content

        try:
            result = model_class.model_validate_json(raw)

            # Semantic validation — example: check for obvious inconsistency
            if hasattr(result, "rating") and hasattr(result, "sentiment"):
                if result.rating >= 4 and result.sentiment == "negative":
                    raise ValueError(
                        f"Inconsistent: rating {result.rating}/5 but sentiment is 'negative'"
                    )

            return result

        except (ValidationError, ValueError) as e:
            if attempt == max_retries - 1:
                raise  # Re-raise on last attempt

            # Feed the error back so the LLM can correct itself
            messages.append({"role": "assistant", "content": raw})
            messages.append({
                "role": "user",
                "content": f"That response had an error: {e}\n\nPlease try again with the corrected output.",
            })

    raise RuntimeError("Max retries exceeded")
```

### Max retries and graceful degradation

When all retries are exhausted, you have three options depending on your pipeline's requirements:

- **Raise and fail** — appropriate when partial data is worse than no data (financial records, medical data)
- **Return partial result** — return whatever fields parsed successfully, mark failed fields as `None`; appropriate when some data is better than none
- **Log and skip** — record the failure with the original text for human review; appropriate in batch pipelines where humans audit failures

---

## 6. Prompt Engineering for Structure

Schema-constrained output offloads structural enforcement to the provider, but the prompt still determines the quality of the extracted content. A well-engineered prompt produces accurate, complete, semantically correct output. A poor prompt produces valid-schema output that is still wrong.

### System prompt for extraction

The system prompt should tell the model its role, the extraction task, and how to handle ambiguity:

```python
system_prompt = """You are a data extraction assistant. Your job is to extract structured information from unstructured text.

Rules:
- Extract only information that is explicitly stated in the text. Do not infer or fabricate values.
- If a field's value is not mentioned in the text, use null for optional fields.
- For list fields, include all matching items mentioned; use an empty list if none are mentioned.
- For enum fields, choose the value that best matches the text's meaning.
- Maintain the original meaning — do not summarize or reinterpret beyond what the text says.
"""
```

### Field descriptions are prompts too

The `description` argument on each Pydantic field is injected into the schema that the model sees. Write descriptions as instructions, not just labels:

```python
# Poor description — ambiguous, no guidance
rating: int = Field(description="Rating")

# Good description — tells the model exactly what to do
rating: int = Field(
    ge=1,
    le=5,
    description="Overall rating from 1 to 5 stars, where 1 is worst and 5 is best. "
                "Infer from the text's tone if not stated explicitly as a number."
)

# Poor description — doesn't explain the options
sentiment: Literal["positive", "negative", "neutral", "mixed"] = Field(
    description="The sentiment"
)

# Good description — explains when to use each value
sentiment: Literal["positive", "negative", "neutral", "mixed"] = Field(
    description="Overall emotional tone. Use 'mixed' when the text expresses both "
                "positive and negative views. Use 'neutral' for factual or balanced text."
)
```

### Few-shot examples

For complex schemas or ambiguous extraction tasks, provide one or two examples in the messages. Few-shot examples are the highest-leverage prompt engineering technique for structured output:

```python
messages = [
    {"role": "system", "content": system_prompt},
    # Example 1
    {
        "role": "user",
        "content": "I've been using this blender for 3 months. It's powerful and quiet, "
                   "but the lid leaks if you overfill it. Overall a solid 4 stars.",
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "product_name": "blender",
            "rating": 4,
            "sentiment": "mixed",
            "pros": ["powerful", "quiet"],
            "cons": ["lid leaks if overfilled"],
            "summary": "A powerful, quiet blender with a minor leaking issue when overfilled.",
        }),
    },
    # Actual input
    {"role": "user", "content": actual_review_text},
]
```

Keep examples representative of the variation in your real inputs. One example with a clearly positive review and one with a mixed review teaches the model how to handle the difference.

### Handling missing information

Explicitly instruct the model to use `null` for optional fields rather than guessing or hallucinating:

```python
# In system prompt:
"If a field cannot be determined from the text, set it to null. "
"Never invent values for fields that are not supported by the text."

# In the schema:
pros: Optional[list[str]] = Field(
    default=None,
    description="List of positive points. Set to null if no positives are mentioned."
)
```

Without this instruction, models tend to fill optional fields with plausible-sounding but fabricated content rather than leaving them empty.

---

## 7. Common Patterns & Pitfalls

### Extraction vs generation

Structured output covers two distinct tasks that require different prompting strategies:

| | **Extraction** | **Generation** |
|---|---|---|
| **Task** | Pull existing information from the input text | Create new structured content based on instructions |
| **Example** | Parse a product review into fields | Generate a product description conforming to a schema |
| **Key instruction** | "Extract only what is in the text" | "Generate content that meets these requirements" |
| **Hallucination risk** | High — model may invent fields not in the text | Low — invention is the goal |
| **Validation focus** | Faithfulness to source text | Quality and completeness of generated content |

### Hallucinated fields

**Problem:** The model invents values for fields that are not present in the source text. Common with required fields and with models trying to be helpful.

**Mitigation:**
- Instruct explicitly: "Extract only what is stated — do not infer or fabricate"
- Make fields optional (`Optional[str] = None`) when the information may not always be present
- In semantic validation, cross-check key extracted values against the original text using substring matching or embedding similarity

### Enum mapping

**Problem:** The source text uses language that is semantically correct but does not map cleanly to your enum values. "The product is decent" maps to `"neutral"` or `"mixed"`?

**Mitigation:**
- Write detailed field descriptions that explain the distinction between similar enum values
- Provide few-shot examples that demonstrate the harder mapping cases
- Consider whether your enum values are well-designed for your use case — sometimes the right fix is to add or rename enum values

### Nested structures

Nested Pydantic models work well for hierarchical data:

```python
class Address(BaseModel):
    street: Optional[str] = None
    city: str
    country: str

class ContactInfo(BaseModel):
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    address: Optional[Address] = None
```

Keep nesting to 2–3 levels. Beyond that, models struggle to maintain context across the deeply nested schema, and constrained decoding can slow significantly on some providers.

### Array extraction

For list fields, the model must decide both how many items to extract and what each item should be. Guidance helps:

```python
pros: list[str] = Field(
    description="List of specific positive points mentioned. Each item should be a "
                "distinct point, 3-10 words. Include all that are mentioned; "
                "return an empty list if none are mentioned."
)
```

Without guidance, models may merge multiple points into one long string, split single points into multiple items, or stop after one or two items even when more are present.

### Partial extraction is correct behavior, not failure

If a product review mentions no cons, `cons: []` is the correct extraction. If a job posting does not state a salary, `salary: null` is correct. Resist the temptation to treat "empty" or "null" fields as extraction failures — they are accurate representations of the source text's content.

Design your downstream systems to handle null and empty values gracefully. The extraction pipeline's job is accuracy, not completeness of every field.

### Large schemas

When a schema exceeds roughly 20 fields, several problems emerge: the model's attention is spread too thin, generation slows, and error rates increase.

For large extractions, break the schema into sub-extractions:

```python
class BasicInfo(BaseModel):
    title: str
    company: str
    location: str
    employment_type: Literal["full-time", "part-time", "contract", "internship"]

class Requirements(BaseModel):
    required_skills: list[str]
    preferred_skills: list[str]
    years_experience: Optional[int] = None
    education_level: Optional[str] = None

class Compensation(BaseModel):
    salary_min: Optional[int] = None
    salary_max: Optional[int] = None
    currency: Optional[str] = None
    benefits: list[str]

# Run three separate extractions, then combine
basic = extract(text, BasicInfo)
requirements = extract(text, Requirements)
compensation = extract(text, Compensation)
```

Each sub-extraction is focused and fast. Combine the results in application code.

---

## 8. Structured Output in the AI Stack

Structured output is not an isolated technique. It is woven through the entire AI stack — from tool use to RAG to agents. Understanding the connections helps you design systems where each component reinforces the others.

### Connection to tool use (Module 06)

Tool arguments ARE structured output. When an LLM calls a function, it generates a JSON object conforming to the function's parameter schema. This is schema-constrained output applied to tool dispatch.

The same principles apply:
- Define precise parameter schemas with descriptions
- Use enums for parameters with fixed valid values
- Validate tool arguments before executing the tool
- Retry with error feedback if argument validation fails

Structured output for data extraction and tool use for action dispatch are the same mechanism serving different purposes.

### Connection to RAG (Module 07)

RAG and structured output combine naturally for knowledge base applications that need to return typed, validated data rather than prose:

```python
class RAGResponse(BaseModel):
    answer: str = Field(description="Direct answer to the question, based only on the provided context")
    confidence: Literal["high", "medium", "low"] = Field(
        description="Confidence level: high if context directly answers the question, "
                    "medium if partially answered, low if context is tangential"
    )
    sources: list[str] = Field(description="Source document names that were used to form the answer")
    answer_found: bool = Field(description="True if the context contained sufficient information to answer")
```

Using `RAGResponse` as the `response_format` gives you a typed, validated RAG response that downstream code can route on: surface a "low confidence" answer differently, handle `answer_found=False` gracefully, and display sources without parsing prose.

### Connection to agents (Module 11)

Agents depend on structured output at every step of the reasoning loop:

- **Structured plans** — the agent's plan of action is a list of typed steps with fields for tool name, arguments, and expected output
- **Tool arguments** — each tool call is a schema-validated JSON object (see connection to Module 06 above)
- **Observations** — tool results are parsed into typed structures before being fed back to the agent
- **State** — agent memory and working state are maintained as typed Pydantic models, not free-form strings

An agent built on structured output throughout its loop is far more reliable and debuggable than one that passes raw strings between steps. You can inspect, validate, and log every transition in a typed, machine-readable format.

### Production patterns

Structured output is the enabling technology for several production AI patterns:

**Database ingestion pipelines** — extract structured records from documents, emails, or web pages and insert them directly into a database. The Pydantic model IS the table schema. Validation happens before any data touches the database.

**API response generation** — use an LLM to generate API responses that conform to a published schema. The `response_format` schema matches the API's response type exactly. No post-processing, no transformation — the LLM output is the API response.

**ETL with AI** — replace brittle regex-and-rule-based ETL steps with LLM extraction. The structured output schema defines the transformation. The retry-with-feedback loop handles edge cases that rigid rules would miss.

**Content management systems** — authors write free-form content; structured output extracts metadata (tags, categories, summary, SEO fields) automatically. Each CMS content type maps to a Pydantic model.

In all of these patterns, the same three components appear: a schema that defines the target structure, an LLM call with `response_format` set, and a validation layer that catches and recovers from failures. Master these three components and you can apply structured output reliably to any domain.

---

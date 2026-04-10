# Project: Data Extractor

Build a CLI tool that extracts structured data from unstructured text using schema-constrained LLM output with Pydantic validation and retry patterns.

## What you'll build

A data extraction pipeline that:
- Defines Pydantic models for three extraction schemas (product review, job posting, contact info)
- Sends unstructured text + JSON schema to the LLM using `response_format`
- Validates LLM output with Pydantic
- Implements extract-validate-retry loop with error feedback
- Tracks token usage and cost across extractions

## Prerequisites

- Completed reading the Module 08 README
- Python 3.11+ with project dependencies installed
- At least one LLM provider API key configured in `.env`

## How to build

Work through the steps below in order. Each step builds on the previous one.

## Steps

### Step 1: Define Pydantic schemas

Create three Pydantic models that define the extraction schemas: `ProductReview`, `JobPosting`, and `ContactInfo`. Each model uses typed fields, `Field` descriptions, `Literal` enums, and `Optional` for nullable fields.

### Step 2: Write sample texts

Define three sample unstructured texts as string constants: a product review, a job posting, and a contact info message. Each contains enough information to extract all required fields.

### Step 3: Extract structured data

Implement `extract(text, model_class, model)` — sends the text and JSON schema (derived from the Pydantic model) to the LLM using `response_format`, parses the JSON response, and returns the raw dict.

### Step 4: Validate output

Implement `validate_output(data, model_class)` — validates the dict against the Pydantic model. Returns the validated model instance or raises `ValidationError`.

### Step 5: Extract with retry

Implement `extract_with_retry(text, model_class, model, max_retries)` — the full pipeline: extract → validate → retry with error feedback. Returns a dict with the validated data, retry count, token usage, and cost.

### Step 6: Display and orchestrate

Implement `print_result(result, schema_name)` to pretty-print extraction results, and `main()` to run extraction on all three sample texts with a session summary.

## How to run

```bash
cd 08-structured-output/project
python solution.py
```

## Expected output

```
============================================================
  Data Extractor — Structured Output Demo
============================================================
  Model: anthropic/claude-sonnet-4-20250514

--- 1. Product Review Extraction ---

  Text: "I bought the SoundMax Pro wireless headphones last month..."
  Schema: ProductReview

  Extracted:
    product_name: SoundMax Pro
    rating: 4
    sentiment: mixed
    pros: ["Excellent 20-hour battery life", "Rich bass and clear mids"]
    cons: ["Uncomfortable after 2 hours", "Pricey at $199"]
    summary: Good wireless headphones with great sound and battery but comfort issues.

  Retries: 0 | Tokens: 312 in / 95 out | Cost: $0.0015

--- 2. Job Posting Extraction ---

  Text: "We're hiring! TechFlow Inc. is looking for a Senior..."
  Schema: JobPosting

  Extracted:
    title: Senior Backend Engineer
    company: TechFlow Inc.
    location: San Francisco, CA (Hybrid)
    salary_min: 145000
    salary_max: 195000
    requirements: ["5+ years Python or Go", "Distributed systems experience", ...]
    benefits: ["Unlimited PTO", "Health/dental/vision", ...]

  Retries: 0 | Tokens: 298 in / 112 out | Cost: $0.0018

--- 3. Contact Info Extraction ---

  Text: "Hey, just met Sarah Chen at the conference..."
  Schema: ContactInfo

  Extracted:
    name: Sarah Chen
    email: sarah.chen@meridian.io
    phone: 415-555-0142
    company: Meridian Design Studio
    role: Head of Product

  Retries: 0 | Tokens: 276 in / 68 out | Cost: $0.0011

============================================================
  Session Summary
============================================================
  Extractions: 3
  Total retries: 0
  Total cost: $0.0044
============================================================
```

## Stretch goals

1. **Interactive mode** — add an interactive loop where users paste text and choose a schema type for extraction
2. **Custom schema** — let users define a schema via JSON and extract using it
3. **Batch extraction** — extract from a list of texts and output results as JSONL

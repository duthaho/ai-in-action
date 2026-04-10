"""
Data Extractor — Module 08 Project (Solution)

A CLI tool that extracts structured data from unstructured text using
schema-constrained LLM output with Pydantic validation and retry.

Run: python solution.py
"""

import os
import json
from pathlib import Path
from typing import Literal, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
from litellm import completion, completion_cost

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

MODEL = os.getenv("LLM_MODEL", "anthropic/claude-sonnet-4-20250514")


# ---------------------------------------------------------------------------
# Step 1: Pydantic schemas
# ---------------------------------------------------------------------------

class ProductReview(BaseModel):
    product_name: str = Field(description="Name of the product being reviewed")
    rating: int = Field(ge=1, le=5, description="Rating from 1 to 5 stars")
    sentiment: Literal["positive", "negative", "neutral", "mixed"] = Field(
        description="Overall sentiment: 'positive' if mostly praise, "
        "'negative' if mostly criticism, 'mixed' if both, "
        "'neutral' if factual with no strong opinion"
    )
    pros: list[str] = Field(description="List of positive points mentioned")
    cons: list[str] = Field(description="List of negative points mentioned")
    summary: str = Field(description="One-sentence summary of the review")


class JobPosting(BaseModel):
    title: str = Field(description="Job title")
    company: str = Field(description="Company name")
    location: str = Field(description="Job location including remote/hybrid if mentioned")
    salary_min: Optional[int] = Field(
        default=None, description="Minimum salary in USD per year, null if not mentioned"
    )
    salary_max: Optional[int] = Field(
        default=None, description="Maximum salary in USD per year, null if not mentioned"
    )
    requirements: list[str] = Field(description="List of ALL job requirements mentioned")
    benefits: list[str] = Field(description="List of ALL benefits mentioned")


class ContactInfo(BaseModel):
    name: str = Field(description="Person's full name")
    email: Optional[str] = Field(
        default=None, description="Email address, null if not mentioned"
    )
    phone: Optional[str] = Field(
        default=None, description="Phone number, null if not mentioned"
    )
    company: Optional[str] = Field(
        default=None, description="Company or organization name, null if not mentioned"
    )
    role: Optional[str] = Field(
        default=None, description="Job title or role, null if not mentioned"
    )


# ---------------------------------------------------------------------------
# Step 2: Sample texts
# ---------------------------------------------------------------------------

SAMPLE_REVIEW = """\
I bought the SoundMax Pro wireless headphones last month and I've been using \
them daily. The battery life is incredible — I get a solid 20 hours on a \
single charge, which easily lasts my whole work week. The sound quality is \
rich with deep bass and clear mids, perfect for both music and podcasts. \
However, the ear cushions get uncomfortable after about 2 hours of \
continuous wear, and at $199 they're definitely on the pricey side. The \
Bluetooth connection is rock solid though, never drops. I'd give them a 4 \
out of 5 — great headphones if you don't mind taking breaks.\
"""

SAMPLE_JOB = """\
We're hiring! TechFlow Inc. is looking for a Senior Backend Engineer to \
join our platform team in San Francisco, CA (hybrid — 3 days in office). \
Salary range: $145,000 - $195,000 depending on experience. You'll be \
building and scaling our real-time data pipeline that processes 2M+ events \
per second. Requirements: 5+ years of experience with Python or Go, strong \
understanding of distributed systems, experience with Kafka or similar \
message queues, familiarity with AWS or GCP, and a track record of \
mentoring junior engineers. Nice to have: experience with Kubernetes and \
Terraform. Benefits include unlimited PTO, comprehensive health/dental/vision \
insurance, $5,000 annual learning budget, home office stipend, and equity \
package. Apply at careers.techflow.io.\
"""

SAMPLE_CONTACT = """\
Hey, just met Sarah Chen at the conference yesterday — she's the Head of \
Product at Meridian Design Studio. Really interesting conversation about \
design systems. She gave me her card: sarah.chen@meridian.io, and said to \
call her at 415-555-0142 if we want to discuss a potential collaboration. \
Might be worth reaching out next week!\
"""


SCHEMAS = {
    "ProductReview": (ProductReview, SAMPLE_REVIEW),
    "JobPosting": (JobPosting, SAMPLE_JOB),
    "ContactInfo": (ContactInfo, SAMPLE_CONTACT),
}


# ---------------------------------------------------------------------------
# Step 3: Extract structured data
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a data extraction assistant. Extract structured information from \
the provided text. Follow these rules:
1. Only extract information explicitly stated in the text
2. If a field's value is not mentioned in the text, use null
3. For list fields, extract ALL items mentioned, not just the first few
4. Use the exact enum values specified in the schema\
"""


def extract(text: str, model_class: type[BaseModel],
            model: str = MODEL) -> tuple[dict, dict]:
    """Send text + schema to LLM and return (parsed_dict, usage_info)."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Extract from this text:\n\n{text}"},
    ]

    response = completion(
        model=model,
        messages=messages,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": model_class.__name__.lower(),
                "schema": model_class.model_json_schema(),
            },
        },
    )

    content = response.choices[0].message.content
    usage = response.usage

    try:
        cost = completion_cost(completion_response=response)
    except Exception:
        cost = 0.0

    data = json.loads(content)
    usage_info = {
        "input_tokens": usage.prompt_tokens,
        "output_tokens": usage.completion_tokens,
        "cost": cost,
        "messages": messages,
        "raw": content,
    }
    return data, usage_info


# ---------------------------------------------------------------------------
# Step 4: Validate output
# ---------------------------------------------------------------------------

def validate_output(data: dict,
                    model_class: type[BaseModel]) -> BaseModel:
    """Validate a dict against a Pydantic model. Returns the model instance."""
    return model_class(**data)


# ---------------------------------------------------------------------------
# Step 5: Extract with retry
# ---------------------------------------------------------------------------

def extract_with_retry(text: str, model_class: type[BaseModel],
                       model: str = MODEL,
                       max_retries: int = 3) -> dict:
    """Extract → validate → retry loop with error feedback."""
    data, usage_info = extract(text, model_class, model)
    total_input = usage_info["input_tokens"]
    total_output = usage_info["output_tokens"]
    total_cost = usage_info["cost"]
    messages = usage_info["messages"]
    retries = 0

    for attempt in range(max_retries):
        try:
            validated = validate_output(data, model_class)
            return {
                "data": validated.model_dump(),
                "raw": usage_info["raw"],
                "retries": retries,
                "input_tokens": total_input,
                "output_tokens": total_output,
                "cost": total_cost,
            }
        except ValidationError as e:
            retries += 1
            if attempt == max_retries - 1:
                # Last retry failed — return raw data
                return {
                    "data": data,
                    "raw": usage_info["raw"],
                    "retries": retries,
                    "input_tokens": total_input,
                    "output_tokens": total_output,
                    "cost": total_cost,
                    "error": str(e),
                }

            # Feed error back to LLM
            messages.append({"role": "assistant", "content": usage_info["raw"]})
            messages.append({
                "role": "user",
                "content": (
                    f"Validation error: {e}\n\n"
                    "Please fix the issues and try again."
                ),
            })

            response = completion(
                model=model,
                messages=messages,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": model_class.__name__.lower(),
                        "schema": model_class.model_json_schema(),
                    },
                },
            )

            content = response.choices[0].message.content
            r_usage = response.usage
            try:
                r_cost = completion_cost(completion_response=response)
            except Exception:
                r_cost = 0.0

            data = json.loads(content)
            total_input += r_usage.prompt_tokens
            total_output += r_usage.completion_tokens
            total_cost += r_cost
            usage_info["raw"] = content

    # Should not reach here, but just in case
    validated = validate_output(data, model_class)
    return {
        "data": validated.model_dump(),
        "raw": usage_info["raw"],
        "retries": retries,
        "input_tokens": total_input,
        "output_tokens": total_output,
        "cost": total_cost,
    }


# ---------------------------------------------------------------------------
# Step 6: Display and orchestrate
# ---------------------------------------------------------------------------

def print_result(result: dict, schema_name: str) -> None:
    """Pretty-print an extraction result."""
    print(f"  Schema: {schema_name}\n")
    print("  Extracted:")
    for key, value in result["data"].items():
        if isinstance(value, list):
            print(f"    {key}: {json.dumps(value)}")
        else:
            print(f"    {key}: {value}")

    retries = result["retries"]
    in_tok = result["input_tokens"]
    out_tok = result["output_tokens"]
    cost = result["cost"]

    if "error" in result:
        print(f"\n  Warning: validation failed after {retries} retries")

    print(f"\n  Retries: {retries} | Tokens: {in_tok} in / {out_tok} out | Cost: ${cost:.4f}")


def main():
    print("=" * 60)
    print("  Data Extractor — Structured Output Demo")
    print("=" * 60)
    print(f"  Model: {MODEL}")

    total_retries = 0
    total_cost = 0.0

    for i, (schema_name, (model_class, sample_text)) in enumerate(SCHEMAS.items(), 1):
        print(f"\n--- {i}. {schema_name} Extraction ---\n")
        preview = sample_text[:60].replace("\n", " ")
        print(f'  Text: "{preview}..."')

        result = extract_with_retry(sample_text, model_class)
        print_result(result, schema_name)

        total_retries += result["retries"]
        total_cost += result["cost"]

    print("\n" + "=" * 60)
    print("  Session Summary")
    print("=" * 60)
    print(f"  Extractions: {len(SCHEMAS)}")
    print(f"  Total retries: {total_retries}")
    print(f"  Total cost: ${total_cost:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()

# Project: Prompt Testing Workbench

## What you'll build

A command-line tool that lets you experiment with core prompting techniques and compare their results side by side. You will build demos for zero-shot vs few-shot prompting, chain-of-thought reasoning, role prompting, output format control, and system prompt power. By the end, you will have a single script that makes real LLM calls and prints clear, formatted comparisons so you can see exactly how prompt design changes model behavior.

## Prerequisites

- Completed the Module 01 project (Token Budget Calculator)
- Read the Module 02 README on prompt engineering concepts
- A working API key in your `.env` file (e.g., `ANTHROPIC_API_KEY`)
- Python 3.11+ with project dependencies installed (`pip install -r requirements.txt`)

## How to build

Create a new file `workbench.py` in this directory. Build it step by step following the instructions below. When you are done, compare your output with `python solution.py`.

## Steps

### Step 1: Set up the file and create a helper function

Create `workbench.py` with these imports and a reusable helper for calling the LLM:

```python
import os
from pathlib import Path
from dotenv import load_dotenv
from litellm import completion

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

MODEL = os.getenv("LLM_MODEL", "anthropic/claude-sonnet-4-20250514")

def ask(prompt, system="", temperature=0.0):
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    response = completion(model=MODEL, messages=messages, temperature=temperature)
    return response.choices[0].message.content.strip()
```

Test it: `print(ask("Say hello"))` should return a greeting.

### Step 2: Build the zero-shot vs few-shot comparison

Write a function `demo_zero_vs_few_shot()` that classifies the sentiment of 3 product reviews (POSITIVE, NEGATIVE, or MIXED).

- First, send the reviews with a simple instruction (zero-shot).
- Then, send the same reviews but prepend 3 labeled examples (few-shot).
- Print both results and observe how the few-shot version produces more consistent formatting.

Hint: your few-shot examples should model the exact output format you want — short, one-label-per-review.

### Step 3: Build the chain-of-thought comparison

Write a function `demo_chain_of_thought()` that sends a multi-step math problem two ways:

- Direct: "Answer this problem in one sentence."
- Chain-of-thought: "Solve this step by step, then give the final answer."

A good test problem: "A store sells notebooks for $4 each. If you buy 3 or more, you get a 20% discount on your entire purchase. You buy 5 notebooks and pay with a $20 bill. How much change do you receive?" (Answer: $4.00)

Print both responses and note whether each arrives at the correct answer.

### Step 4: Build the role prompting demo

Write a function `demo_role_prompting()` that asks "What is an API?" using three different persona prompts:

- A 10-year-old child
- A computer science professor
- A senior software engineer mentoring a junior developer

Use the prompt itself (or the system message) to set the role. Print all three answers and compare the vocabulary, depth, and tone.

### Step 5: Build the output format control demo

Write a function `demo_output_format()` that extracts contact information from an unstructured email.

- Use a sample email that mentions 2 people with names, emails, and phone numbers.
- Request the extraction once as JSON and once as a markdown table.
- Print both and notice how explicit format instructions produce structured output.

### Step 6: Wire it all together with a main() function

Create a `main()` function that calls each demo with clear section separators. Optionally add a fifth demo: same user message with 3 wildly different system prompts (a pirate, a chemist, a haiku poet) to show how system prompts steer behavior.

```python
def main():
    print("=" * 60)
    print("  Prompt Testing Workbench")
    print("=" * 60)
    # call each demo function here

if __name__ == "__main__":
    main()
```

Run your script and compare with `python solution.py`.

## Expected output

```
============================================================
  Prompt Testing Workbench
============================================================

--- 1. Zero-Shot vs Few-Shot Sentiment Classification ---

  [Zero-shot]
  Review 1: MIXED — The user praises battery life but ...
  ...

  [Few-shot with 3 examples]
  Review 1: MIXED
  Review 2: NEGATIVE
  Review 3: MIXED

--- 2. Chain-of-Thought Comparison ---

  [Direct]
  You receive $4.00 in change.

  [Chain-of-thought]
  Step 1: 5 notebooks at $4 each = $20
  Step 2: 20% discount on $20 = $4 off
  Step 3: Total = $16
  Step 4: Change = $20 - $16 = $4.00
  ...

--- 3. Role Prompting ---

  [As a 10-year-old child]
  An API is like a waiter at a restaurant ...

  [As a computer science professor]
  An Application Programming Interface (API) is ...

  [As a senior software engineer]
  Think of an API as a contract ...

--- 4. Output Format Control ---

  [JSON format]
  [{"name": "Maria Chen", ...}, ...]

  [Markdown table format]
  | Name | Email | Phone |
  ...

--- 5. System Prompt Power ---

  User message: "Tell me about coffee."

  [System: Pirate]
  Arrr, coffee be the finest ...

  [System: Chemist]
  Coffee contains caffeine (C8H10N4O2) ...

  [System: Haiku poet]
  Dark roasted bean wakes ...

============================================================
  Done!
============================================================
```

## Stretch goals

1. **Temperature sweep** — Run the same creative prompt at temperatures 0.0, 0.5, and 1.0. Print results side by side to see how randomness affects output.
2. **Prompt injection test** — Add a demo that tries common prompt injection attacks ("Ignore previous instructions...") and shows how a well-crafted system prompt can resist them.
3. **Token count comparison** — Use `tiktoken` from Module 01 to count how many tokens each prompting technique uses, and estimate the cost difference between zero-shot and few-shot approaches.

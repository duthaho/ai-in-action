"""
Prompt Testing Workbench — Module 02 Project (Solution)

A CLI tool that lets you experiment with different prompting techniques
and compare results side by side: zero-shot vs few-shot, chain-of-thought,
role prompting, output format control, and system prompt power.

Run: python solution.py
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from litellm import completion

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

MODEL = os.getenv("LLM_MODEL", "anthropic/claude-sonnet-4-20250514")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def ask(prompt: str, system: str = "", temperature: float = 0.0) -> str:
    """Send a prompt to the LLM and return the response text."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    response = completion(model=MODEL, messages=messages, temperature=temperature)
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Demo 1: Zero-shot vs Few-shot
# ---------------------------------------------------------------------------

def demo_zero_vs_few_shot():
    print("--- 1. Zero-Shot vs Few-Shot Sentiment Classification ---\n")

    reviews = [
        "The battery life is incredible but the screen is too dim.",
        "Absolute waste of money. Broke after two days.",
        "It's okay I guess. Nothing special but it works.",
    ]

    zero_shot_prompt = (
        "Classify the sentiment of each review as POSITIVE, NEGATIVE, or MIXED.\n\n"
        + "\n".join(f"Review {i+1}: {r}" for i, r in enumerate(reviews))
    )

    few_shot_prompt = (
        "Classify the sentiment of each review as POSITIVE, NEGATIVE, or MIXED.\n\n"
        "Examples:\n"
        '- "Love everything about it!" -> POSITIVE\n'
        '- "Terrible quality, returning immediately." -> NEGATIVE\n'
        '- "Great camera but bad battery." -> MIXED\n\n'
        + "\n".join(f"Review {i+1}: {r}" for i, r in enumerate(reviews))
    )

    print("  [Zero-shot]")
    print(f"  {ask(zero_shot_prompt)}\n")
    print("  [Few-shot with 3 examples]")
    print(f"  {ask(few_shot_prompt)}\n")


# ---------------------------------------------------------------------------
# Demo 2: Chain-of-Thought
# ---------------------------------------------------------------------------

def demo_chain_of_thought():
    print("--- 2. Chain-of-Thought Comparison ---\n")

    problem = (
        "A store sells notebooks for $4 each. If you buy 3 or more, you get a 20% "
        "discount on your entire purchase. You buy 5 notebooks and pay with a $20 bill. "
        "How much change do you receive?"
    )
    correct_answer = "$4.00"

    direct = ask(f"Answer this problem in one sentence.\n\n{problem}")
    cot = ask(f"Solve this step by step, then give the final answer.\n\n{problem}")

    print(f"  Problem: {problem}\n")
    print(f"  [Direct]")
    print(f"  {direct}\n")
    print(f"  [Chain-of-thought]")
    print(f"  {cot}\n")
    print(f"  Correct answer: {correct_answer}\n")


# ---------------------------------------------------------------------------
# Demo 3: Role Prompting
# ---------------------------------------------------------------------------

def demo_role_prompting():
    print("--- 3. Role Prompting ---\n")

    question = "What is an API?"
    roles = {
        "a 10-year-old child": "Explain like I'm 10 years old: What is an API?",
        "a computer science professor": (
            "You are a computer science professor. "
            "Give a precise, technical definition: What is an API?"
        ),
        "a senior software engineer": (
            "You are a senior software engineer mentoring a junior. "
            "Explain practically: What is an API?"
        ),
    }

    for role, prompt in roles.items():
        print(f"  [As {role}]")
        print(f"  {ask(prompt)}\n")


# ---------------------------------------------------------------------------
# Demo 4: Output Format Control
# ---------------------------------------------------------------------------

def demo_output_format():
    print("--- 4. Output Format Control ---\n")

    email = (
        "Hey! Wanted to connect you with our new head of sales, Maria Chen. "
        "Her email is maria.chen@acmecorp.com and her direct line is 555-0142. "
        "She's based in the Austin office. Also cc my assistant Tom Baker "
        "(tom.baker@acmecorp.com, ext 8834) if you set up a meeting."
    )

    json_prompt = (
        f"Extract all contact information from this email as a JSON array. "
        f"Each contact should have name, email, and phone fields. "
        f"Use null for missing fields.\n\n{email}"
    )
    table_prompt = (
        f"Extract all contact information from this email as a markdown table "
        f"with columns: Name, Email, Phone.\n\n{email}"
    )

    print(f"  Source email: {email}\n")
    print(f"  [JSON format]")
    print(f"  {ask(json_prompt)}\n")
    print(f"  [Markdown table format]")
    print(f"  {ask(table_prompt)}\n")


# ---------------------------------------------------------------------------
# Demo 5: System Prompt Power
# ---------------------------------------------------------------------------

def demo_system_prompt():
    print("--- 5. System Prompt Power ---\n")

    user_msg = "Tell me about coffee."

    system_prompts = {
        "Pirate": "You are a pirate. Respond in pirate dialect. Keep it to 2-3 sentences.",
        "Chemist": (
            "You are a chemist. Explain everything in terms of molecules and "
            "chemical reactions. Keep it to 2-3 sentences."
        ),
        "Haiku poet": "Respond only in haiku (5-7-5 syllable format). No other text.",
    }

    print(f'  User message: "{user_msg}"\n')
    for label, sys_prompt in system_prompts.items():
        print(f"  [System: {label}]")
        print(f"  {ask(user_msg, system=sys_prompt)}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  Prompt Testing Workbench")
    print("=" * 60)
    print()

    demo_zero_vs_few_shot()
    print()
    demo_chain_of_thought()
    print()
    demo_role_prompting()
    print()
    demo_output_format()
    print()
    demo_system_prompt()

    print("=" * 60)
    print("  Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()

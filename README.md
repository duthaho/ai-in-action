# Learn AI Engineering

A hands-on, module-based curriculum for backend developers learning AI engineering — from LLM fundamentals to production-ready systems.

Each module includes deep technical explanations, working Python code, and exercises designed for engineers who already know how to build software and want to understand the AI layer.

## Modules

| # | Topic | Description |
|---|-------|-------------|
| 01 | [How LLMs Work](01-how-llms-work/) | Tokens, embeddings, attention, inference mechanics |
| 02 | [Tool Use / Function Calling](02-tool-use/) | Bridging LLMs to real-world actions via structured tool calls |

## Getting Started

### Prerequisites

- Python 3.11+
- An Anthropic API key (set in `.env` as `ANTHROPIC_API_KEY`)

### Setup

```bash
# Clone the repo
git clone <repo-url> learn-ai
cd learn-ai

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Enter a module and install its dependencies
cd 01-how-llms-work
cp .env.example .env      # add your API key
pip install -r requirements.txt

# Run the app
python app.py

# Run tests
pytest
```

Each module is self-contained with its own `requirements.txt`, `.env.example`, and tests.

## Module Structure

Every module follows a consistent layout:

```
XX-module-name/
├── README.md           # Deep-dive technical guide
├── app.py              # Working implementation
├── requirements.txt    # Module dependencies
├── .env.example        # Required environment variables
└── test_*.py           # Tests and exercises
```

## License

For personal learning use.

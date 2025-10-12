"""
llmsays: One-line LLM queries with smart hybrid routing via OpenRouter (free models).
Requires OPENROUTER_API_KEY env var (free signup at openrouter.ai).
"""

import argparse
import os

from openai import OpenAI

from .router import get_category

# Global client
_client = None


def _get_client():
    global _client
    if _client is None:
        if not os.getenv("OPENROUTER_API_KEY"):
            raise ValueError(
                "Set OPENROUTER_API_KEY env var (free at openrouter.ai/keys)."
            )
        _client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )
    return _client


# Models map
MODELS = {
    "simple": "google/gemma-3n-2b:free",
    "moderate": "google/gemma-3n-4b:free",
    "complex": "google/gemma-3-12b:free",
    "creative": "arliai/qwq-32b-rpr-v1:free",
    "tool-use": "agentica/deepcoder-14b-preview:free",
}


def llmsays(query: str, max_tokens: int = 256) -> str:
    client = _get_client()
    category = get_category(query)
    model_slug = MODELS.get(category, "google/gemma-2-9b:free")

    try:
        response = client.chat.completions.create(
            model=model_slug,
            messages=[{"role": "user", "content": query}],
            max_tokens=max_tokens,
            temperature=0.1,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        fallback_model = "google/gemma-2-9b:free"
        response = client.chat.completions.create(
            model=fallback_model,
            messages=[{"role": "user", "content": query}],
            max_tokens=max_tokens,
            temperature=0.1,
        )
        return response.choices[0].message.content.strip()


# CLI
def cli():
    parser = argparse.ArgumentParser(
        description="llmsays: One-line LLM queries via OpenRouter."
    )
    parser.add_argument("query", help="The prompt/query for the LLM.")
    parser.add_argument(
        "--max-tokens", type=int, default=256, help="Max output tokens (default: 256)."
    )
    args = parser.parse_args()

    print(llmsays(args.query, args.max_tokens))


if __name__ == "__main__":
    cli()
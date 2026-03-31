"""
llmsays package: One-line LLM queries with prompt-tier routing and provider failover.
"""

import argparse
import os
from typing import Dict, Iterable, List, Optional, Tuple
from openai import OpenAI
from .router import get_tier

_CLIENTS: Dict[str, OpenAI] = {}

PROVIDERS = {
    "Groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "env_key": "GROQ_API_KEY",
    },
    "NIM": {
        "base_url": "https://integrate.api.nvidia.com/v1",
        "env_key": "NIVIDIA_API_KEY",
        "alt_env_key": "NVIDIA_API_KEY",
    },
    "Openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "env_key": "OPENROUTER_API_KEY",
    },
    "Fireworks": {
        "base_url": "https://api.fireworks.ai/inference/v1",
        "env_key": "FIREWORKSAI_API_KEY",
    },
    "Baseten": {
        "base_url": "https://inference.baseten.co/v1",
        "env_key": "BASETEN_API_KEY",
    },
}

MODELS = {
    "small": {
        "Groq": "openai/gpt-oss-20b",
        "NIM": "nvidia/nvidia-nemotron-nano-9b-v2",
        "Openrouter": "stepfun/step-3.5-flash",
        "Fireworks": "fireworks/models/qwen3-8b",
        "Baseten": "openai/gpt-oss-120b",
    },
    "medium": {
        "Groq": "qwen/qwen3-32b",
        "NIM": "nvidia/llama-3.3-nemotron-super-49b-v1.5",
        "Openrouter": "google/gemini-3-flash-preview",
        "Fireworks": "fireworks/models/qwen3-vl-30b-a3b-instruct",
        "Baseten": "MiniMaxAI/MiniMax-M2.5",
    },
    "large": {
        "Groq": "llama-3.3-70b-versatile",
        "NIM": "nvidia/nemotron-3-super-120b-a12b",
        "Openrouter": "deepseek/deepseek-v3.2-speciale",
        "Fireworks": "fireworks/models/qwen3-vl-235b-a22b-thinking",
        "Baseten": "moonshotai/Kimi-K2.5",
    },
    "extra_large": {
        "Groq": "openai/gpt-oss-120b",
        "NIM": "nvidia/llama-3.1-nemotron-ultra-253b-v1",
        "Openrouter": "anthropic/claude-opus-4.6",
        "Fireworks": "fireworks/models/qwen3-coder-480b-a35b-instruct",
        "Baseten": "zai-org/GLM-5",
    },
}


def _resolve_provider_key(provider_name: str) -> Optional[str]:
    provider = PROVIDERS[provider_name]
    key = os.getenv(provider["env_key"])
    alt_key_name = provider.get("alt_env_key")
    if not key and alt_key_name:
        key = os.getenv(alt_key_name)
    return key


def _provider_order(provider_preference: Optional[Iterable[str]] = None) -> List[str]:
    default_order = ["Groq", "NIM", "Openrouter", "Fireworks", "Baseten"]
    if not provider_preference:
        return default_order

    order: List[str] = []
    for name in provider_preference:
        for canonical in default_order:
            if canonical.lower() == str(name).strip().lower() and canonical not in order:
                order.append(canonical)
                break
    return order or default_order


def _get_client(provider_name: str) -> OpenAI:
    if provider_name in _CLIENTS:
        return _CLIENTS[provider_name]

    api_key = _resolve_provider_key(provider_name)
    if not api_key:
        raise ValueError(f"Missing API key for {provider_name}.")

    provider = PROVIDERS[provider_name]
    _CLIENTS[provider_name] = OpenAI(base_url=provider["base_url"], api_key=api_key)
    return _CLIENTS[provider_name]

def llmsays(
    query: object,
    max_tokens: int = 256,
    temperature: float = 0.1,
    provider_preference: Optional[Iterable[str]] = None,
) -> str:
    query_text = str(query).strip()
    if not query_text:
        raise ValueError("Prompt cannot be empty.")

    tier = get_tier(query_text)
    provider_errors: List[Tuple[str, str]] = []

    for provider_name in _provider_order(provider_preference):
        model_slug = MODELS[tier][provider_name]
        try:
            client = _get_client(provider_name)
            response = client.chat.completions.create(
                model=model_slug,
                messages=[{"role": "user", "content": query_text}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            content = response.choices[0].message.content
            return content.strip() if isinstance(content, str) else str(content)
        except Exception as exc:
            provider_errors.append((provider_name, str(exc)))

    failed = "; ".join(f"{name}: {err}" for name, err in provider_errors)
    raise RuntimeError(
        "All providers failed for tier "
        f"'{tier}'. Configure at least one valid API key and reachable endpoint. "
        f"Errors: {failed}"
    )

def cli():
    parser = argparse.ArgumentParser(description="llmsays CLI")
    parser.add_argument("query", help="Prompt for LLM")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens")
    parser.add_argument(
        "--providers",
        nargs="*",
        default=None,
        help="Optional provider order, e.g. --providers Groq Openrouter",
    )
    args = parser.parse_args()
    print(llmsays(args.query, max_tokens=args.max_tokens, provider_preference=args.providers))

if __name__ == "__main__":
    cli()
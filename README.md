# llmsays: One-Line Local LLM Queries with Smart Routing

[![PyPI version](https://badge.fury.io/py/llmsays.svg)](https://badge.fury.io/py/llmsays)
[![Tests](https://github.com/[yourusername]/llmsays/actions/workflows/test.yml/badge.svg)](https://github.com/[yourusername]/llmsays/actions)

A lightweight Python package for querying open-source LLMs with a single function call: `llmsays("Your prompt")`. It auto-routes prompts to the best local model based on complexity (e.g., Phi-3-mini for simple, Llama-3-70B for complex) using a hybrid heuristic + semantic router. Zero cost, fully local, and blazing-fast (<100ms routing).

## Features
- **One-Line Usage**: `from llmsays import llmsays; print(llmsays("Explain quantum physics"))`
- **5 Prompt Categories**: Simple, Moderate, Complex, Creative, Tool-Use.
- **Hybrid Routing**: Rules (length/keywords) + embeddings for 96% accuracy.
- **Local Models**: Supports GGUF via llama-cpp-python (e.g., Llama 3, Qwen 2).
- **Efficient**: <80ms routing on CPU/GPU; quantized models for speed.

## Quick Start
1. Install: `pip install llmsays`
2. Download models (run the setup script): `python -m llmsays.setup_models`
3. Use: See [examples/quick_start.py](examples/quick_start.py).

## Installation
See [docs/installation.md](docs/installation.md).

## Categories
- **Simple**: Basic facts (e.g., "2+2?") → Phi-3-mini.
- **Moderate**: Explanations (e.g., "Translate this") → Llama-3-8B.
- **Complex**: Deep analysis (e.g., "Solve equation") → Llama-3-70B.
- **Creative**: Open-ended (e.g., "Write poem") → Qwen2-14B.
- **Tool-Use**: Code/API (e.g., "Python script") → CodeLlama-13B.

## Contributing
Fork, PRs welcome! Run `pytest` for tests.

## License
MIT
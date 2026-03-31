# llmsays

One-line LLM calls with automatic prompt-tier routing and provider failover.

## Usage

```python
from llmsays import llmsays

user_prompt = input("Here goes your prompt: ")
response = llmsays(user_prompt)
print(response)
```

`llmsays()` decides prompt tier using a hybrid router powered by `sentence-transformers/paraphrase-MiniLM-L3-v2`, then picks the mapped model for each provider.

## Prompt Tiers

- `small`
- `medium`
- `large`
- `extra_large`

## Model Matrix

### small
- Groq: `openai/gpt-oss-20b`
- NIM: `nvidia/nvidia-nemotron-nano-9b-v2`
- Openrouter: `stepfun/step-3.5-flash`
- Fireworks: `fireworks/models/qwen3-8b`
- Baseten: `openai/gpt-oss-120b`

### medium
- Groq: `qwen/qwen3-32b`
- NIM: `nvidia/llama-3.3-nemotron-super-49b-v1.5`
- Openrouter: `google/gemini-3-flash-preview`
- Fireworks: `fireworks/models/qwen3-vl-30b-a3b-instruct`
- Baseten: `MiniMaxAI/MiniMax-M2.5`

### large
- Groq: `llama-3.3-70b-versatile`
- NIM: `nvidia/nemotron-3-super-120b-a12b`
- Openrouter: `deepseek/deepseek-v3.2-speciale`
- Fireworks: `fireworks/models/qwen3-vl-235b-a22b-thinking`
- Baseten: `moonshotai/Kimi-K2.5`

### extra_large
- Groq: `openai/gpt-oss-120b`
- NIM: `nvidia/llama-3.1-nemotron-ultra-253b-v1`
- Openrouter: `anthropic/claude-opus-4.6`
- Fireworks: `fireworks/models/qwen3-coder-480b-a35b-instruct`
- Baseten: `zai-org/GLM-5`

## API Keys

Set at least one API key. Multiple keys enable automatic provider failover.

- `GROQ_API_KEY`
- `OPENROUTER_API_KEY`
- `NIVIDIA_API_KEY`
- `FIREWORKSAI_API_KEY`
- `BASETEN_API_KEY`

## CLI

```bash
llmsays "Explain transformers in simple terms"
llmsays "Analyze this legal clause" --providers Groq Openrouter
```

## Development

```bash
pytest
```

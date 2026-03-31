# Installation

1. `pip install llmsays`
2. Package dependencies are installed automatically from `pyproject.toml`:
	- `openai`, `groq`, `openrouter`, `baseten`, `fireworks-ai`
	- `semantic-router[hybrid]`, `sentence-transformers`
3. Set at least one provider key (more keys enables failover):
	- `export GROQ_API_KEY=...`
	- `export OPENROUTER_API_KEY=...`
	- `export NIVIDIA_API_KEY=...`
	- `export FIREWORKSAI_API_KEY=...`
	- `export BASETEN_API_KEY=...`
4. Use as a one-liner:
	- `from llmsays import llmsays`
	- `print(llmsays("Explain quantum tunneling in simple words"))`
5. Run tests:
	- `pytest -q`
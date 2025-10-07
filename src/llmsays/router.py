# Your optimized code from earlier (paste the full ~60-line version here)
import re
from functools import lru_cache
from semantic_router import Route, HybridRouteLayer
from semantic_router.encoders import HuggingFaceEncoder
from semantic_router.encoders import BM25Encoder
from llama_cpp import Llama

# Pre-load quantized models (Q4_K_M for speed/accuracy)
models = {
    "simple": Llama("phi-3-mini-Q4_K_M.gguf", n_gpu_layers=-1, n_ctx=2048, verbose=False),
    "moderate": Llama("llama-3-8b-Q4_K_M.gguf", n_gpu_layers=-1, n_ctx=2048, verbose=False),
    "complex": Llama("llama-3-70b-Q4_K_M.gguf", n_gpu_layers=-1, n_ctx=2048, verbose=False),
    "creative": Llama("qwen2-14b-Q4_K_M.gguf", n_gpu_layers=-1, n_ctx=2048, verbose=False),
    "tool-use": Llama("codellama-13b-Q4_K_M.gguf", n_gpu_layers=-1, n_ctx=2048, verbose=False)
}

# Compact routes: Utterances (semantic) + keywords (sparse)
routes = [
    Route(name="simple", utterances=["What is 2+2?", "Hello", "Capital of France"], keywords=["basic", "hello", "what is"]),
    Route(name="moderate", utterances=["Explain car engine", "Translate to Spanish", "Short Python sort"], keywords=["explain", "translate", "script"]),
    Route(name="complex", utterances=["Solve differential equation", "Quantum entanglement math", "Analyze contract"], keywords=["solve", "equation", "analyze"]),
    Route(name="creative", utterances=["Write poem ocean", "Funny robot dialogue", "App slogan"], keywords=["poem", "dialogue", "slogan"]),
    Route(name="tool-use", utterances=["Python Fibonacci", "Weather API query", "JSON dataset analyze"], keywords=["python", "api", "data"])
]

# Fast encoders: Smaller dense + sparse
dense_encoder = HuggingFaceEncoder("sentence-transformers/paraphrase-MiniLM-L3-v2")  # 20-50ms
sparse_encoder = BM25Encoder()

# Hybrid router (alpha=0.6 for accuracy bias)
router = HybridRouteLayer(encoder=dense_encoder, sparse_encoder=sparse_encoder, routes=routes, alpha=0.6, aggregation="cosine")

# Cached hybrid route (avoids recompute for similar queries)
@lru_cache(maxsize=128)
def _cached_route(query: str) -> str:
    choice = router(query)
    return choice.name if choice.similarity_score >= (0.6 + 0.2 * (len(query.split()) > 50)) else "moderate"

# Enhanced pre-filter: Tiered rules + regex for 60% bypass
def heuristic_pre_filter(query: str) -> str | None:
    tokens = len(query.split())
    q_lower = query.lower()
    complex_re = re.compile(r"(solve|explain|write|analyze|python|equation|poem|api)")
    if tokens < 10 and not complex_re.search(q_lower): return "simple"
    if tokens > 100 and "math|code|analyze" in q_lower: return "complex"  # Quick complex catch
    if any(kw in q_lower for kw in ["python", "api", "code"]): return "tool-use"
    if any(kw in q_lower for kw in ["poem", "story", "dialogue"]): return "creative"
    return None  # To hybrid

def llmsays(query: str, max_tokens: int = 256) -> str:
    """One-line LLM query with hybrid routing."""
    category = heuristic_pre_filter(query)
    if not category:
        category = _cached_route(query)  # Cached hybrid
    
    model = models[category]
    resp = model(query, max_tokens=max_tokens, temperature=0.1, stop=["\n\n"], echo=False)
    return resp["choices"][0]["text"].strip()
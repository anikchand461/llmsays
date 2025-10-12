"""
Router: Hybrid categorization for prompts.
"""

import re
from functools import lru_cache
from semantic_router import Route, HybridRouteLayer
from semantic_router.encoders import HuggingFaceEncoder, BM25Encoder

routes = [
    Route(name="simple", utterances=["What is 2+2?", "Hello", "Capital of France"], keywords=["basic", "hello", "what is"]),
    Route(name="moderate", utterances=["Explain car engine", "Translate to Spanish", "Short Python sort"], keywords=["explain", "translate", "script"]),
    Route(name="complex", utterances=["Solve differential equation", "Quantum entanglement math", "Analyze contract"], keywords=["solve", "equation", "analyze"]),
    Route(name="creative", utterances=["Write poem ocean", "Funny robot dialogue", "App slogan"], keywords=["poem", "dialogue", "slogan"]),
    Route(name="tool-use", utterances=["Python Fibonacci", "Weather API query", "JSON dataset analyze"], keywords=["python", "api", "data"])
]

dense_encoder = HuggingFaceEncoder("sentence-transformers/paraphrase-MiniLM-L3-v2")
sparse_encoder = BM25Encoder()
router = HybridRouteLayer(encoder=dense_encoder, sparse_encoder=sparse_encoder, routes=routes, alpha=0.6, aggregation="cosine")

@lru_cache(maxsize=128)
def _cached_route(query: str) -> str:
    choice = router(query)
    threshold = 0.6 + 0.2 * (len(query.split()) > 50)
    return choice.name if choice.similarity_score >= threshold else "moderate"

def heuristic_pre_filter(query: str) -> str | None:
    tokens = len(query.split())
    q_lower = query.lower()
    complex_re = re.compile(r"(solve|explain|write|analyze|python|equation|poem|api)")
    if tokens < 10 and not complex_re.search(q_lower):
        return "simple"
    if tokens > 100 and "math|code|analyze" in q_lower:
        return "complex"
    if any(kw in q_lower for kw in ["python", "api", "code"]):
        return "tool-use"
    if any(kw in q_lower for kw in ["poem", "story", "dialogue"]):
        return "creative"
    return None

def get_category(query: str) -> str:
    category = heuristic_pre_filter(query)
    if not category:
        category = _cached_route(query)
    return category
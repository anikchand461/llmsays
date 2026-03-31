"""
Router: Hybrid categorization for prompts.
"""

import re
from functools import lru_cache
from typing import Any, Optional
from semantic_router import Route
from semantic_router.encoders import HuggingFaceEncoder, BM25Encoder

try:
    from semantic_router import HybridRouteLayer as _HybridRouterClass
except ImportError:
    from semantic_router import HybridRouter as _HybridRouterClass

routes = [
    Route(
        name="small",
        utterances=["What is 2+2?", "Hello", "Capital of France"],
        keywords=["basic", "hello", "what is"],
    ),
    Route(
        name="medium",
        utterances=["Explain car engine", "Translate to Spanish", "Summarize this text"],
        keywords=["explain", "translate", "summary"],
    ),
    Route(
        name="large",
        utterances=["Solve differential equation", "Analyze contract", "Design a database schema"],
        keywords=["solve", "equation", "analyze", "design"],
    ),
    Route(
        name="extra_large",
        utterances=[
            "Write a full production architecture with tradeoffs",
            "Perform deep legal risk analysis",
            "Create a multi-stage reasoning plan",
        ],
        keywords=["architecture", "risk", "multi-stage", "research"],
    ),
]


@lru_cache(maxsize=1)
def _get_router() -> Any:
    dense_encoder = HuggingFaceEncoder("sentence-transformers/paraphrase-MiniLM-L3-v2")
    sparse_encoder = BM25Encoder()
    kwargs = {
        "encoder": dense_encoder,
        "sparse_encoder": sparse_encoder,
        "routes": routes,
        "alpha": 0.6,
        "aggregation": "cosine",
    }

    try:
        return _HybridRouterClass(**kwargs)
    except TypeError:
        kwargs.pop("aggregation", None)
        try:
            return _HybridRouterClass(**kwargs)
        except TypeError:
            return _HybridRouterClass(routes=routes, encoder=dense_encoder, sparse_encoder=sparse_encoder)

@lru_cache(maxsize=128)
def _cached_route(query: str) -> str:
    threshold = 0.58 + 0.2 * (len(query.split()) > 70)
    try:
        choice = _get_router()(query)
        choice_name = getattr(choice, "name", None)
        similarity = getattr(choice, "similarity_score", None)
        if choice_name in {"small", "medium", "large", "extra_large"}:
            if similarity is None or similarity >= threshold:
                return choice_name
    except Exception:
        pass
    return "medium"

def heuristic_pre_filter(query: str) -> Optional[str]:
    tokens = len(query.split())
    q_lower = query.lower()
    heavy_re = re.compile(
        r"(prove|derivation|theorem|architecture|multi-step|legal|contract|research|design)",
    )
    code_re = re.compile(r"(python|api|code|debug|sql|dataset|json)")
    medium_re = re.compile(r"(explain|summarize|translate|compare|describe)")
    extreme_re = re.compile(r"(full|deep|global|tradeoff|tradeoffs|comprehensive|end-to-end)")

    if heavy_re.search(q_lower) and extreme_re.search(q_lower):
        return "extra_large"
    if tokens > 120 or (tokens > 60 and heavy_re.search(q_lower)):
        return "extra_large"
    if medium_re.search(q_lower):
        return "medium"
    if tokens < 10 and not heavy_re.search(q_lower) and not code_re.search(q_lower):
        return "small"
    if heavy_re.search(q_lower) or (tokens > 35 and code_re.search(q_lower)):
        return "large"
    if tokens <= 35:
        return "medium"
    return None

def get_tier(query: str) -> str:
    tier = heuristic_pre_filter(query)
    if not tier:
        tier = _cached_route(query)
    return tier


def get_category(query: str) -> str:
    """Backward-compatible alias for older integrations."""
    return get_tier(query)
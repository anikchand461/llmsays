import pytest

from llmsays.router import get_category, heuristic_pre_filter


@pytest.fixture
def sample_queries():
    return {
        "simple": "deviation is 2+2?",
        "moderate": "Explain car engine",
        "complex": "Solve dy/dx = x^2 + y^2",
        "creative": "Write a poem about ocean",
        "tool-use": "Python Fibonacci up to 100",
    }


def test_heuristic_pre_filter(sample_queries):
    assert heuristic_pre_filter(sample_queries["simple"]) == "simple"
    assert heuristic_pre_filter(sample_queries["tool-use"]) == "tool-use"
    assert heuristic_pre_filter("Hello world") == "simple"  # Short, no complex keywords


def test_get_category(sample_queries):
    assert get_category(sample_queries)["simple"] == "simple"
    assert get_category(sample_queries)["complex"] == "complex"
    # Note: Semantic may vary; mock embeddings in CI if flaky


def test_fallback_category():
    assert get_category("Random unrelated query") == "moderate"
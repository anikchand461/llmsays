import pytest
from llmsays.router import get_category, heuristic_pre_filter

@pytest.fixture
def sample_queries():
    return {
        "small": "What is 2+2?",
        "medium": "Explain how a car engine works",
        "large": "Analyze this contract and list legal risks",
        "extra_large": "Create a full architecture with tradeoffs and deep legal analysis for a global payments platform"
    }

def test_heuristic_pre_filter(sample_queries):
    assert heuristic_pre_filter(sample_queries["small"]) == "small"
    assert heuristic_pre_filter(sample_queries["large"]) == "large"
    assert heuristic_pre_filter("Hello world") == "small"

def test_get_category(sample_queries):
    assert get_category(sample_queries["small"]) == "small"
    assert get_category(sample_queries["extra_large"]) == "extra_large"

def test_fallback_category():
    assert get_category("Explain cloud service models") == "medium"
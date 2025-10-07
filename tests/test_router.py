"""
Basic tests for llmsays router.
"""

import pytest
from llmsays import llmsays

def test_llmsays_simple():
    """Test simple prompt routes correctly."""
    response = llmsays("What is 2+2?")
    assert len(response) > 0  # Basic check; expand with mocks if needed

def test_llmsays_complex():
    """Test complex prompt."""
    response = llmsays("Solve dy/dx = x^2 + y^2")
    assert "solution" in response.lower() or "integral" in response.lower()  # Loose check

def test_llmsays_edge_case():
    """Test empty/edge prompt."""
    response = llmsays("Hello")
    assert len(response) > 0
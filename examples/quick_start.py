#!/usr/bin/env python3
"""
Quick start: Run `python examples/quick_start.py` after setting OPENROUTER_API_KEY.
"""

import os

from llmsays import llmsays

# Set your key (or export OPENROUTER_API_KEY=sk-or-...)
os.environ["OPENROUTER_API_KEY"] = "your_key_here"  # Replace

if __name__ == "__main__":
    print("Simple:", llmsays("What is 2+2?"))
    print("Complex:", llmsays("Solve dy/dx = x^2 + y^2"))
    print("Creative:", llmsays("Write a short poem about stars"))
    print(
        "Tool-Use:", llmsays("Write Python code for Fibonacci sequence")
    )
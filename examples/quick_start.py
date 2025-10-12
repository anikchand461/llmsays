#!/usr/bin/env python3
"""
Quick start example for llmsays.
"""

import os
from llmsays import llmsays

os.environ["OPENROUTER_API_KEY"] = "your_key_here"  # Replace or set env

if __name__ == "__main__":
    print("Simple:", llmsays("What is 2+2?"))
    print("Complex:", llmsays("Solve dy/dx = x^2 + y^2"))
    print("Creative:", llmsays("Write a poem about stars"))
    print("Tool-Use:", llmsays("Write Python Fibonacci code"))
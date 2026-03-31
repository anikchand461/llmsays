#!/usr/bin/env python3
"""
Quick start example for llmsays.
"""

import os
from llmsays import llmsays

# Set one or more provider keys for automatic failover.
os.environ["GROQ_API_KEY"] = "your_key_here"
# os.environ["OPENROUTER_API_KEY"] = "your_key_here"
# os.environ["NIVIDIA_API_KEY"] = "your_key_here"
# os.environ["FIREWORKSAI_API_KEY"] = "your_key_here"
# os.environ["BASETEN_API_KEY"] = "your_key_here"

if __name__ == "__main__":
    print("Small:", llmsays("What is 2+2?"))
    print("Medium:", llmsays("Explain how a car engine works"))
    print("Large:", llmsays("Analyze this contract and summarize legal risk"))
    print(
        "Extra Large:",
        llmsays("Create a full production architecture with tradeoffs for a global fintech app"),
    )
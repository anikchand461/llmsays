"""
Quick start example for llmsays.
"""

from llmsays import llmsays

if __name__ == "__main__":
    # Simple query (routes to Phi-3-mini)
    print("Simple:", llmsays("What is 2+2?"))
    
    # Complex query (routes to Llama-3-70B)
    print("Complex:", llmsays("Solve dy/dx = x^2 + y^2"))
    
    # Creative query (routes to Qwen2-14B)
    print("Creative:", llmsays("Write a poem about stars"))
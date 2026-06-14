from scre.query_aware_reducer import SCRE
from pathlib import Path
import json

def check_sample(file_path, query, window=0):
    engine = SCRE()
    text = Path(file_path).read_text()
    
    # Run reduction
    result = engine.reduce(
        text=text,
        query=query,
        max_sentences=4,
        context_window=window
    )
    
    print(f"\n--- FILE: {file_path} ---")
    print(f"Query: {query}")
    print(f"Original Characters: {len(text)}")
    print(f"Reduced Characters: {len(result['context'])}")
    print(f"Reduction Ratio: {result['metadata']['reduction_ratio']:.2%}")
    print("\n--- PRESERVED CONTEXT (What the LLM sees) ---")
    print(result['context'])
    print("-" * 40)

if __name__ == "__main__":
    # Test 1: Structured Data
    check_sample(
        "data/large_Sample.txt", 
        "Who manages Project Atlas?", 
        window=0
    )
    
    # Test 2: Narrative with Reasoning (needs context window)
    check_sample(
        "data/sample.txt", 
        "What is the goal of Part I of the book?", 
        window=1
    )
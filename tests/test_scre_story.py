import pytest
from scre.query_aware_reducer import SCRE
from pathlib import Path

FILE_PATH = str(Path(__file__).parent.parent / "data" / "new_sample.txt")


@pytest.mark.skipif(
    not Path(FILE_PATH).exists(),
    reason=f"fixture file missing: {FILE_PATH}",
)
def test_story_reduction():
    # Initialize the SCRE library
    engine = SCRE()

    # Load the new story
    file_path = FILE_PATH
    story_text = Path(file_path).read_text()
    
    # Define the complex query
    query = "What was the mission of Echo and how did Arin help fulfill it?"
    
    # Execute reduction with a context window of 1 to preserve narrative "connective tissue"
    result = engine.reduce(
        text=story_text,
        query=query,
        max_sentences=5,
        context_window=1
    )
    
    print(f"--- SCRE TEST: {Path(file_path).name} ---")
    print(f"Query: {query}")
    print("-" * 50)
    
    # Metrics
    original_chars = len(story_text)
    reduced_chars = len(result['context'])
    reduction_pct = result['metadata']['reduction_ratio']
    
    print(f"Original Size: ~{original_chars // 4} tokens")
    print(f"Reduced Size:  ~{reduced_chars // 4} tokens")
    print(f"Token Reduction: {reduction_pct:.2%}")
    print("-" * 50)
    
    print("PRESERVED CONTEXT FOR LLM:")
    print(result['context'])
    
    # Validation of meaning
    print("-" * 50)
    if "Echo" in result['context'] and "Arin" in result['context'] and "Restore" in result['context']:
        print("RESULT: Meaning Preserved. Key entities and actions are present.")
    else:
        print("RESULT: Meaning possibly fragmented. Adjust weights or window.")

if __name__ == "__main__":
    test_story_reduction()
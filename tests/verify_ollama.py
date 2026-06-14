import json
import argparse
from pathlib import Path
from scre.scre_pipeline import run_scre
from scre.scre_answer_engine import answer_question

def verify_with_ollama(story_path: str, question: str):
    """
    Test script to verify SCRE using a story file and a local Ollama model.
    Compares Original Context vs Reduced Context and checks LLM output.
    """
    # 1. Load the Story
    if not Path(story_path).exists():
        print(f"Error: {story_path} not found.")
        return
    
    story_text = Path(story_path).read_text()
    
    # --- CONFIGURATION ---
    # Set your local Ollama model name here
    MY_OLLAMA_MODEL = "gemma4:e2b" 
    
    # 2. Define the Ask (Targeting the Knowledge Graph path)
    print(f"Testing SCRE with Model: {MY_OLLAMA_MODEL}")
    print(f"File: {story_path}")
    print(f"Question: {question}\n")

    # 3. Get Baseline Answer (Full Context)
    print("Requesting answer from Full Context (Baseline)...")
    full_result = answer_question(
        reduced_context=story_text,
        user_question=question,
        mode="ollama",
        model_name=MY_OLLAMA_MODEL
    )

    # 4. Run SCRE Pipeline (Reduced Context)
    print("Requesting answer from Reduced Context (SCRE)...")
    result = run_scre(
        document_text=story_text,
        user_question=question,
        max_sentences=6,
        answer_mode="ollama",
        context_window=1,
        model_name=MY_OLLAMA_MODEL
    )

    # 5. Report Comparisons
    metrics = result["metrics"]
    print("--- CONTEXT REDUCTION METRICS ---")
    print(f"Original Context: {metrics['original_chars']} characters")
    print(f"Reduced Context:  {metrics['reduced_chars']} characters")
    print(f"Reduction Ratio:  {metrics['reduction_ratio']:.2%}")
    print("-" * 30)

    print("\n--- ORIGINAL CONTEXT (TRUNCATED FOR DISPLAY) ---")
    print(story_text[:1000] + "..." if len(story_text) > 1000 else story_text)

    print("\n--- REDUCED CONTEXT SENT TO LLM ---")
    print(result["reduced_context"])
    
    print("\n--- OLLAMA MODEL RESPONSES COMPARISON ---")
    print(f"FULL CONTEXT ANSWER (Baseline):\n{full_result['answer']}")
    print("-" * 20)
    print(f"REDUCED CONTEXT ANSWER (SCRE):\n{result['answer']['answer']}")
    print("-" * 30)

if __name__ == "__main__":
    # Note: Ensure 'ollama serve' is running and the model (e.g. gemma4:e2b) is pulled.
    parser = argparse.ArgumentParser(
        description="Test SCRE dynamically with a document and a question.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "-f", "--file",
        help="Path to the text file to use as context."
    )
    parser.add_argument(
        "-q", "--question",
        help="The question to ask about the context."
    )
    args = parser.parse_args()

    file_path = args.file
    user_question = args.question

    if not file_path:
        file_path = input("Enter the path to the text file: ")

    if not user_question:
        user_question = input("Enter the question you want to ask: ")

    print("-" * 30)
    verify_with_ollama(story_path=file_path, question=user_question)
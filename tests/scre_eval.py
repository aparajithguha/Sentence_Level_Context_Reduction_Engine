from pathlib import Path

from qa_dataset import FACT_QA_DATASET, WORKFLOW_QA_DATASET, DECISION_QA_DATASET, REASONING_QA_DATASET, CAUSAL_QA_DATASET
from scre.scre_answer_engine import answer_question
from scre.scre_pipeline import run_scre


def calculate_mrs(result: dict, item: dict) -> float:
    """
    Generalized Meaning Retention Score (MRS).
    Evaluates if the expected answer and key question terms are retained in the reduced context.
    """
    context_lower = result["reduced_context"].lower()
    expected_lower = item["expected_answer"].lower()
    
    # 1. Is the expected answer present?
    ans_retention = 1.0 if expected_lower in context_lower else 0.0
    
    # 2. Are key query terms present?
    query_terms = [w.lower() for w in item["question"].split() if len(w) > 3]
    if not query_terms:
        return ans_retention
        
    term_retention = sum(1.0 for t in query_terms if t in context_lower) / len(query_terms)
    
    # Weight the expected answer more heavily (70%) than the question terms (30%)
    return (ans_retention * 0.7) + (term_retention * 0.3)


def exact_match(actual: str, expected: str) -> bool:
    a_clean = " ".join(actual.strip().lower().split())
    e_clean = " ".join(expected.strip().lower().split())
    return e_clean in a_clean or a_clean in e_clean


def print_result(item: dict, result: dict, full_answer: dict) -> bool:
    metrics = result["metrics"]
    reduced_answer = result["answer"]["answer"]
    expected_answer = item["expected_answer"]
    reduced_correct = exact_match(reduced_answer, expected_answer)
    full_correct = exact_match(full_answer["answer"], expected_answer)
    consistent = exact_match(reduced_answer, full_answer["answer"])

    print("\n" + "=" * 80)
    print(f"QUESTION: {item['question']}")
    print("=" * 80)
    print(
        "Estimated tokens:",
        metrics["original_estimated_tokens"],
        "->",
        metrics["reduced_estimated_tokens"],
        f"({metrics['reduction_ratio']:.2%} reduction)",
    )
    print("Selected sentences:", metrics["selected_sentence_count"])
    print("Full-context answer:", full_answer["answer"])
    print("Reduced-context answer:", reduced_answer)
    print("Expected:", expected_answer)
    print("Full correct:", full_correct)
    print("Reduced correct:", reduced_correct)
    print("Full vs reduced consistent:", consistent)
    
    mrs = calculate_mrs(result, item)
    print(f"Meaning Retention Score (MRS): {mrs:.2f}")
    
    print("\nReduced context:\n")
    print(result["reduced_context"])

    return reduced_correct


def run_dataset(name: str, dataset: list[dict], context_window: int) -> dict:
    correct_answers = 0
    total_reduction = 0.0
    total_items = len(dataset)

    print("\n" + "#" * 80)
    print(f"DATASET: {name}")
    print("#" * 80)

    for item in dataset:
        document = Path(item["document_path"]).read_text(encoding="utf-8")
        result = run_scre(
            document,
            item["question"],
            max_sentences=4,
            context_window=context_window,
        )
        full_answer = answer_question(
            reduced_context=document,
            user_question=item["question"],
            mode="extractive",
        )
        if print_result(item, result, full_answer):
            correct_answers += 1
        total_reduction += result["metrics"]["reduction_ratio"]

    accuracy = correct_answers / total_items
    avg_reduction = total_reduction / total_items
    
    print("\n" + "-" * 80)
    print(
        "Reduced-context accuracy:",
        f"{correct_answers}/{total_items} ({accuracy:.2%})",
    )
    print(f"Average token reduction: {avg_reduction:.2%}")
    
    return {"accuracy": accuracy, "avg_reduction": avg_reduction}


if __name__ == "__main__":
    summary = {}
    
    summary["fact"] = run_dataset(
        name="Fact Retrieval",
        dataset=FACT_QA_DATASET,
        context_window=0,
    )
    summary["reasoning"] = run_dataset(
        name="Reasoning Retrieval",
        dataset=REASONING_QA_DATASET,
        context_window=0,
    )
    summary["workflow"] = run_dataset(
        name="Workflow Retrieval",
        dataset=WORKFLOW_QA_DATASET,
        context_window=0,
    )
    summary["decision"] = run_dataset(
        name="Decision Retrieval",
        dataset=DECISION_QA_DATASET,
        context_window=0,
    )
    summary["causal"] = run_dataset(
        name="Causal Retrieval",
        dataset=CAUSAL_QA_DATASET,
        context_window=0,
    )

    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    for ds_name, stats in summary.items():
        print(f"{ds_name.capitalize()}: Acc {stats['accuracy']:.2%}, Red {stats['avg_reduction']:.2%}")

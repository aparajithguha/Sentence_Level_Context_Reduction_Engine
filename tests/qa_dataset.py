FACT_QA_DATASET = [
    {
        "document_path": "data/new_sample.txt",
        "question": "What is CAP theorem?",
        "expected_answer": "distributed systems",
    },
    {
        "document_path": "data/new_sample.txt",
        "question": "What is OAuth?",
        "expected_answer": "authorization",
    },
]

WORKFLOW_QA_DATASET = [
    {
        "document_path": "data/new_sample.txt",
        "question": "What are the stages of a RAG pipeline?",
        "expected_answer": "Embedding Generation",
    },
]

DECISION_QA_DATASET = [
    {
        "document_path": "data/new_sample.txt",
        "question": "What database was selected?",
        "expected_answer": "local",
    },
]

REASONING_QA_DATASET = [
    {
        "document_path": "data/new_test2.txt",
        "question": "Why was Ollama selected?",
        "expected_answer": "privacy",
    },
    {
        "document_path": "data/new_sample.txt",
        "question": "Why was local LLMs chosen?",
        "expected_answer": "privacy",
    },
    {
        "document_path": "data/new_sample.txt",
        "question": "How does the decision satisfy the constraint?",
        "expected_answer": "local",
    },
]

CAUSAL_QA_DATASET = [
    {
        "document_path": "data/new_sample.txt",
        "question": "Why did premature optimization matter?",
        "expected_answer": "complexity",
    },
]

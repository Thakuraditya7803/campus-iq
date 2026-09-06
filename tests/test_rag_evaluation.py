import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from qdrant_client import QdrantClient

from ai.embeddings.embeddings import EmbeddingModel
from ai.rag.rag import search_documents


client = QdrantClient(path=".qdrant")
embedding_model = EmbeddingModel()


TEST_CASES = [
    {
        "question": "What is the minimum attendance requirement?",
        "expected_chunk": 0,
    },
    {
        "question": "What happens if attendance falls below 75%?",
        "expected_chunk": 2,
    },
    {
        "question": "Can attendance be relaxed?",
        "expected_chunk": 3,
    },
    {
        "question": "What documents are required for attendance relaxation?",
        "expected_chunk": 3,
    },
    {
        "question": "What are the attendance requirements for B.Tech students?",
        "expected_chunk": 0,
    },
]


print("\n" + "=" * 70)
print("CAMPUSIQ RETRIEVAL EVALUATION")
print("=" * 70)


correct = 0


for test in TEST_CASES:

    question = test["question"]
    expected_chunk = test["expected_chunk"]

    results = search_documents(
        client,
        embedding_model,
        question,
        top_k=2
    )

    retrieved_chunks = [
        result.get("chunk_id")
        for result in results
    ]

    passed = expected_chunk in retrieved_chunks

    if passed:
        correct += 1

    print("\nQuestion:")
    print(question)

    print(f"Expected chunk: {expected_chunk}")
    print(f"Retrieved chunks: {retrieved_chunks}")

    print(f"Result: {'PASS' if passed else 'FAIL'}")


accuracy = correct / len(TEST_CASES)


print("\n" + "=" * 70)
print(f"Retrieval Accuracy: {accuracy:.2%}")
print("=" * 70)


client.close()
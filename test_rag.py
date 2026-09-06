import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from qdrant_client import QdrantClient

from ai.embeddings.embeddings import EmbeddingModel
from ai.rag.rag import search_documents, check_evidence
from ai.rag.generator import generate_answer


# Connect to Qdrant
client = QdrantClient(path=".qdrant")

# Load embedding model
embedding_model = EmbeddingModel()


# User question
query = "What happens if my attendance falls below 75%?"


print("\n" + "=" * 60)
print("USER QUESTION")
print("=" * 60)

print(query)


# Retrieve relevant documents
documents = search_documents(
    client,
    embedding_model,
    query,
    top_k=2
)


# Display retrieved documents
print("\n" + "=" * 60)
print("RETRIEVED DOCUMENTS")
print("=" * 60)

for i, document in enumerate(documents, start=1):

    print(f"\nResult {i}")
    print(f"Score: {document['score']:.4f}")
    print(f"Source: {document['source']}")
    print(f"Page: {document['page']}")
    print(f"Chunk: {document.get('chunk_id')}")


# Evidence Guard
print("\n" + "=" * 60)
print("EVIDENCE GUARD")
print("=" * 60)

evidence = check_evidence(
    documents,
    min_score=0.50,
    min_gap=0.02
)

print(
    f"Best Score    : {evidence['best_score']:.4f}"
)

print(
    f"Second Score  : {evidence['second_score']:.4f}"
)

print(
    f"Score Gap     : {evidence['score_gap']:.4f}"
)

print(
    f"Decision      : "
    f"{'PASS' if evidence['has_evidence'] else 'ABSTAIN'}"
)

print(
    f"Reason        : {evidence['reason']}"
)


# Generate answer only if evidence is strong
if evidence["has_evidence"]:

    print("\nPASS - Strong evidence found")

    print("\n" + "=" * 60)
    print("GENERATING ANSWER")
    print("=" * 60)

    answer = generate_answer(
        query,
        documents
    )

else:

    print("\nABSTAIN - Weak evidence")

    answer = (
        "I couldn't find reliable information about that "
        "in the campus knowledge base."
    )


# Final answer
print("\n" + "=" * 60)
print("CAMPUSIQ AI ANSWER")
print("=" * 60)

print("\n" + answer)


# Close Qdrant
client.close()
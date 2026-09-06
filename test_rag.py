
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from qdrant_client import QdrantClient

from ai.embeddings.embeddings import EmbeddingModel
from ai.rag.rag import search_documents, check_evidence
from ai.rag.generator import generate_answer


# ============================================================
# CONNECT TO QDRANT
# ============================================================

client = QdrantClient(path=".qdrant")


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

embedding_model = EmbeddingModel()


# ============================================================
# USER QUESTION
# ============================================================

query = "What happens if my attendance falls below 75%?"


print("\n" + "=" * 60)
print("USER QUESTION")
print("=" * 60)

print(query)


# ============================================================
# RETRIEVE RELEVANT DOCUMENTS
# ============================================================

documents = search_documents(
    client,
    embedding_model,
    query,
    top_k=2
)


# ============================================================
# DISPLAY RETRIEVED DOCUMENTS
# ============================================================

print("\n" + "=" * 60)
print("RETRIEVED DOCUMENTS")
print("=" * 60)

for i, document in enumerate(documents, start=1):

    print(f"\nResult {i}")

    print(
        f"Vector Score : "
        f"{document['score']:.4f}"
    )

    print(
        f"Rerank Score : "
        f"{document.get('rerank_score', 0.0):.4f}"
    )

    print(
        f"Source       : "
        f"{document['source']}"
    )

    print(
        f"Page         : "
        f"{document['page']}"
    )

    print(
        f"Chunk        : "
        f"{document.get('chunk_id')}"
    )


# ============================================================
# EVIDENCE GUARD V2
# ============================================================

print("\n" + "=" * 60)
print("EVIDENCE GUARD V2")
print("=" * 60)


evidence = check_evidence(
    documents,
    vector_threshold=0.45,
    rerank_threshold=0.0
)


print(
    f"Best Vector Score : "
    f"{evidence['best_vector_score']:.4f}"
)

print(
    f"Best Rerank Score : "
    f"{evidence['best_rerank_score']:.4f}"
)

print(
    f"Decision          : "
    f"{'PASS' if evidence['has_evidence'] else 'ABSTAIN'}"
)

print(
    f"Reason            : "
    f"{evidence['reason']}"
)


# ============================================================
# GENERATE ANSWER ONLY IF EVIDENCE IS STRONG
# ============================================================

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


# ============================================================
# FINAL ANSWER
# ============================================================

print("\n" + "=" * 60)
print("CAMPUSIQ AI ANSWER")
print("=" * 60)

print("\n" + answer)


# ============================================================
# CLOSE QDRANT
# ============================================================

client.close()

from qdrant_client import QdrantClient

from ai.embeddings.embeddings import EmbeddingModel
from ai.rag.rag import ask_campusiq


# ============================================================
# INITIALIZE
# ============================================================

client = QdrantClient(
    path=".qdrant"
)

embedding_model = EmbeddingModel()


# ============================================================
# QUESTIONS
# ============================================================

questions = [

    "What is the minimum attendance requirement?",

    "What happens if attendance falls below 75%?",

    "Can attendance be relaxed?",

    "What is the hostel fee?",

    "Who is the placement officer?"
]


# ============================================================
# RUN CAMPUSIQ
# ============================================================

for question in questions:

    result = ask_campusiq(
        client=client,
        embedding_model=embedding_model,
        query=question
    )

    print("\n" + "=" * 70)
    print("FINAL RESULT")
    print("=" * 70)

    print(
        f"Status : {result['status']}"
    )

    print(
        f"Answer : {result['answer']}"
    )

    print(
        f"Model  : {result['model']}"
    )

    print("\nSources:")

    for source in result["sources"]:

        print(
            f"- {source['source']} "
            f"(Page {source['page']}, "
            f"Chunk {source['chunk_id']})"
        )
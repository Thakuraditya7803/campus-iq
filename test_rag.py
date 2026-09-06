from qdrant_client import QdrantClient

from ai.embeddings.embeddings import EmbeddingModel
from ai.rag.rag import search_documents
from ai.rag.generator import generate_answer


client = QdrantClient(path=".qdrant")

embedding_model = EmbeddingModel()

question = "How much attendance do I need?"

documents = search_documents(
    client,
    embedding_model,
    question,
    top_k=3
)

print("\n==============================")
print("Retrieved Documents")
print("==============================")

for doc in documents:
    print(f"\nScore: {doc['score']:.4f}")
    print(f"Source: {doc['source']}")
    print(f"Page: {doc['page']}")

try:
    answer = generate_answer(
        question,
        documents
    )

    print("\n==============================")
    print("CampusIQ AI Answer")
    print("==============================")

    print(answer)

finally:
    client.close()
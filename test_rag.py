import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))


from qdrant_client import QdrantClient

from ai.embeddings.embeddings import EmbeddingModel
from ai.rag.rag import search_documents
from ai.rag.generator import generate_answer


client = QdrantClient(path=".qdrant")

embedding_model = EmbeddingModel()


query = "What happens if my attendance falls below 75%?"


print("\n" + "=" * 60)
print("USER QUESTION")
print("=" * 60)

print(query)


documents = search_documents(
    client,
    embedding_model,
    query,
    top_k=2
)


print("\n" + "=" * 60)
print("RETRIEVED DOCUMENTS")
print("=" * 60)


for i, document in enumerate(documents, start=1):

    print(f"\nResult {i}")
    print(f"Score: {document['score']:.4f}")
    print(f"Source: {document['source']}")
    print(f"Page: {document['page']}")
    print(f"Chunk: {document.get('chunk_id')}")


print("\n" + "=" * 60)
print("GENERATING ANSWER")
print("=" * 60)


answer = generate_answer(
    query,
    documents
)


print("\nCampusIQ AI Answer:\n")
print(answer)


client.close()
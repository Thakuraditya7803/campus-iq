import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from qdrant_client import QdrantClient

from ai.embeddings.embeddings import EmbeddingModel
from ai.rag.rag import search_documents


client = QdrantClient(path=".qdrant")
embedding_model = EmbeddingModel()


queries = [
    "What is the minimum attendance requirement?",
    "What happens if attendance falls below 75%?",
    "Can attendance be relaxed?",
    "What documents are required for attendance relaxation?",
    "What are the attendance requirements for B.Tech students?",
]


for query in queries:

    print("\n" + "=" * 60)
    print(f"QUERY: {query}")
    print("=" * 60)

    results = search_documents(
        client,
        embedding_model,
        query,
        top_k=2
    )

    for i, result in enumerate(results, start=1):

        print(f"\nResult {i}")
        print(f"Score: {result['score']:.4f}")
        print(f"Source: {result['source']}")
        print(f"Page: {result['page']}")
        print(f"Chunk ID: {result.get('chunk_id')}")
        print(f"Text:\n{result['text']}")


client.close()
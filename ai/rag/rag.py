from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from ai.embeddings.embeddings import EmbeddingModel
from ingestion.pdf.extract import extract_text_from_pdf
from ingestion.pdf.chunker import chunk_text


COLLECTION_NAME = "campus_documents"


def create_collection(client, vector_size):

    collections = client.get_collections().collections
    existing = [c.name for c in collections]

    if COLLECTION_NAME in existing:
        print(f"Collection '{COLLECTION_NAME}' already exists.")
        return

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE
        )
    )

    print(f"Created collection: {COLLECTION_NAME}")


def index_pdf(client, embedding_model, pdf_path):

    text = extract_text_from_pdf(str(pdf_path))

    chunks = chunk_text(
        text,
        chunk_size=1000,
        overlap=200
    )

    print(f"Document: {pdf_path.name}")
    print(f"Generated {len(chunks)} chunks")

    embeddings = []

    for i, chunk in enumerate(chunks):

        print(f"Embedding chunk {i + 1}/{len(chunks)}")

        embedding = embedding_model.generate_embedding(chunk)

        embeddings.append(
            {
                "chunk_id": i,
                "text": chunk,
                "embedding": embedding
            }
        )

    create_collection(
        client,
        vector_size=len(embeddings[0]["embedding"])
    )

    points = []

    for item in embeddings:

        point = PointStruct(
            id=item["chunk_id"],
            vector=item["embedding"],
            payload={
                "source": pdf_path.name,
                "page": 1,
                "chunk_id": item["chunk_id"],
                "text": item["text"]
            }
        )

        points.append(point)

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )

    print(f"Indexed {len(points)} chunks from {pdf_path.name}")


def search_documents(client, embedding_model, query, top_k=3):

    query_embedding = embedding_model.generate_embedding(query)

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        limit=top_k,
    ).points

    documents = []

    for result in results:

        documents.append({
    "text": result.payload["text"],
    "source": result.payload["source"],
    "page": result.payload["page"],
    "chunk_id": result.payload.get("chunk_id"),
    "score": result.score
})

    return documents
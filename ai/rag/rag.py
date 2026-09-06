from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from ai.embeddings.embeddings import EmbeddingModel
from ingestion.pdf.extract import extract_text_from_pdf


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

    embedding = embedding_model.generate_embedding(text)

    create_collection(
        client,
        vector_size=len(embedding)
    )

    point = PointStruct(
        id=1,
        vector=embedding,
        payload={
            "source": pdf_path.name,
            "page": 1,
            "text": text
        }
    )

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[point]
    )

    print(f"Indexed: {pdf_path.name}")


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
            "score": result.score
        })

    return documents
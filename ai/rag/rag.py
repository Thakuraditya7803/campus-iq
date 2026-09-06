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

def check_evidence(documents, min_score=0.50, min_gap=0.02):
    """
    Evaluate the strength of retrieved evidence.

    Returns:
        {
            "has_evidence": bool,
            "best_score": float,
            "second_score": float,
            "score_gap": float,
            "reason": str
        }
    """

    if not documents:
        return {
            "has_evidence": False,
            "best_score": 0.0,
            "second_score": 0.0,
            "score_gap": 0.0,
            "reason": "No documents retrieved"
        }

    best_score = documents[0]["score"]

    if len(documents) > 1:
        second_score = documents[1]["score"]
    else:
        second_score = 0.0

    score_gap = best_score - second_score

    # Evidence is strong enough when:
    # 1. Best result has reasonable similarity
    # 2. Best result is meaningfully better than second result

    has_evidence = (
        best_score >= min_score
        and score_gap >= min_gap
    )

    if has_evidence:
        reason = "Strong evidence"
    else:
        reason = "Weak or ambiguous evidence"

    return {
        "has_evidence": has_evidence,
        "best_score": best_score,
        "second_score": second_score,
        "score_gap": score_gap,
        "reason": reason
    }

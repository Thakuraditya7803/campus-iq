from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from ai.embeddings.embeddings import EmbeddingModel
from ingestion.pdf.extract import extract_text_from_pdf
from ingestion.pdf.chunker import chunk_text
from ai.rag.reranker import Reranker
from ai.rag.generator import generate_answer


COLLECTION_NAME = "campus_documents"

reranker = Reranker()


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



def search_documents(
    client,
    embedding_model,
    query,
    top_k=5,
    retrieval_k=5
):
    """
    Retrieve candidate documents from Qdrant and
    rerank them using a Cross-Encoder.
    """

    query_embedding = embedding_model.generate_embedding(query)

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        limit=retrieval_k,
    ).points

    # --------------------------------------------------
    # STEP 1: Vector retrieval
    # --------------------------------------------------

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        limit=retrieval_k,
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

    if not documents:
        return []

    # --------------------------------------------------
    # STEP 2: Reranking
    # --------------------------------------------------

    print("\nVECTOR RETRIEVAL DEBUG")

    for i, document in enumerate(documents, start=1):
        print(
        f"{i}. Chunk={document['chunk_id']} "
        f"Vector={document['score']:.4f}"
    )


    reranked_documents = reranker.rerank(
    query,
    documents,
    top_k=top_k
)

    print("\nRERANKING DEBUG")

    for i, document in enumerate(reranked_documents, start=1):
      print(
        f"{i}. "
        f"Chunk={document['chunk_id']} "
        f"Vector={document['score']:.4f} "
        f"Rerank={document['rerank_score']:.4f}"
    )
      return reranked_documents


def check_evidence(
    documents,
    vector_threshold=0.45,
    rerank_threshold=-3.0,
    score_gap_threshold=1.0
):
    """
    Evidence Guard V3

    Uses multiple signals:
    - Vector similarity
    - Cross-encoder rerank score
    - Difference between the best and second-best rerank score

    Goal:
    Reduce false negatives while keeping false positives low.
    """

    if not documents:
        return {
            "has_evidence": False,
            "best_vector_score": 0.0,
            "best_rerank_score": 0.0,
            "score_gap": 0.0,
            "reason": "No documents retrieved"
        }

    best = documents[0]

    vector_score = best["score"]
    rerank_score = best.get("rerank_score", float("-inf"))

    second_rerank_score = (
        documents[1].get("rerank_score", float("-inf"))
        if len(documents) > 1
        else float("-inf")
    )

    score_gap = (
        rerank_score - second_rerank_score
        if second_rerank_score != float("-inf")
        else 999.0
    )

    # Strong semantic evidence
    strong_vector = vector_score >= vector_threshold

    # Cross-encoder accepts moderately negative scores
    acceptable_rerank = rerank_score >= rerank_threshold

    # Evidence is clearly stronger than alternatives
    strong_separation = score_gap >= score_gap_threshold

    # V3 decision
    has_evidence = (
        strong_vector and acceptable_rerank
    ) or (
        acceptable_rerank and strong_separation
    )

    if has_evidence:
        reason = "Strong evidence"
    elif not strong_vector:
        reason = "Low semantic similarity"
    elif not acceptable_rerank:
        reason = "Low reranker relevance"
    else:
        reason = "Insufficient evidence separation"

    return {
        "has_evidence": has_evidence,
        "best_vector_score": vector_score,
        "best_rerank_score": rerank_score,
        "score_gap": score_gap,
        "reason": reason
    }
def ask_campusiq(
    client,
    embedding_model,
    query
):
    """
    Complete CampusIQ question-answering pipeline.

    Flow:
        Query
        ↓
        Retrieval
        ↓
        Reranking
        ↓
        Evidence Guard
        ↓
        Generator
    """

    print("\n" + "=" * 70)
    print("CAMPUSIQ")
    print("=" * 70)

    print(f"Question: {query}")

    # ========================================================
    # STEP 1 — RETRIEVAL + RERANKING
    # ========================================================

    documents = search_documents(
        client=client,
        embedding_model=embedding_model,
        query=query,
        top_k=2,
        retrieval_k=5
    )

    # ========================================================
    # STEP 2 — EVIDENCE GUARD
    # ========================================================

    evidence = check_evidence(
        documents
    )

    print("\nEVIDENCE CHECK")
    print("-" * 40)

    print(
        f"Evidence      : {evidence['has_evidence']}"
    )

    print(
        f"Vector Score  : {evidence['best_vector_score']:.4f}"
    )

    print(
        f"Rerank Score  : {evidence['best_rerank_score']:.4f}"
    )

    print(
        f"Score Gap     : {evidence['score_gap']:.4f}"
    )

    print(
        f"Reason        : {evidence['reason']}"
    )

    # ========================================================
    # STEP 3 — ABSTAIN IF EVIDENCE IS WEAK
    # ========================================================

    if not evidence["has_evidence"]:

        print("\nEvidence insufficient.")
        print("CampusIQ is abstaining.")

        return {
            "status": "abstained",

            "answer": (
                "I couldn't find reliable information about that "
                "in the campus knowledge base."
            ),

            "sources": [],

            "evidence": evidence,

            "model": None
        }

    # ========================================================
    # STEP 4 — GENERATE GROUNDED ANSWER
    # ========================================================

    result = generate_answer(
        query=query,
        documents=documents
    )

    # ========================================================
    # STEP 5 — RETURN STRUCTURED RESPONSE
    # ========================================================

    return {
        "status": "answered",

        "answer": result["answer"],

        "sources": result["sources"],

        "evidence": evidence,

        "model": result["model"]
    }
from pathlib import Path

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


COLLECTION_NAME = "campus_documents"


def extract_text(pdf_path):
    reader = PdfReader(pdf_path)

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()

        if text:
            pages.append({
                "page": page_number,
                "text": text.strip()
            })

    return pages


def chunk_text(text, chunk_size=500):
    words = text.split()

    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks


def main():

    print("Loading embedding model...")

    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Connecting to Qdrant...")

    client = QdrantClient(path=".qdrant")

    # Create collection if it doesn't exist
    collections = client.get_collections().collections

    collection_names = [collection.name for collection in collections]

    if COLLECTION_NAME not in collection_names:

        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=384,
                distance=Distance.COSINE
            )
        )

        print("Created Qdrant collection.")

    pdf_folder = Path("data/raw")

    points = []

    point_id = 1

    for pdf_file in pdf_folder.glob("*.pdf"):

        print(f"\nProcessing: {pdf_file.name}")

        pages = extract_text(pdf_file)

        for page in pages:

            chunks = chunk_text(page["text"])

            for chunk in chunks:

                embedding = model.encode(chunk).tolist()

                points.append(
                    PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload={
                            "text": chunk,
                            "source": pdf_file.name,
                            "page": page["page"]
                        }
                    )
                )

                point_id += 1

    if points:

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )

    print(f"\nInserted {len(points)} chunks into Qdrant.")

    print("\nCampusIQ knowledge base is ready!")

def search_documents(client, model, query, top_k=3):

    query_embedding = model.encode(query).tolist()

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        limit=top_k,
    ).points

    print("\n==============================")
    print(f"QUESTION: {query}")
    print("==============================")

    for i, result in enumerate(results, start=1):

        print(f"\nResult {i}")
        print(f"Score: {result.score:.4f}")
        print(f"Source: {result.payload['source']}")
        print(f"Page: {result.payload['page']}")
        print(f"Text: {result.payload['text']}")

    return [
        {
            "text": result.payload["text"],
            "source": result.payload["source"],
            "page": result.payload["page"],
            "score": result.score
        }
        for result in results
    ]
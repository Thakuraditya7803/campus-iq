from pathlib import Path

from qdrant_client import QdrantClient

from ai.embeddings.embeddings import EmbeddingModel
from ai.rag.rag import index_pdf, COLLECTION_NAME


client = QdrantClient(path=".qdrant")

embedding_model = EmbeddingModel()

# Delete old collection
try:
    client.delete_collection(COLLECTION_NAME)
    print("Old collection deleted.")
except Exception:
    print("Collection did not exist.")


pdf_path = Path("data/raw/attendance_policy.pdf")

index_pdf(
    client,
    embedding_model,
    pdf_path
)

print("\nPDF indexing completed successfully!")

client.close()
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

from ai.rag.rag import search_documents, COLLECTION_NAME
from ai.rag.generator import generate_answer


model = SentenceTransformer("all-MiniLM-L6-v2")

client = QdrantClient(path=".qdrant")


question = "How much attendance do I need?"


documents = search_documents(
    client,
    model,
    question,
    top_k=3
)


answer = generate_answer(
    question,
    documents
)


print("\n==============================")
print("CampusIQ Answer")
print("==============================")
print(answer)
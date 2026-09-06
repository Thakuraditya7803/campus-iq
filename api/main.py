from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from qdrant_client import QdrantClient

from ai.embeddings.embeddings import EmbeddingModel
from ai.rag.rag import ask_campusiq


# --------------------------------------------------
# FastAPI Application
# --------------------------------------------------

app = FastAPI(
    title="CampusIQ API",
    description="AI-powered campus knowledge assistant",
    version="1.0.0"
)


# --------------------------------------------------
# CORS
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# Initialize Services
# --------------------------------------------------

client = QdrantClient(
    path=".qdrant"
)

embedding_model = EmbeddingModel()


# --------------------------------------------------
# Request Model
# --------------------------------------------------

class AskRequest(BaseModel):
    question: str


# --------------------------------------------------
# Health Check
# --------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "CampusIQ API"
    }


# --------------------------------------------------
# Ask CampusIQ
# --------------------------------------------------

@app.post("/ask")
def ask(request: AskRequest):

    question = request.question.strip()

    if not question:
        return {
            "status": "error",
            "answer": "Please provide a question.",
            "sources": [],
            "evidence": {},
            "model": None
        }

    result = ask_campusiq(
        client=client,
        embedding_model=embedding_model,
        query=question
    )

    return result

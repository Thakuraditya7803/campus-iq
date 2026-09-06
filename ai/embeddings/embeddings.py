from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    def __init__(self):
        print("Loading embedding model...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def generate_embedding(self, text: str):
        return self.model.encode(text).tolist()


if __name__ == "__main__":
    model = EmbeddingModel()

    text = "Students must maintain 75% attendance."

    embedding = model.generate_embedding(text)

    print("Embedding generated!")
    print("Embedding dimensions:", len(embedding))
    print("First 5 values:", embedding[:5])
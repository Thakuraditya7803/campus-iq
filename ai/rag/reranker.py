
from sentence_transformers import CrossEncoder


class Reranker:

    def __init__(self):

        print("Loading reranker model...")

        self.model = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

        print("Reranker loaded!")


    def rerank(self, query, documents, top_k=2):

        if not documents:
            return []

        pairs = [
            [query, document["text"]]
            for document in documents
        ]

        scores = self.model.predict(pairs)

        reranked = []

        for document, score in zip(documents, scores):

            item = document.copy()

            item["rerank_score"] = float(score)

            reranked.append(item)

        reranked.sort(
            key=lambda x: x["rerank_score"],
            reverse=True
        )

        return reranked[:top_k]


if __name__ == "__main__":

    reranker = Reranker()

    query = "Can attendance be relaxed?"

    documents = [
        {
            "text": "Students must maintain 75% overall attendance.",
            "source": "attendance_policy.pdf",
            "page": 1,
            "chunk_id": 0,
            "score": 0.59
        },
        {
            "text": "Up to 5–10% relaxation may be granted at the discretion of the Dean/HOD.",
            "source": "attendance_policy.pdf",
            "page": 1,
            "chunk_id": 3,
            "score": 0.58
        }
    ]

    results = reranker.rerank(
        query,
        documents,
        top_k=2
    )

    print("\nRERANKED RESULTS")

    for i, result in enumerate(results, start=1):

        print(f"\nResult {i}")
        print(f"Chunk: {result['chunk_id']}")
        print(f"Original score: {result['score']:.4f}")
        print(f"Rerank score: {result['rerank_score']:.4f}")
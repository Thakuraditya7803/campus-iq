
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from qdrant_client import QdrantClient

from ai.embeddings.embeddings import EmbeddingModel
from ai.rag.rag import search_documents


client = QdrantClient(path=".qdrant")
embedding_model = EmbeddingModel()


TEST_CASES = [
    # Answerable
    {"question": "What is the minimum attendance requirement?", "expected_chunk": 0},
    {"question": "What happens if attendance falls below 75%?", "expected_chunk": 2},
    {"question": "Can attendance be relaxed?", "expected_chunk": 3},
    {"question": "What documents are required for attendance relaxation?", "expected_chunk": 3},
    {"question": "What are the attendance requirements for B.Tech students?", "expected_chunk": 0},
    {"question": "What percentage of attendance is required?", "expected_chunk": 0},
    {"question": "What happens to students below the attendance requirement?", "expected_chunk": 2},
    {"question": "Is medical emergency a valid reason for attendance relaxation?", "expected_chunk": 3},
    {"question": "Is supporting documentation required for relaxation?", "expected_chunk": 3},
    {"question": "What is the minimum attendance for each subject?", "expected_chunk": 0},

    # Unanswerable
    {"question": "What is the hostel fee?", "expected_chunk": None},
    {"question": "When is the next campus placement drive?", "expected_chunk": None},
    {"question": "Who is the head of the computer department?", "expected_chunk": None},
    {"question": "What is the university examination timetable?", "expected_chunk": None},
    {"question": "How many students are enrolled in B.Tech?", "expected_chunk": None},
    {"question": "What is the mess fee?", "expected_chunk": None},
    {"question": "What are the hostel rules?", "expected_chunk": None},
    {"question": "Who is the placement officer?", "expected_chunk": None},
    {"question": "What companies are visiting for placements?", "expected_chunk": None},
    {"question": "When does the semester begin?", "expected_chunk": None},
]


print("\n" + "=" * 70)
print("CAMPUSIQ RETRIEVAL EVALUATION")
print("=" * 70)


retrieval_hits_at_1 = 0
retrieval_hits_at_2 = 0

answerable_questions = 0

scores = []


for test in TEST_CASES:

    question = test["question"]
    expected_chunk = test["expected_chunk"]

    results = search_documents(
        client,
        embedding_model,
        question,
        top_k=2
    )

    retrieved_chunks = [
        result.get("chunk_id")
        for result in results
    ]

    best_score = results[0]["score"] if results else 0

    scores.append(best_score)

    print("\n" + "-" * 70)
    print("Question:")
    print(question)

    print(f"Expected chunk: {expected_chunk}")
    print(f"Retrieved chunks: {retrieved_chunks}")
    print(f"Best score: {best_score:.4f}")

    # Only answerable questions are used for Recall@K
    if expected_chunk is not None:

        answerable_questions += 1

        # Recall@1
        if len(retrieved_chunks) >= 1:
            if retrieved_chunks[0] == expected_chunk:
                retrieval_hits_at_1 += 1

        # Recall@2
        if expected_chunk in retrieved_chunks:
            retrieval_hits_at_2 += 1

        hit_1 = (
            len(retrieved_chunks) >= 1
            and retrieved_chunks[0] == expected_chunk
        )

        hit_2 = expected_chunk in retrieved_chunks

        print(
            f"Recall@1: {'HIT' if hit_1 else 'MISS'}"
        )

        print(
            f"Recall@2: {'HIT' if hit_2 else 'MISS'}"
        )

    else:

        print("Unanswerable question - not included in Recall@K")


# Calculate metrics

recall_at_1 = (
    retrieval_hits_at_1 / answerable_questions
    if answerable_questions
    else 0
)

recall_at_2 = (
    retrieval_hits_at_2 / answerable_questions
    if answerable_questions
    else 0
)

average_score = (
    sum(scores) / len(scores)
    if scores
    else 0
)


print("\n" + "=" * 70)
print("CAMPUSIQ RETRIEVAL REPORT")
print("=" * 70)

print("\nRETRIEVAL METRICS")
print("-" * 40)

print(
    f"Answerable Questions : {answerable_questions}"
)

print(
    f"Recall@1             : {recall_at_1:.2%}"
)

print(
    f"Recall@2             : {recall_at_2:.2%}"
)

print(
    f"Average Best Score   : {average_score:.4f}"
)

print("\n" + "=" * 70)

client.close()

import sys
from pathlib import Path

# Allow imports from project root
sys.path.append(str(Path(__file__).resolve().parents[1]))

from qdrant_client import QdrantClient

from ai.embeddings.embeddings import EmbeddingModel
from ai.rag.rag import search_documents, check_evidence


# ============================================================
# INITIALIZE
# ============================================================

client = QdrantClient(path=".qdrant")
embedding_model = EmbeddingModel()


# ============================================================
# TEST DATASET
# ============================================================

TEST_CASES = [

    # --------------------------------------------------------
    # ANSWERABLE QUESTIONS
    # --------------------------------------------------------

    {
        "question": "What is the minimum attendance requirement?",
        "expected_chunk": 1
    },

    {
        "question": "What happens if attendance falls below 75%?",
        "expected_chunk": 2
    },

    {
        "question": "Can attendance be relaxed?",
        "expected_chunk": 3
    },

    {
        "question": "What documents are required for attendance relaxation?",
        "expected_chunk": 3
    },

    {
        "question": "What are the attendance requirements for B.Tech students?",
        "expected_chunk": 1
    },

    {
        "question": "What percentage of attendance is required?",
        "expected_chunk": 1
    },

    {
        "question": "What happens to students below the attendance requirement?",
        "expected_chunk": 2
    },

    {
        "question": "Is medical emergency a valid reason for attendance relaxation?",
        "expected_chunk": 3
    },

    {
        "question": "Is supporting documentation required for relaxation?",
        "expected_chunk": 3
    },

    {
        "question": "What is the minimum attendance for each subject?",
        "expected_chunk": 1
    },


    # --------------------------------------------------------
    # UNANSWERABLE QUESTIONS
    # --------------------------------------------------------

    {
        "question": "What is the hostel fee?",
        "expected_chunk": None
    },

    {
        "question": "When is the next campus placement drive?",
        "expected_chunk": None
    },

    {
        "question": "Who is the head of the computer department?",
        "expected_chunk": None
    },

    {
        "question": "What is the university examination timetable?",
        "expected_chunk": None
    },

    {
        "question": "How many students are enrolled in B.Tech?",
        "expected_chunk": None
    },

    {
        "question": "What is the mess fee?",
        "expected_chunk": None
    },

    {
        "question": "What are the hostel rules?",
        "expected_chunk": None
    },

    {
        "question": "Who is the placement officer?",
        "expected_chunk": None
    },

    {
        "question": "What companies are visiting for placements?",
        "expected_chunk": None
    },

    {
        "question": "When does the semester begin?",
        "expected_chunk": None
    }
]


# ============================================================
# START EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("CAMPUSIQ RETRIEVAL EVALUATION")
print("=" * 70)


# ============================================================
# RETRIEVAL METRICS
# ============================================================

retrieval_hits_at_1 = 0
retrieval_hits_at_2 = 0

answerable_questions = 0

scores = []


# ============================================================
# EVIDENCE GUARD METRICS
# ============================================================

true_positives = 0
true_negatives = 0
false_positives = 0
false_negatives = 0


# ============================================================
# RUN TEST CASES
# ============================================================

for test in TEST_CASES:

    question = test["question"]
    expected_chunk = test["expected_chunk"]


    # --------------------------------------------------------
    # SEARCH DOCUMENTS
    # --------------------------------------------------------

    results = search_documents(
        client,
        embedding_model,
        question,
        top_k=5,
        retrieval_k=5
    )


    # --------------------------------------------------------
    # GET RETRIEVED CHUNKS
    # --------------------------------------------------------

    retrieved_chunks = [
        result.get("chunk_id")
        for result in results
    ]


    # --------------------------------------------------------
    # BEST VECTOR SCORE
    # --------------------------------------------------------

    best_score = (
        results[0]["score"]
        if results
        else 0
    )

    scores.append(best_score)


    # --------------------------------------------------------
    # EVIDENCE GUARD V3
    # --------------------------------------------------------

    evidence = check_evidence(
        results,

        vector_threshold=0.45,

        rerank_threshold=-3.0,

        score_gap_threshold=1.0
    )


    predicted_answerable = evidence["has_evidence"]

    actual_answerable = (
        expected_chunk is not None
    )


    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    if actual_answerable and predicted_answerable:

        true_positives += 1

    elif not actual_answerable and not predicted_answerable:

        true_negatives += 1

    elif not actual_answerable and predicted_answerable:

        false_positives += 1

    elif actual_answerable and not predicted_answerable:

        false_negatives += 1


    # ========================================================
    # PRINT TEST RESULT
    # ========================================================

    print("\n" + "-" * 70)

    print("Question:")
    print(question)

    print(
        f"Expected chunk: {expected_chunk}"
    )

    print(
        f"Retrieved chunks: {retrieved_chunks}"
    )

    print(
        f"Best score: {best_score:.4f}"
    )


    # --------------------------------------------------------
    # EVIDENCE GUARD RESULT
    # --------------------------------------------------------

    print(
        f"Evidence Guard: "
        f"{'PASS' if predicted_answerable else 'ABSTAIN'} "
        f"| Vector={evidence['best_vector_score']:.4f} "
        f"| Rerank={evidence['best_rerank_score']:.4f} "
        f"| Gap={evidence['score_gap']:.4f} "
        f"| {evidence['reason']}"
    )


    # ========================================================
    # RETRIEVAL EVALUATION
    # ========================================================

    if expected_chunk is not None:

        answerable_questions += 1


        # ----------------------------------------------------
        # RECALL @ 1
        # ----------------------------------------------------

        hit_1 = (
            len(retrieved_chunks) >= 1
            and retrieved_chunks[0] == expected_chunk
        )

        if hit_1:
            retrieval_hits_at_1 += 1


        # ----------------------------------------------------
        # RECALL @ 2
        # ----------------------------------------------------

        hit_2 = (
            expected_chunk in retrieved_chunks[:2]
        )

        if hit_2:
            retrieval_hits_at_2 += 1


        print(
            f"Recall@1: "
            f"{'HIT' if hit_1 else 'MISS'}"
        )

        print(
            f"Recall@2: "
            f"{'HIT' if hit_2 else 'MISS'}"
        )


    else:

        print(
            "Unanswerable question - "
            "not included in Recall@K"
        )


# ============================================================
# CALCULATE RETRIEVAL METRICS
# ============================================================

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


# ============================================================
# RETRIEVAL REPORT
# ============================================================

print("\n" + "=" * 70)
print("CAMPUSIQ RETRIEVAL REPORT")
print("=" * 70)

print("\nRETRIEVAL METRICS")
print("-" * 40)

print(
    f"Answerable Questions : "
    f"{answerable_questions}"
)

print(
    f"Recall@1             : "
    f"{recall_at_1:.2%}"
)

print(
    f"Recall@2             : "
    f"{recall_at_2:.2%}"
)

print(
    f"Average Best Score   : "
    f"{average_score:.4f}"
)


# ============================================================
# EVIDENCE GUARD METRICS
# ============================================================

total_questions = len(TEST_CASES)


# ------------------------------------------------------------
# ACCURACY
# ------------------------------------------------------------

guard_accuracy = (
    (true_positives + true_negatives)
    / total_questions
    if total_questions
    else 0
)


# ------------------------------------------------------------
# PRECISION
# ------------------------------------------------------------

guard_precision = (
    true_positives
    / (true_positives + false_positives)
    if (true_positives + false_positives) > 0
    else 0
)


# ------------------------------------------------------------
# RECALL
# ------------------------------------------------------------

guard_recall = (
    true_positives
    / (true_positives + false_negatives)
    if (true_positives + false_negatives) > 0
    else 0
)


# ------------------------------------------------------------
# F1 SCORE
# ------------------------------------------------------------

guard_f1 = (
    2
    * guard_precision
    * guard_recall
    / (guard_precision + guard_recall)
    if (guard_precision + guard_recall) > 0
    else 0
)


# ============================================================
# EVIDENCE GUARD V3 REPORT
# ============================================================

print("\n" + "=" * 70)
print("EVIDENCE GUARD V3 REPORT")
print("=" * 70)


print("\nCONFUSION MATRIX")
print("-" * 40)

print(
    f"True Positives  : "
    f"{true_positives}"
)

print(
    f"True Negatives  : "
    f"{true_negatives}"
)

print(
    f"False Positives : "
    f"{false_positives}"
)

print(
    f"False Negatives : "
    f"{false_negatives}"
)


print("\nGUARD METRICS")
print("-" * 40)

print(
    f"Accuracy  : "
    f"{guard_accuracy:.2%}"
)

print(
    f"Precision : "
    f"{guard_precision:.2%}"
)

print(
    f"Recall    : "
    f"{guard_recall:.2%}"
)

print(
    f"F1 Score  : "
    f"{guard_f1:.2%}"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("CAMPUSIQ EVALUATION COMPLETE")
print("=" * 70)

print(
    f"Retrieval Recall@1 : "
    f"{recall_at_1:.2%}"
)

print(
    f"Retrieval Recall@2 : "
    f"{recall_at_2:.2%}"
)

print(
    f"Evidence Accuracy  : "
    f"{guard_accuracy:.2%}"
)

print(
    f"Evidence Precision : "
    f"{guard_precision:.2%}"
)

print(
    f"Evidence Recall    : "
    f"{guard_recall:.2%}"
)

print(
    f"Evidence F1        : "
    f"{guard_f1:.2%}"
)

print(
    f"False Positives    : "
    f"{false_positives}"
)

print(
    f"False Negatives    : "
    f"{false_negatives}"
)

print("=" * 70)


# ============================================================
# CLOSE QDRANT
# ============================================================

client.close()
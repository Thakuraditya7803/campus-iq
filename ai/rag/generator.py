
import os

from dotenv import load_dotenv
from openai import OpenAI


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# OPENROUTER CLIENT
# ============================================================

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)


# ============================================================
# FREE MODEL FALLBACK CHAIN
# ============================================================

MODELS = [
    "minimax/minimax-m3:free",
    "inclusionai/ling-3.0-flash-fin:free",
    "google/gemma-4-26b-a4b-it:free",
    "z-ai/glm-5.2:free",
    "nvidia/nemotron-3.5-lightning:free",
]


# ============================================================
# ABSTENTION MESSAGE
# ============================================================

ABSTAIN_MESSAGE = (
    "I couldn't find reliable information about that "
    "in the campus knowledge base."
)


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(query, documents):

    """
    Generate a grounded answer using retrieved documents.

    Parameters
    ----------
    query : str
        User's question.

    documents : list
        Retrieved and reranked documents.

    Returns
    -------
    dict
        {
            "answer": str,
            "sources": list,
            "model": str | None
        }
    """


    # --------------------------------------------------------
    # NO DOCUMENTS
    # --------------------------------------------------------

    if not documents:

        return {
            "answer": ABSTAIN_MESSAGE,
            "sources": [],
            "model": None
        }


    # ========================================================
    # BUILD CONTEXT
    # ========================================================

    context_parts = []


    for i, document in enumerate(documents, start=1):

        context_parts.append(
            f"""
SOURCE {i}

Document:
{document.get("source", "Unknown")}

Page:
{document.get("page", "Unknown")}

Chunk ID:
{document.get("chunk_id", "Unknown")}

Content:
{document.get("text", "")}
"""
        )


    context = "\n".join(context_parts)


    # ========================================================
    # RAG PROMPT
    # ========================================================

    prompt = f"""
You are CampusIQ, an AI-powered campus knowledge assistant.

Your task is to answer the student's question using ONLY
the provided campus knowledge.

STRICT RULES:

1. Use ONLY the information contained in the context.

2. Do NOT use outside knowledge.

3. Do NOT guess or invent information.

4. If the context does not contain enough information to
   answer the question, respond exactly with:

"I couldn't find reliable information about that in the
campus knowledge base."

5. Give a concise, clear and direct answer.

6. Preserve important numbers, percentages, dates,
   conditions and requirements exactly.

7. Do not add information that is not supported by
   the context.

8. Do not mention unrelated information from the context.

9. Do not mention the internal retrieval process.

10. Do not say that you are an AI language model.

11. If the answer is supported by the context, mention
    the relevant document and page naturally.

------------------------------------------------------------

CAMPUS KNOWLEDGE:

{context}

------------------------------------------------------------

STUDENT QUESTION:

{query}

------------------------------------------------------------

ANSWER:
"""


    # ========================================================
    # MODEL FALLBACK LOOP
    # ========================================================

    for model in MODELS:

        try:

            print(f"\nTrying model: {model}")


            response = client.chat.completions.create(

                model=model,

                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are CampusIQ. "
                            "Follow the grounding rules strictly."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],

                temperature=0.1,

                max_tokens=500
            )


            # ------------------------------------------------
            # EXTRACT ANSWER
            # ------------------------------------------------

            answer = response.choices[0].message.content


            if not answer:

                print(
                    f"Empty response from model: {model}"
                )

                continue


            answer = answer.strip()


            # ------------------------------------------------
            # BUILD SOURCE INFORMATION
            # ------------------------------------------------

            sources = []


            for document in documents:

                sources.append(
                    {
                        "source": document.get(
                            "source"
                        ),

                        "page": document.get(
                            "page"
                        ),

                        "chunk_id": document.get(
                            "chunk_id"
                        ),

                        "vector_score": document.get(
                            "score"
                        ),

                        "rerank_score": document.get(
                            "rerank_score"
                        )
                    }
                )


            # ------------------------------------------------
            # SUCCESS
            # ------------------------------------------------

            print(
                f"Answer generated using: {model}"
            )


            return {
                "answer": answer,
                "sources": sources,
                "model": model
            }


        # ====================================================
        # MODEL FAILURE
        # ====================================================

        except Exception as e:

            print(
                f"\nModel failed: {model}"
            )

            print(
                f"Error: {e}"
            )

            print(
                "Trying next model..."
            )


    # ========================================================
    # ALL MODELS FAILED
    # ========================================================

    return {
        "answer": (
            "Unable to generate an answer right now. "
            "Please try again later."
        ),
        "sources": [],
        "model": None
    }


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 70)
    print("CAMPUSIQ GENERATOR TEST")
    print("=" * 70)


    # --------------------------------------------------------
    # TEST DOCUMENT
    # --------------------------------------------------------

    test_documents = [

        {
            "source": "attendance_policy.pdf",

            "page": 1,

            "chunk_id": 1,

            "score": 0.6271,

            "rerank_score": 6.9083,

            "text": """
1. Minimum Attendance Requirement

75% overall attendance across all courses.

75% per subject for B.Tech/BE programs.

Minimum 50% attendance per individual subject.
"""
        }
    ]


    # --------------------------------------------------------
    # TEST QUESTION
    # --------------------------------------------------------

    test_question = (
        "What is the minimum attendance requirement?"
    )


    # --------------------------------------------------------
    # GENERATE ANSWER
    # --------------------------------------------------------

    result = generate_answer(
        test_question,
        test_documents
    )


    # ========================================================
    # DISPLAY RESULT
    # ========================================================

    print("\n" + "=" * 70)
    print("ANSWER")
    print("=" * 70)

    print(
        result["answer"]
    )


    print("\n" + "=" * 70)
    print("SOURCES")
    print("=" * 70)


    for source in result["sources"]:

        print(
            f"Document : {source['source']}"
        )

        print(
            f"Page     : {source['page']}"
        )

        print(
            f"Chunk    : {source['chunk_id']}"
        )

        print(
            f"Vector   : {source['vector_score']}"
        )

        print(
            f"Rerank   : {source['rerank_score']}"
        )

        print("-" * 40)


    print("\n" + "=" * 70)
    print("MODEL")
    print("=" * 70)

    print(
        result["model"]
    )

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
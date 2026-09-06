import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)


MODELS = [
    "minimax/minimax-m3:free",
    "inclusionai/ling-3.0-flash-fin:free",
    "google/gemma-4-26b-a4b-it:free",
    "z-ai/glm-5.2:free",
    "nvidia/nemotron-3.5-lightning:free",
]


def generate_answer(query, documents):

    if not documents:
        return (
            "I couldn't find reliable information about that "
            "in the campus knowledge base."
        )

    context_parts = []

    for i, document in enumerate(documents, start=1):

        context_parts.append(
            f"""
SOURCE {i}
Document: {document['source']}
Page: {document['page']}

Content:
{document['text']}
"""
        )

    context = "\n".join(context_parts)

    prompt = f"""
You are CampusIQ, an AI-powered campus knowledge assistant.

Answer the student's question using ONLY the provided campus
knowledge.

STRICT RULES:

1. Do not use outside knowledge.
2. Do not invent information.
3. If the answer cannot be found in the provided context,
   say:
   "I couldn't find reliable information about that in the
   campus knowledge base."
4. Give a concise and clear answer.
5. Mention the relevant source document and page.
6. Do not mention unrelated information from the context.

CAMPUS KNOWLEDGE:

{context}

STUDENT QUESTION:

{query}

ANSWER:
"""

    for model in MODELS:

        try:

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
            )

            return response.choices[0].message.content

        except Exception as e:

            print(f"\nModel failed: {model}")
            print(f"Error: {e}")
            print("Trying next model...\n")

    return "Unable to generate an answer right now."
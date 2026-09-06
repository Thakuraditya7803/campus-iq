import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)


def generate_answer(question, retrieved_documents):

    context = "\n\n".join(
        [
            f"Source: {doc['source']}\n"
            f"Page: {doc['page']}\n"
            f"Content: {doc['text']}"
            for doc in retrieved_documents
        ]
    )

    prompt = f"""
You are CampusIQ, an AI-powered campus knowledge assistant.

Your job is to answer questions using ONLY the provided campus
knowledge base.

Rules:

1. Do not invent information.
2. Do not use outside knowledge.
3. If the answer cannot be found in the context, say:
   "I couldn't find reliable information about that in the campus knowledge base."
4. Always provide the source document and page.
5. Give a concise and clear answer.
6. If the policy contains multiple requirements, mention the relevant ones.

Campus Knowledge:

{context}

User Question:

{question}
"""

    response = client.chat.completions.create(
        model="inclusionai/ling-3.0-flash-fin:free",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
    )

    return response.choices[0].message.content
import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
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

Answer the user's question using ONLY the provided campus context.

If the answer cannot be found in the context, say:
"I couldn't find reliable information about that in the campus knowledge base."

Do not invent rules, dates, policies, or requirements.

Always mention the source and page used.

Campus Context:
{context}

User Question:
{question}
"""

    response = client.responses.create(
        model="gpt-5-mini",
        input=prompt
    )

    return response.output_text
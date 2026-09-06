import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)


class EmbeddingModel:

    def __init__(self):
        self.model = "nvidia/nemotron-3-embed-1b:free"

    def generate_embedding(self, text: str):

        response = client.embeddings.create(
            model=self.model,
            input=text,
            encoding_format="float",
        )

        return response.data[0].embedding


if __name__ == "__main__":

    model = EmbeddingModel()

    text = "Students must maintain 75% attendance."

    embedding = model.generate_embedding(text)

    print("Embedding generated!")
    print("Embedding dimensions:", len(embedding))
    print("First 5 values:", embedding[:5])
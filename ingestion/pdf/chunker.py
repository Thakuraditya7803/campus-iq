def chunk_text(text, chunk_size=1000, overlap=200):
    """
    Split text into overlapping chunks.

    chunk_size: maximum characters per chunk
    overlap: characters shared between consecutive chunks
    """

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


if __name__ == "__main__":

    sample_text = """
    Students must maintain 75% overall attendance across all courses.
    Students in B.Tech and BE programs must maintain 75% attendance
    per subject. A minimum of 50% attendance is required per individual
    subject.
    """

    chunks = chunk_text(sample_text)

    for i, chunk in enumerate(chunks, start=1):
        print(f"\n--- Chunk {i} ---")
        print(chunk)
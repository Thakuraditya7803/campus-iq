import re


def chunk_text(text, chunk_size=1000, overlap=200):
    """
    Create section-aware chunks from extracted PDF text.
    """

    if not text:
        return []

    # Normalize whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n+", "\n", text)

    # Detect numbered sections.
    sections = re.split(
        r"(?=\n?\s*\d+\.\s+[A-Z])",
        text
    )

    sections = [
        section.strip()
        for section in sections
        if section.strip()
    ]

    chunks = []

    for section in sections:

        # If section is small enough, keep it together
        if len(section) <= chunk_size:

            chunks.append(section)

        else:

            start = 0

            while start < len(section):

                end = start + chunk_size

                chunk = section[start:end].strip()

                if chunk:
                    chunks.append(chunk)

                start += chunk_size - overlap

    return chunks


if __name__ == "__main__":

    sample_text = """
Universal SkillTech University (USTU)

Attendance Policy

1. Minimum Attendance Requirement

75% overall attendance across all courses.
75% per subject for B.Tech/BE programs.
Minimum 50% attendance per individual subject.

2. Consequences of Falling Below 75%

Student is placed on the detained list.
Barred from appearing in university examinations.

3. Condonation / Relaxation

Up to 5–10% relaxation may be granted.
Medical emergency, NCC/NSS duty, inter-college events.
"""

    chunks = chunk_text(sample_text)

    print(f"Generated {len(chunks)} chunks")

    for i, chunk in enumerate(chunks, start=1):

        print(f"\n--- Chunk {i} ---")
        print(chunk)
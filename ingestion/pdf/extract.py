from pathlib import Path
from pypdf import PdfReader


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from all pages of a PDF."""

    reader = PdfReader(pdf_path)

    text = []

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text.append(page_text)

    return "\n".join(text)


if __name__ == "__main__":
    pdf_folder = Path("data/raw")

    for pdf_file in pdf_folder.glob("*.pdf"):
        print(f"\nProcessing: {pdf_file.name}")

        text = extract_text_from_pdf(str(pdf_file))

        print(f"Characters extracted: {len(text)}")
        print("\n--- PREVIEW ---")
        print(text[:1000])
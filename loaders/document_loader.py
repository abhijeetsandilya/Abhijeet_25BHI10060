from pathlib import Path
from pypdf import PdfReader
from docx import Document


def load_document(file_path: str) -> str:

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    file_extension = path.suffix.lower()

    # Load PDF
    if file_extension == ".pdf":
        reader = PdfReader(file_path)

        text = ""
        for page in reader.pages:
            extracted_text = page.extract_text()

            if extracted_text:
                text += extracted_text + "\n"

        return text

    # Load TXT
    elif file_extension == ".txt":
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()

    # Load DOCX
    elif file_extension == ".docx":
        document = Document(file_path)

        text = "\n".join(
            paragraph.text for paragraph in document.paragraphs
        )

        return text

    else:
        raise ValueError(
            f"Unsupported file format: {file_extension}. "
            "Supported formats are PDF, TXT, and DOCX."
        )
import pdfplumber
# NEW - replace with this
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

def load_pdf(pdf_path: str) -> str:
    """
    Reads a PDF file and extracts all text from it.
    Returns the full text as a single string.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found at path: {pdf_path}")
    
    full_text = ""
    
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:  # some pages may be empty or image-only
                full_text += f"\n[Page {page_num + 1}]\n{text}"
    
    if not full_text.strip():
        raise ValueError("No text could be extracted from this PDF.")
    
    print(f"✅ PDF loaded: {len(pdf.pages)} pages, {len(full_text)} characters extracted")
    return full_text


def chunk_text(full_text: str) -> list:
    """
    Splits the full text into overlapping chunks.
    - chunk_size: 500 tokens
    - chunk_overlap: 50 tokens (prevents context loss at boundaries)
    Uses RecursiveCharacterTextSplitter which respects sentence boundaries.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
        # tries to split at paragraphs first, then sentences, then words
    )
    
    chunks = splitter.split_text(full_text)
    
    print(f"✅ Text chunked: {len(chunks)} chunks created")
    return chunks


def load_and_chunk_pdf(pdf_path: str) -> list:
    """
    Main function — loads PDF and returns chunks in one call.
    This is what the rest of the pipeline will use.
    """
    print(f"\n📄 Loading PDF: {pdf_path}")
    full_text = load_pdf(pdf_path)
    chunks = chunk_text(full_text)
    return chunks
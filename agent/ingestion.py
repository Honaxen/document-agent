"""
ingestion.py
------------
Document loading, cleaning, and chunking.

Supports: .txt, .pdf files
"""

import re
from pathlib import Path


def load_document(path: str) -> str:
    """
    Load a document from file.
    Supports .txt and .pdf formats.

    Args:
        path: Path to the document file

    Returns:
        Raw text content of the document
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")

    if path.suffix == '.txt':
        return path.read_text(encoding='utf-8')

    elif path.suffix == '.pdf':
        try:
            import PyPDF2
            text = ""
            with open(path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
            return text
        except ImportError:
            raise ImportError("pip install pypdf2 to load PDF files")

    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


def clean_text(text: str) -> str:
    """
    Clean raw document text.
    - Normalize whitespace
    - Remove null bytes
    - Strip leading/trailing spaces
    """
    text = re.sub(r'\x00', '', text)
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r' +', ' ', text)
    return text.strip()


def chunk_document(text: str, sentences_per_chunk: int = 4,
                   min_chunk_length: int = 50) -> list:
    """
    Split document into chunks by sentence boundaries.

    Args:
        text: Cleaned document text
        sentences_per_chunk: Number of sentences per chunk
        min_chunk_length: Minimum characters for a valid chunk

    Returns:
        List of text chunks
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []

    for i in range(0, len(sentences), sentences_per_chunk):
        chunk = ' '.join(sentences[i:i + sentences_per_chunk])
        if len(chunk.strip()) >= min_chunk_length:
            chunks.append(chunk.strip())

    return chunks


def ingest(path: str, sentences_per_chunk: int = 4) -> list:
    """
    Full ingestion pipeline: load -> clean -> chunk.

    Args:
        path: Path to document file
        sentences_per_chunk: Chunk size in sentences

    Returns:
        List of text chunks ready for embedding
    """
    raw = load_document(path)
    cleaned = clean_text(raw)
    chunks = chunk_document(cleaned, sentences_per_chunk)
    return chunks
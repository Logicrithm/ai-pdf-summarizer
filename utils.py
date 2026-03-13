"""
Text Utilities — Step 2
-----------------------
Chunking with word-level overlap and page metadata.
"""

import re


def clean_text(text: str) -> str:
    """Normalize whitespace and remove noise characters."""
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def chunk_pages(pages: list[dict], chunk_size: int = 800, overlap: int = 100) -> list[dict]:
    """
    Split extracted pages into chunks with word-level overlap.

    Each chunk stores:
        {"text": chunk_text, "page": page_number, "chunk_id": index}

    Args:
        pages: List of dicts from extract_pages() with 'text' and 'page'.
        chunk_size: Words per chunk.
        overlap: Overlapping words between chunks.

    Returns:
        List of chunk dicts.
    """
    # Build a flat list of (word, page_number) pairs
    word_page_pairs = []
    for page_info in pages:
        cleaned = clean_text(page_info["text"])
        words = cleaned.split()
        for word in words:
            word_page_pairs.append((word, page_info["page"]))

    if not word_page_pairs:
        return []

    chunks = []
    step = max(chunk_size - overlap, 1)
    chunk_id = 0

    for i in range(0, len(word_page_pairs), step):
        window = word_page_pairs[i : i + chunk_size]
        if not window:
            break

        chunk_text = " ".join(w for w, _ in window)
        # Primary page = most common page in this chunk
        page_counts = {}
        for _, page in window:
            page_counts[page] = page_counts.get(page, 0) + 1
        primary_page = max(page_counts, key=page_counts.get)

        chunks.append({
            "text": chunk_text,
            "page": primary_page,
            "chunk_id": chunk_id,
        })
        chunk_id += 1

        if i + chunk_size >= len(word_page_pairs):
            break

    return chunks

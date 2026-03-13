"""
Cache System — Step 4
---------------------
Pickle-based caching with MD5 hash of PDF content.
Stores chunks, embeddings, FAISS index, and summary.
"""

import hashlib
import pickle
import os
import numpy as np
import faiss


CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")


def get_cache_path(pdf_bytes: bytes) -> str:
    """Get cache file path based on MD5 hash of PDF content."""
    pdf_hash = hashlib.md5(pdf_bytes).hexdigest()
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{pdf_hash}.pkl")


def get_file_hash(pdf_bytes: bytes) -> str:
    """Compute MD5 hash of PDF bytes."""
    return hashlib.md5(pdf_bytes).hexdigest()


def save_cache(pdf_bytes: bytes, chunks: list[dict], embeddings: np.ndarray,
               faiss_index: faiss.IndexFlatL2, summary: str = None) -> None:
    """
    Save processing artifacts to disk.

    Stored: chunks, embeddings, FAISS index, summary.
    """
    cache_path = get_cache_path(pdf_bytes)

    # Serialize FAISS index to bytes
    faiss_bytes = faiss.serialize_index(faiss_index)

    data = {
        "chunks": chunks,
        "embeddings": embeddings,
        "faiss_index_bytes": faiss_bytes,
        "summary": summary,
    }

    with open(cache_path, "wb") as f:
        pickle.dump(data, f)


def load_cache(pdf_bytes: bytes) -> dict | None:
    """
    Load cached artifacts if they exist.

    Returns:
        Dict with keys: chunks, embeddings, faiss_index, summary.
        Or None if no cache exists.
    """
    cache_path = get_cache_path(pdf_bytes)

    if not os.path.exists(cache_path):
        return None

    with open(cache_path, "rb") as f:
        data = pickle.load(f)

    # Deserialize FAISS index
    data["faiss_index"] = faiss.deserialize_index(data.pop("faiss_index_bytes"))

    return data

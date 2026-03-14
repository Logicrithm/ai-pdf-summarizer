"""
Embedding Engine — Step 3
--------------------------
Local embeddings using sentence-transformers (all-MiniLM-L6-v2).
FAISS IndexFlatL2 for vector search.
Completely free — no API key needed.
"""

import numpy as np
import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor


# Load model once and cache across reruns
@st.cache_resource
def _load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

_embed_model = _load_model()


def embed_chunks(client, chunks: list[dict], max_workers: int = 4) -> np.ndarray:
    """
    Embed all chunks using local sentence-transformers model.

    Args:
        client: Unused (kept for interface compatibility).
        chunks: List of chunk dicts with 'text' key.
        max_workers: Unused (batch encoding is already efficient).

    Returns:
        numpy array of shape (num_chunks, embedding_dim).
    """
    texts = [chunk["text"] for chunk in chunks]
    embeddings = _embed_model.encode(texts, show_progress_bar=False)
    return np.array(embeddings, dtype="float32")


def embed_query(client, query: str) -> np.ndarray:
    """Embed a single query string."""
    embedding = _embed_model.encode(query)
    return np.array([embedding], dtype="float32")


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    """Build a FAISS L2 index from embeddings."""
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index


def search_faiss(index: faiss.IndexFlatL2, query_embedding: np.ndarray, k: int = 5):
    """
    Search FAISS for top-k results.

    Returns:
        Tuple of (distances, indices) arrays.
    """
    k = min(k, index.ntotal)
    distances, indices = index.search(query_embedding, k)
    return distances[0], indices[0]

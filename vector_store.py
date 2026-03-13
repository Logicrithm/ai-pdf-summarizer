"""
Vector Store
-------------
Lightweight vector search using numpy cosine similarity.
Includes confidence filtering to reject out-of-scope questions.
"""

import numpy as np

# Similarity threshold — chunks below this score are considered irrelevant.
# Range: 0.0 (unrelated) to 1.0 (identical). Tuned for nomic-embed-text.
CONFIDENCE_THRESHOLD = 0.3


class VectorIndex:
    """In-memory vector index with cosine similarity and confidence filtering."""

    def __init__(self, embeddings: np.ndarray):
        """
        Args:
            embeddings: numpy array of shape (num_chunks, embedding_dim).
        """
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        self.embeddings = embeddings / norms
        self.count = embeddings.shape[0]

    def search(self, query_embedding: np.ndarray, k: int = 3) -> tuple[list[int], list[float]]:
        """
        Find the top-k most similar chunks using cosine similarity.

        Args:
            query_embedding: numpy array of shape (1, embedding_dim).
            k: Number of results to return.

        Returns:
            Tuple of (indices, scores):
            - indices: chunk indices sorted by relevance
            - scores: cosine similarity scores (0-1)
        """
        k = min(k, self.count)

        norm = np.linalg.norm(query_embedding)
        if norm > 0:
            query_normalized = query_embedding / norm
        else:
            query_normalized = query_embedding

        similarities = np.dot(self.embeddings, query_normalized.flatten())

        top_indices = np.argsort(similarities)[::-1][:k]
        top_scores = similarities[top_indices].tolist()

        return top_indices.tolist(), top_scores

    def search_with_filter(self, query_embedding: np.ndarray, k: int = 3) -> tuple[list[int], list[float]]:
        """
        Search with confidence filtering — reject low-relevance chunks.

        Returns:
            Tuple of (filtered_indices, filtered_scores).
            May return fewer than k results if chunks are below threshold.
        """
        indices, scores = self.search(query_embedding, k)

        filtered_indices = []
        filtered_scores = []
        for idx, score in zip(indices, scores):
            if score >= CONFIDENCE_THRESHOLD:
                filtered_indices.append(idx)
                filtered_scores.append(score)

        return filtered_indices, filtered_scores


def build_index(embeddings: np.ndarray) -> VectorIndex:
    """Build a vector index from embeddings."""
    return VectorIndex(embeddings)

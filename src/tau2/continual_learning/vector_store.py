"""
Vector Store Abstraction for RAG-based Continual Learning.

This module provides a unified interface for different vector database backends
(ChromaDB, FAISS, etc.) to support semantic memory retrieval in CL methods.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import json
import numpy as np
from pathlib import Path


class VectorStore(ABC):
    """Abstract base class for vector stores."""

    @abstractmethod
    def add(
        self,
        id: str,
        embedding: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
        document: Optional[str] = None,
    ):
        """
        Add a vector to the store.

        Args:
            id: Unique identifier for the vector
            embedding: The embedding vector
            metadata: Optional metadata to associate with the vector
            document: Optional document text
        """
        pass

    @abstractmethod
    def add_batch(
        self,
        ids: List[str],
        embeddings: np.ndarray,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        documents: Optional[List[str]] = None,
    ):
        """
        Add multiple vectors to the store.

        Args:
            ids: List of unique identifiers
            embeddings: Array of embedding vectors
            metadatas: Optional list of metadata dicts
            documents: Optional list of document texts
        """
        pass

    @abstractmethod
    def query(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Query the store for similar vectors.

        Args:
            query_embedding: The query vector
            k: Number of results to return
            filter: Optional metadata filter

        Returns:
            List of results with 'id', 'distance', 'metadata', 'document'
        """
        pass

    @abstractmethod
    def delete(self, id: str):
        """Delete a vector by ID."""
        pass

    @abstractmethod
    def clear(self):
        """Clear all vectors from the store."""
        pass

    @abstractmethod
    def save(self, path: str):
        """Save the store to disk."""
        pass

    @abstractmethod
    def load(self, path: str):
        """Load the store from disk."""
        pass

    def __len__(self):
        """Return the number of vectors in the store."""
        return 0


class InMemoryVectorStore(VectorStore):
    """
    Simple in-memory vector store using numpy for similarity search.

    Uses cosine similarity for retrieval. Suitable for small to medium
    datasets (up to ~10k vectors). For larger datasets, use FAISS or ChromaDB.
    """

    def __init__(self):
        """Initialize the in-memory store."""
        self.ids = []
        self.embeddings = None  # Will be np.ndarray of shape (n, dim)
        self.metadatas = []
        self.documents = []
        self.id_to_index = {}

    def add(
        self,
        id: str,
        embedding: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
        document: Optional[str] = None,
    ):
        """Add a single vector."""
        # Convert to numpy if needed
        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding)

        # Ensure 1D
        if embedding.ndim > 1:
            embedding = embedding.flatten()

        # Remove existing if duplicate ID
        if id in self.id_to_index:
            self.delete(id)

        # Add to store
        if self.embeddings is None:
            self.embeddings = embedding.reshape(1, -1)
        else:
            self.embeddings = np.vstack([self.embeddings, embedding])

        self.ids.append(id)
        self.metadatas.append(metadata or {})
        self.documents.append(document or "")
        self.id_to_index[id] = len(self.ids) - 1

    def add_batch(
        self,
        ids: List[str],
        embeddings: np.ndarray,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        documents: Optional[List[str]] = None,
    ):
        """Add multiple vectors efficiently."""
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings)

        # Ensure 2D
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        n = len(ids)
        metadatas = metadatas or [{} for _ in range(n)]
        documents = documents or ["" for _ in range(n)]

        for i in range(n):
            self.add(ids[i], embeddings[i], metadatas[i], documents[i])

    def query(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Query using cosine similarity."""
        if self.embeddings is None or len(self.ids) == 0:
            return []

        # Convert query to numpy
        if not isinstance(query_embedding, np.ndarray):
            query_embedding = np.array(query_embedding)

        # Ensure 1D
        if query_embedding.ndim > 1:
            query_embedding = query_embedding.flatten()

        # Compute cosine similarities
        query_norm = np.linalg.norm(query_embedding)
        embeddings_norm = np.linalg.norm(self.embeddings, axis=1)

        # Avoid division by zero
        query_norm = max(query_norm, 1e-10)
        embeddings_norm = np.maximum(embeddings_norm, 1e-10)

        # Cosine similarity
        similarities = np.dot(self.embeddings, query_embedding) / (
            embeddings_norm * query_norm
        )

        # Convert to distances (1 - similarity for consistency with other stores)
        distances = 1 - similarities

        # Apply metadata filter if provided
        valid_indices = list(range(len(self.ids)))
        if filter:
            valid_indices = [
                i for i in valid_indices
                if self._matches_filter(self.metadatas[i], filter)
            ]

        # Sort by distance (ascending)
        valid_distances = [(i, distances[i]) for i in valid_indices]
        valid_distances.sort(key=lambda x: x[1])

        # Take top k
        top_k = valid_distances[:k]

        # Format results
        results = []
        for idx, dist in top_k:
            results.append({
                'id': self.ids[idx],
                'distance': float(dist),
                'similarity': float(1 - dist),
                'metadata': self.metadatas[idx],
                'document': self.documents[idx],
            })

        return results

    def _matches_filter(self, metadata: Dict[str, Any], filter: Dict[str, Any]) -> bool:
        """Check if metadata matches filter criteria."""
        for key, value in filter.items():
            if key not in metadata:
                return False
            if metadata[key] != value:
                return False
        return True

    def delete(self, id: str):
        """Delete a vector by ID."""
        if id not in self.id_to_index:
            return

        idx = self.id_to_index[id]

        # Remove from arrays
        self.ids.pop(idx)
        self.metadatas.pop(idx)
        self.documents.pop(idx)

        if self.embeddings is not None:
            self.embeddings = np.delete(self.embeddings, idx, axis=0)
            if self.embeddings.shape[0] == 0:
                self.embeddings = None

        # Rebuild index
        self.id_to_index = {id: i for i, id in enumerate(self.ids)}

    def clear(self):
        """Clear all data."""
        self.ids = []
        self.embeddings = None
        self.metadatas = []
        self.documents = []
        self.id_to_index = {}

    def save(self, path: str):
        """Save to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save metadata
        metadata = {
            'ids': self.ids,
            'metadatas': self.metadatas,
            'documents': self.documents,
        }
        with open(path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        # Save embeddings
        if self.embeddings is not None:
            np.save(path / 'embeddings.npy', self.embeddings)

    def load(self, path: str):
        """Load from disk."""
        path = Path(path)

        # Load metadata
        with open(path / 'metadata.json', 'r') as f:
            metadata = json.load(f)

        self.ids = metadata['ids']
        self.metadatas = metadata['metadatas']
        self.documents = metadata['documents']

        # Load embeddings
        embeddings_path = path / 'embeddings.npy'
        if embeddings_path.exists():
            self.embeddings = np.load(embeddings_path)
        else:
            self.embeddings = None

        # Rebuild index
        self.id_to_index = {id: i for i, id in enumerate(self.ids)}

    def __len__(self):
        """Return number of vectors."""
        return len(self.ids)


class FAISSVectorStore(VectorStore):
    """
    FAISS-based vector store for efficient similarity search.

    Requires: pip install faiss-cpu (or faiss-gpu)
    """

    def __init__(self, dimension: int = 768, index_type: str = "flatl2"):
        """
        Initialize FAISS store.

        Args:
            dimension: Dimensionality of embeddings
            index_type: Type of FAISS index ('flatl2', 'flatip', 'ivfflat', 'hnsw')
        """
        try:
            import faiss
        except ImportError:
            raise ImportError(
                "FAISS not installed. Install with: pip install faiss-cpu"
            )

        self.dimension = dimension
        self.index_type = index_type
        self.ids = []
        self.metadatas = []
        self.documents = []
        self.id_to_index = {}

        # Create FAISS index
        if index_type == "flatl2":
            self.index = faiss.IndexFlatL2(dimension)
        elif index_type == "flatip":
            self.index = faiss.IndexFlatIP(dimension)
        elif index_type == "ivfflat":
            quantizer = faiss.IndexFlatL2(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, 100)
        elif index_type == "hnsw":
            self.index = faiss.IndexHNSWFlat(dimension, 32)
        else:
            raise ValueError(f"Unknown index type: {index_type}")

    def add(
        self,
        id: str,
        embedding: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
        document: Optional[str] = None,
    ):
        """Add a single vector."""
        import faiss

        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding, dtype=np.float32)

        if embedding.dtype != np.float32:
            embedding = embedding.astype(np.float32)

        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)

        # Remove existing if duplicate
        if id in self.id_to_index:
            self.delete(id)

        # Add to FAISS
        self.index.add(embedding)

        # Add metadata
        self.ids.append(id)
        self.metadatas.append(metadata or {})
        self.documents.append(document or "")
        self.id_to_index[id] = len(self.ids) - 1

    def add_batch(
        self,
        ids: List[str],
        embeddings: np.ndarray,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        documents: Optional[List[str]] = None,
    ):
        """Add multiple vectors."""
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings, dtype=np.float32)

        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)

        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        n = len(ids)
        metadatas = metadatas or [{} for _ in range(n)]
        documents = documents or ["" for _ in range(n)]

        # Add to FAISS
        self.index.add(embeddings)

        # Add metadata
        for i in range(n):
            self.ids.append(ids[i])
            self.metadatas.append(metadatas[i])
            self.documents.append(documents[i])
            self.id_to_index[ids[i]] = len(self.ids) - 1

    def query(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Query using FAISS."""
        if len(self.ids) == 0:
            return []

        if not isinstance(query_embedding, np.ndarray):
            query_embedding = np.array(query_embedding, dtype=np.float32)

        if query_embedding.dtype != np.float32:
            query_embedding = query_embedding.astype(np.float32)

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Search FAISS
        distances, indices = self.index.search(query_embedding, min(k * 2, len(self.ids)))

        # Flatten results
        distances = distances[0]
        indices = indices[0]

        # Filter and format
        results = []
        for dist, idx in zip(distances, indices):
            if idx == -1:  # FAISS uses -1 for invalid results
                continue

            # Apply metadata filter
            if filter and not self._matches_filter(self.metadatas[idx], filter):
                continue

            results.append({
                'id': self.ids[idx],
                'distance': float(dist),
                'similarity': float(1.0 / (1.0 + dist)),  # Convert distance to similarity
                'metadata': self.metadatas[idx],
                'document': self.documents[idx],
            })

            if len(results) >= k:
                break

        return results

    def _matches_filter(self, metadata: Dict[str, Any], filter: Dict[str, Any]) -> bool:
        """Check if metadata matches filter."""
        for key, value in filter.items():
            if key not in metadata or metadata[key] != value:
                return False
        return True

    def delete(self, id: str):
        """
        Delete not directly supported in FAISS.
        Need to rebuild index without the deleted vector.
        """
        raise NotImplementedError("FAISS does not support efficient deletion. Use clear() and re-add.")

    def clear(self):
        """Clear all data."""
        self.index.reset()
        self.ids = []
        self.metadatas = []
        self.documents = []
        self.id_to_index = {}

    def save(self, path: str):
        """Save to disk."""
        import faiss

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, str(path / 'index.faiss'))

        # Save metadata
        metadata = {
            'ids': self.ids,
            'metadatas': self.metadatas,
            'documents': self.documents,
            'dimension': self.dimension,
            'index_type': self.index_type,
        }
        with open(path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

    def load(self, path: str):
        """Load from disk."""
        import faiss

        path = Path(path)

        # Load FAISS index
        self.index = faiss.read_index(str(path / 'index.faiss'))

        # Load metadata
        with open(path / 'metadata.json', 'r') as f:
            metadata = json.load(f)

        self.ids = metadata['ids']
        self.metadatas = metadata['metadatas']
        self.documents = metadata['documents']
        self.dimension = metadata['dimension']
        self.index_type = metadata['index_type']
        self.id_to_index = {id: i for i, id in enumerate(self.ids)}

    def __len__(self):
        """Return number of vectors."""
        return len(self.ids)


def create_vector_store(store_type: str = "inmemory", **kwargs) -> VectorStore:
    """
    Factory function to create vector stores.

    Args:
        store_type: Type of store ('inmemory', 'faiss')
        **kwargs: Additional arguments for the store

    Returns:
        VectorStore instance
    """
    if store_type == "inmemory":
        return InMemoryVectorStore()
    elif store_type == "faiss":
        return FAISSVectorStore(**kwargs)
    else:
        raise ValueError(f"Unknown store type: {store_type}")

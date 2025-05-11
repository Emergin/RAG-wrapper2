import faiss
import numpy as np
import os
import pickle
from typing import List, Dict, Tuple, Any

class VectorStore:
    """Vector database for storing and retrieving document embeddings"""
    
    def __init__(self, embedding_dim: int = None):
        """Initialize the vector store with a specified embedding dimension"""
        self.index = None
        self.chunks = []
        self.file_sources = []
        self.embedding_dim = embedding_dim
        
    def init_index(self, embedding_dim: int) -> None:
        """Initialize the FAISS index with the specified dimension"""
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatL2(embedding_dim)
        
    def add_documents(self, embeddings: np.ndarray, chunks: List[str], source: str) -> None:
        """Add document embeddings and metadata to the vector store"""
        if self.index is None:
            self.init_index(embeddings.shape[1])
        
        if embeddings.shape[0] != len(chunks):
            raise ValueError(f"Number of embeddings ({embeddings.shape[0]}) doesn't match number of chunks ({len(chunks)})")
        
        # Convert embeddings to float32 if needed
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        
        # Add to FAISS index
        self.index.add(embeddings)
        
        # Store metadata
        self.chunks.extend(chunks)
        self.file_sources.extend([source] * len(chunks))
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """Search for the k most similar documents to the query embedding"""
        if self.index is None:
            raise ValueError("Index not initialized. Add documents first.")
        
        # Ensure the query is the right shape and type
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        if query_embedding.dtype != np.float32:
            query_embedding = query_embedding.astype(np.float32)
        
        # Search the index
        distances, indices = self.index.search(query_embedding, k)
        
        # Prepare results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.chunks) and idx >= 0:  # Ensure index is valid
                results.append({
                    'chunk': self.chunks[idx],
                    'score': float(distances[0][i]),
                    'source': self.file_sources[idx],
                    'index': int(idx)
                })
        
        return results
    
    def save(self, directory: str) -> None:
        """Save the vector store to disk"""
        if self.index is None:
            raise ValueError("Cannot save an uninitialized index")
        
        os.makedirs(directory, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, os.path.join(directory, "faiss_index.bin"))
        
        # Save metadata
        with open(os.path.join(directory, "metadata.pkl"), 'wb') as f:
            pickle.dump({
                'chunks': self.chunks,
                'file_sources': self.file_sources,
                'embedding_dim': self.embedding_dim
            }, f)
    
    @classmethod
    def load(cls, directory: str) -> 'VectorStore':
        """Load a vector store from disk"""
        # Load metadata
        with open(os.path.join(directory, "metadata.pkl"), 'rb') as f:
            metadata = pickle.load(f)
        
        # Create instance
        store = cls()
        store.chunks = metadata['chunks']
        store.file_sources = metadata['file_sources']
        store.embedding_dim = metadata['embedding_dim']
        
        # Load FAISS index
        store.index = faiss.read_index(os.path.join(directory, "faiss_index.bin"))
        
        return store
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store"""
        if self.index is None:
            return {"status": "Not initialized"}
        
        return {
            "document_count": len(self.chunks),
            "unique_sources": len(set(self.file_sources)),
            "embedding_dimension": self.embedding_dim,
            "sources": list(set(self.file_sources))
        }
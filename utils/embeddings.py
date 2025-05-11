import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Dict, Tuple, Any
import pickle
import os

class TFIDFEmbedder:
    """Class for generating and managing TF-IDF embeddings for document chunks"""
    
    def __init__(self, max_features: int = 5000):
        """Initialize the TF-IDF vectorizer"""
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2),
            lowercase=True,
            norm='l2'
        )
        self.fitted = False
        
    def fit(self, documents: List[str]) -> None:
        """Fit the TF-IDF vectorizer on the provided documents"""
        if not documents:
            raise ValueError("No documents provided for fitting the vectorizer")
        
        self.vectorizer.fit(documents)
        self.fitted = True
        
    def transform(self, documents: List[str]) -> np.ndarray:
        """Transform documents into TF-IDF embeddings"""
        if not self.fitted:
            raise ValueError("Vectorizer must be fitted before transformation")
            
        return self.vectorizer.transform(documents).toarray()
    
    def fit_transform(self, documents: List[str]) -> np.ndarray:
        """Fit the vectorizer and transform documents in one step"""
        if not documents:
            raise ValueError("No documents provided for fitting and transformation")
            
        self.fitted = True
        return self.vectorizer.fit_transform(documents).toarray()
    
    def get_feature_names(self) -> List[str]:
        """Get the feature names (terms) used by the vectorizer"""
        if not self.fitted:
            raise ValueError("Vectorizer must be fitted before accessing feature names")
            
        return self.vectorizer.get_feature_names_out()
    
    def save(self, file_path: str) -> None:
        """Save the fitted vectorizer to disk"""
        if not self.fitted:
            raise ValueError("Cannot save an unfitted vectorizer")
            
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
    
    @classmethod
    def load(cls, file_path: str) -> 'TFIDFEmbedder':
        """Load a saved vectorizer from disk"""
        with open(file_path, 'rb') as f:
            vectorizer = pickle.load(f)
        
        embedder = cls()
        embedder.vectorizer = vectorizer
        embedder.fitted = True
        return embedder

    def get_most_important_terms(self, doc_idx: int, embeddings: np.ndarray, top_n: int = 10) -> List[Tuple[str, float]]:
        """Get the most important terms for a specific document based on TF-IDF scores"""
        if not self.fitted:
            raise ValueError("Vectorizer must be fitted before extracting important terms")
            
        feature_names = self.get_feature_names()
        
        # Get TF-IDF scores for the document
        tfidf_scores = embeddings[doc_idx]
        
        # Get indices of top terms sorted by TF-IDF score
        top_indices = np.argsort(tfidf_scores)[::-1][:top_n]
        
        # Get terms and their scores
        top_terms = [(feature_names[i], tfidf_scores[i]) for i in top_indices]
        
        return top_terms
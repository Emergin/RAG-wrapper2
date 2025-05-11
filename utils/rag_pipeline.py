import os
import openai
from typing import List, Dict, Any
import numpy as np
from dotenv import load_dotenv

from .vector_store import VectorStore
from .embeddings import TFIDFEmbedder

# Load environment variables
load_dotenv()

class RAGPipeline:
    """Retrieval-Augmented Generation pipeline using TF-IDF and OpenAI GPT"""
    
    def __init__(self, vector_store: VectorStore, embedder: TFIDFEmbedder):
        """Initialize the RAG pipeline with the vector store and embedder"""
        self.vector_store = vector_store
        self.embedder = embedder
        self.api_key = os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
            
        openai.api_key = self.api_key
    
    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant document chunks for the query"""
        # Transform query to embedding using the same TF-IDF vectorizer
        query_embedding = self.embedder.transform([query])
        
        # Search for relevant chunks
        results = self.vector_store.search(query_embedding, k=k)
        
        return results
    
    def generate(self, query: str, retrieved_docs: List[Dict[str, Any]], 
                 model: str = "gpt-3.5-turbo", 
                 temperature: float = 0.3) -> Dict[str, Any]:
        """Generate a response using GPT based on the query and retrieved documents"""
        # Prepare context from retrieved documents
        context = "\n\n".join([f"Document {i+1} (Source: {doc['source']}):\n{doc['chunk']}" 
                              for i, doc in enumerate(retrieved_docs)])
        
        # Create system prompt
        system_prompt = """You are an assistant that answers questions based on the provided documents. 
        Your answers should be based solely on the information in these documents. 
        If the information to answer the question is not in the documents, say "I don't have enough information to answer this question."
        Always cite your sources by referencing the document number."""
        
        # Create messages for the API call
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Here are some relevant documents:\n\n{context}\n\nBased on these documents, please answer the following question: {query}"}
        ]
        
        # Call the OpenAI API
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=1000
            )
            
            answer = response.choices[0].message.content
            
            return {
                "answer": answer,
                "sources": [doc["source"] for doc in retrieved_docs],
                "chunks": [doc["chunk"] for doc in retrieved_docs],
                "model": model,
                "tokens_used": response.usage.total_tokens
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "answer": "An error occurred while generating the response."
            }
    
    def query(self, query: str, k: int = 5, model: str = "gpt-3.5-turbo", 
              temperature: float = 0.3) -> Dict[str, Any]:
        """Complete RAG pipeline: retrieve documents and generate an answer"""
        # Retrieve relevant documents
        retrieved_docs = self.retrieve(query, k=k)
        
        # If no documents found, return early
        if not retrieved_docs:
            return {
                "answer": "No relevant documents found to answer this question.",
                "sources": [],
                "chunks": [],
                "model": model
            }
        
        # Generate answer
        result = self.generate(query, retrieved_docs, model=model, temperature=temperature)
        
        # Add query for reference
        result["query"] = query
        
        return result
    
    def analyze_retrieval(self, query: str, k: int = 5) -> Dict[str, Any]:
        """Analyze the retrieval process for explainability"""
        # Transform query to embedding
        query_embedding = self.embedder.transform([query])
        
        # Get the most important terms in the query
        query_terms = self.embedder.get_most_important_terms(0, query_embedding)
        
        # Retrieve documents
        retrieved_docs = self.retrieve(query, k=k)
        
        # For each retrieved document, get the most important terms
        doc_terms = []
        for doc in retrieved_docs:
            doc_idx = doc["index"]
            doc_embedding = self.embedder.transform([doc["chunk"]])
            terms = self.embedder.get_most_important_terms(0, doc_embedding)
            doc_terms.append({
                "source": doc["source"],
                "score": doc["score"],
                "terms": terms
            })
        
        return {
            "query": query,
            "query_terms": query_terms,
            "retrieved_docs": retrieved_docs,
            "doc_terms": doc_terms
        }
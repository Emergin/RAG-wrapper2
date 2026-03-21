import os
import google.generativeai as genai
from typing import List, Dict, Any
from dotenv import load_dotenv

from .vector_store import VectorStore
from .embeddings import TFIDFEmbedder

load_dotenv()

class RAGPipeline:
    """RAG pipeline using TF-IDF retrieval + Google Gemini generation"""

    def __init__(self, vector_store: VectorStore, embedder: TFIDFEmbedder):
        self.vector_store = vector_store
        self.embedder    = embedder
        self.api_key     = os.getenv("GEMINI_API_KEY", "")
        if self.api_key:
            genai.configure(api_key=self.api_key)

    def set_api_key(self, key: str):
        self.api_key = key
        genai.configure(api_key=key)

    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        query_embedding = self.embedder.transform([query])
        return self.vector_store.search(query_embedding, k=k)

    def generate(self, query: str, retrieved_docs: List[Dict[str, Any]],
                 model: str = "gemini-1.5-flash",
                 temperature: float = 0.3) -> Dict[str, Any]:

        context = "\n\n".join([
            f"Document {i+1} (Source: {doc['source']}):\n{doc['chunk']}"
            for i, doc in enumerate(retrieved_docs)
        ])

        prompt = f"""You are an assistant that answers questions based only on the provided documents.
If the answer is not in the documents, say "I don't have enough information to answer this question."
Always cite the document number you used.

Documents:
{context}

Question: {query}"""

        try:
            gemini = genai.GenerativeModel(
                model_name=model,
                generation_config={"temperature": temperature, "max_output_tokens": 1000}
            )
            response = gemini.generate_content(prompt)
            answer   = response.text

            return {
                "answer":  answer,
                "sources": [doc["source"] for doc in retrieved_docs],
                "chunks":  [doc["chunk"]  for doc in retrieved_docs],
                "model":   model,
            }
        except Exception as e:
            return {
                "error":  str(e),
                "answer": f"Error generating response: {str(e)}",
                "sources": [],
                "chunks":  [],
                "model":   model,
            }

    def query(self, query: str, k: int = 5,
              model: str = "gemini-1.5-flash",
              temperature: float = 0.3) -> Dict[str, Any]:

        retrieved_docs = self.retrieve(query, k=k)

        if not retrieved_docs:
            return {
                "answer":  "No relevant documents found.",
                "sources": [],
                "chunks":  [],
                "model":   model,
            }

        result          = self.generate(query, retrieved_docs, model=model, temperature=temperature)
        result["query"] = query
        return result

    def analyze_retrieval(self, query: str, k: int = 5) -> Dict[str, Any]:
        query_embedding = self.embedder.transform([query])
        query_terms     = self.embedder.get_most_important_terms(0, query_embedding)
        retrieved_docs  = self.retrieve(query, k=k)

        doc_terms = []
        for doc in retrieved_docs:
            doc_embedding = self.embedder.transform([doc["chunk"]])
            terms = self.embedder.get_most_important_terms(0, doc_embedding)
            doc_terms.append({
                "source": doc["source"],
                "score":  doc["score"],
                "terms":  terms,
            })

        return {
            "query":         query,
            "query_terms":   query_terms,
            "retrieved_docs": retrieved_docs,
            "doc_terms":     doc_terms,
        }

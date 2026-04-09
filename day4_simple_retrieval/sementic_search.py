"""
Day 4 Soft Assignment - Simple Semantic Retrieval (Core of RAG).

Implements semantic search using sentence-transformers and cosine similarity.
This is the foundation of every Retrieval-Augmented Generation (RAG) system."""

import numpy as np
from typing import List, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

class SemanticRetriever:
    def __init__(self, model_name='all-MiniLM-L6-v2')->None:
        print(f"Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.documents: List[str] = []
        self.embeddings: np.ndarray = np.array([])

    def load_documents(self, filepath: str = "documents.txt") -> None:
           """Loads documents from a text file, one document per line."""
           with open(filepath, 'r', encoding='utf-8') as f:
                self.documents = [line.strip() for line in f if line.strip()]
           print(f"Loaded {len(self.documents)} documents.")
   
    def generate_embeddings(self) -> None:
        """Convert all documents into vector embeddings."""
        print("Generating embeddings for documents...")
        self.embeddings = self.model.encode(self.documents, convert_to_numpy=True)
        print("Embeddings generated.\n")

    def retrieve_top_k(
              self, query: str, k: int = 3
            ) -> List[Tuple[int,str, float]]:
        """Given a query, return the top-k most similar documents."""          
        if len(self.documents) ==0 or len(self.embeddings) ==0:
            raise ValueError("Documents and embeddings must be loaded and generated before retrieval.")
    
        #encode query with the same model to get its embedding
        query_embedding = self.model.encode([query], convert_to_numpy=True)

        #cosine similarity between query embedding and document embeddings
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]

        #get indices of top-k most similar documents
        top_k_indices = np.argsort(similarities)[::-1][:k]
    
        results: List[Tuple[int, str, float]] = []
        for rank, idx in enumerate(top_k_indices, 1):
            results.append((rank, self.documents[idx], similarities[idx]))
        return results
    
if __name__ == "__main__":
    # Intialize the semantic search system, load documents, generate embeddings, and perform a sample query.
    retriever= SemanticRetriever()
    retriever.load_documents("documents.txt")
    retriever.generate_embeddings()
    """Semantic search is ready, you can now enter queries (or 'exit' to quit):"""

    while True:
        query = input("Enter your search query: ")
        if query.lower() == 'exit':
            break
        if not query:
            continue

        print(f"Searching for: '{query}'...")
        results = retriever.retrieve_top_k(query, k=3)

        print("\nTop 3 results:")
        for rank, doc, score in results:
            print(f"{rank}. {doc} (Similarity: {score:.4f})")
        print("-" * 70)    
    
   
   
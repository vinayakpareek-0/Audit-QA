import os
import json
import numpy as np
from rank_bm25 import BM25Okapi
import chromadb
from sentence_transformers import SentenceTransformer
from flashrank import Ranker, RerankRequest
import sys
import time
from pathlib import Path

root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

from core.load_config import load_config

class BusinessRetriever:
    """Handles hybrid retrieval using ChromaDB and BM25 with FlashRank reranking."""
    
    def __init__(self):
        self.config = load_config()
        self.embed_model = SentenceTransformer(self.config['embeddings']['model_name'])
        self.chroma_client = chromadb.PersistentClient(path=self.config['paths']['vectorstore_dir'])
        
        col_name = self.config['retrieval'].get('collection_name', 'business_kb')
        self.collection = self.chroma_client.get_collection(name=col_name)
        
        self._initialize_bm25()
        self.ranker = Ranker()

    def _initialize_bm25(self):
        results = self.collection.get()
        self.documents = results['documents']
        self.metadatas = results['metadatas']
        
        if not self.documents:
            self.documents = ["Placeholder"]
        
        tokenized_corpus = [doc.lower().split() for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def get_hybrid_context(self, query: str, top_k: int = 5):
        query_embedding = self.embed_model.encode(query).tolist()
        vector_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k * 2
        )
        
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        top_bm25_indices = np.argsort(bm25_scores)[::-1][:top_k * 2]
        
        candidates = []
        seen_texts = set()
        
        for doc in vector_results['documents'][0]:
            if doc not in seen_texts:
                candidates.append({"text": doc})
                seen_texts.add(doc)
        
        for idx in top_bm25_indices:
            doc = self.documents[idx]
            if doc not in seen_texts:
                candidates.append({"text": doc})
                seen_texts.add(doc)
                
        rerank_request = RerankRequest(query=query, passages=candidates)
        results = self.ranker.rerank(rerank_request)
        
        top_candidates = results[:top_k]
        return "\n\n---\n\n".join([r['text'] for r in top_candidates])

if __name__ == "__main__":
    retriever = BusinessRetriever()
    print(retriever.get_hybrid_context("What are Owlflow services?"))

if __name__ == "__main__":
    # Internal Test
    try:
        retriever = BusinessRetriever()
        q = "What are Owlflow services?"
        print(f"Query: {q}")
        print("-" * 30)
        context = retriever.get_hybrid_context(q)
        print(context)
    except Exception as e:
        print(f"Error during retrieval test: {e}")

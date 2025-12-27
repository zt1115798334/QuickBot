import faiss
import numpy as np
import os
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class VectorIndex:
    def __init__(self):
        self.index_path = os.getenv("FAISS_INDEX_PATH", "./faiss_index.bin")
        self.documents_path = os.getenv("DOCUMENTS_STORAGE_PATH", "./documents.json")
        self.dimension = 512  # Dimension for BAAI/bge-small-zh-v1.5
        self.index = None
        self.documents = []
        
        # Load existing index and documents if they exist
        self.load()
    
    def load(self):
        """Load existing index and documents"""
        # Load documents
        if os.path.exists(self.documents_path):
            with open(self.documents_path, 'r', encoding='utf-8') as f:
                self.documents = json.load(f)
        
        # Load index
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
        else:
            # Create new index if it doesn't exist
            self.index = faiss.IndexFlatIP(self.dimension)
    
    def save(self):
        """Save index and documents"""
        # Save documents
        with open(self.documents_path, 'w', encoding='utf-8') as f:
            json.dump(self.documents, f, ensure_ascii=False, indent=2)
        
        # Save index
        faiss.write_index(self.index, self.index_path)
    
    def add_documents(self, documents_with_embeddings):
        """Add documents with their embeddings to the index"""
        if not documents_with_embeddings:
            return
        
        # Extract embeddings and documents
        embeddings = []
        new_documents = []
        
        for doc_info in documents_with_embeddings:
            if 'embedding' in doc_info and 'text' in doc_info:
                embeddings.append(doc_info['embedding'])
                new_documents.append(doc_info)
        
        if not embeddings:
            return
        
        # Convert to numpy array and add to index
        embeddings_np = np.array(embeddings, dtype=np.float32)
        self.index.add(embeddings_np)
        
        # Add to documents list
        self.documents.extend(new_documents)
        
        # Save changes
        self.save()
    
    def search(self, query_embedding, top_k=5):
        """Search for similar documents based on query embedding"""
        if not query_embedding or self.index.ntotal == 0:
            return []
        
        # Convert query embedding to numpy array
        query_np = np.array([query_embedding], dtype=np.float32)
        
        # Perform search
        distances, indices = self.index.search(query_np, top_k)
        
        # Get results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents):
                result = self.documents[idx].copy()
                result['similarity'] = float(distances[0][i])
                results.append(result)
        
        return results
    
    def get_document_count(self):
        """Get total number of documents in the index"""
        return len(self.documents)
    
    def clear_all(self):
        """Clear all documents and index"""
        self.index = faiss.IndexFlatIP(self.dimension)
        self.documents = []
        self.save()

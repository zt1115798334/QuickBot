from embedding import TextEmbedder
from vector_index_factory import VectorIndexFactory
import uuid

class KnowledgeBase:
    def __init__(self):
        self.embedder = TextEmbedder()
        self.vector_index = VectorIndexFactory.create_index()
    
    def add_document(self, text, metadata=None):
        """Add a single document to the knowledge base"""
        if not text or not isinstance(text, str):
            return None
        
        # Generate unique ID
        doc_id = str(uuid.uuid4())
        
        # Create document object
        document = {
            'id': doc_id,
            'text': text,
            'metadata': metadata or {},
            'timestamp': metadata.get('timestamp') if metadata and 'timestamp' in metadata else None
        }
        
        # Generate embedding
        embedding = self.embedder.embed_text(text)
        if not embedding:
            return None
        
        document['embedding'] = embedding
        
        # Add to vector index
        self.vector_index.add_documents([document])
        
        return doc_id
    
    def add_documents(self, texts, metadata_list=None):
        """Add multiple documents to the knowledge base"""
        if not texts or not isinstance(texts, list):
            return []
        
        metadata_list = metadata_list or [{} for _ in texts]
        if len(metadata_list) != len(texts):
            metadata_list = [{} for _ in texts]  # Use empty metadata if lengths don't match
        
        # Generate embeddings for all texts
        embeddings = self.embedder.embed_texts(texts)
        
        # Create document objects
        documents = []
        doc_ids = []
        
        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            doc_id = str(uuid.uuid4())
            document = {
                'id': doc_id,
                'text': text,
                'metadata': metadata_list[i] or {},
                'embedding': embedding,
                'timestamp': metadata_list[i].get('timestamp') if metadata_list[i] and 'timestamp' in metadata_list[i] else None
            }
            documents.append(document)
            doc_ids.append(doc_id)
        
        # Add to vector index
        self.vector_index.add_documents(documents)
        
        return doc_ids
    
    def delete_document(self, doc_id):
        """Delete a document by ID"""
        # This is a simplified implementation
        # In a real scenario, you would need to rebuild the index after deletion
        # For now, we'll just clear the entire index and re-add all documents except the one to delete
        documents = self.vector_index.documents
        
        # Filter out the document to delete
        remaining_documents = [doc for doc in documents if doc['id'] != doc_id]
        
        if len(remaining_documents) == len(documents):
            return False  # Document not found
        
        # Clear and rebuild the index
        self.vector_index.clear_all()
        if remaining_documents:
            self.vector_index.add_documents(remaining_documents)
        
        return True
    
    def update_document(self, doc_id, new_text, new_metadata=None):
        """Update a document by ID"""
        # Delete the old document
        if not self.delete_document(doc_id):
            return False  # Document not found
        
        # Add the updated document
        new_metadata = new_metadata or {}
        self.add_document(new_text, new_metadata)
        
        return True
    
    def search(self, query, top_k=5):
        """Search for relevant documents based on query"""
        if not query or not isinstance(query, str):
            return []
        
        # Generate embedding for query
        query_embedding = self.embedder.embed_text(query)
        if not query_embedding:
            return []
        
        # Search in vector index
        results = self.vector_index.search(query_embedding, top_k)
        
        return results
    
    def get_all_documents(self):
        """Get all documents in the knowledge base"""
        return self.vector_index.documents.copy()
    
    def get_document_count(self):
        """Get total number of documents"""
        return self.vector_index.get_document_count()
    
    def clear_all(self):
        """Clear all documents from the knowledge base"""
        self.vector_index.clear_all()
        return True

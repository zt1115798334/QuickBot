import os
import json
import numpy as np
from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility
)
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class MilvusIndex:
    def __init__(self):
        self.host = os.getenv("MILVUS_HOST", "localhost")
        self.port = int(os.getenv("MILVUS_PORT", "19530"))
        self.collection_name = os.getenv("MILVUS_COLLECTION", "knowledge_base")
        self.dimension = 512  # Dimension for BAAI/bge-small-zh-v1.5
        self.collection = None
        
        # Connect to Milvus
        self.connect()
        
        # Create or load collection
        self.create_collection()
        
        # Load documents cache
        self.documents = []
        self.load_documents_cache()
    
    def connect(self):
        """Connect to Milvus server"""
        try:
            connections.connect(
                alias="default",
                host=self.host,
                port=self.port
            )
            print(f"Connected to Milvus at {self.host}:{self.port}")
        except Exception as e:
            print(f"Failed to connect to Milvus: {e}")
            raise
    
    def create_collection(self):
        """Create collection if it doesn't exist"""
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dimension),
            FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=10000),  # 增加最大长度以支持更长文本
            FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=1024)
        ]
        
        schema = CollectionSchema(fields, description="Knowledge base collection")
        
        # Create collection if it doesn't exist
        if not utility.has_collection(self.collection_name):
            self.collection = Collection(name=self.collection_name, schema=schema)
            print(f"Created collection: {self.collection_name}")
            
            # Create index
            index_params = {
                "metric_type": "IP",  # Inner Product
                "index_type": "FLAT",  # Flat index for exact search
                "params": {}
            }
            self.collection.create_index(field_name="embedding", index_params=index_params)
            print("Created index on embedding field")
        else:
            self.collection = Collection(name=self.collection_name)
            print(f"Loaded existing collection: {self.collection_name}")
        
        # Load collection into memory
        self.collection.load()
    
    def load_documents_cache(self):
        """Load documents from Milvus to cache"""
        self.documents = []
        
        if self.collection is None:
            return
        
        # Get all entities from collection
        try:
            res = self.collection.query(
                expr="",
                output_fields=["id", "doc_id", "text", "metadata"],
                limit=16384  # 添加limit参数防止空表达式错误，设置为Milvus允许的最大值
            )
            
            for entity in res:
                doc = {
                    "id": entity["doc_id"],
                    "text": entity["text"],
                    "metadata": json.loads(entity["metadata"])
                }
                self.documents.append(doc)
                
            print(f"Loaded {len(self.documents)} documents from Milvus")
        except Exception as e:
            print(f"Failed to load documents: {e}")
    
    def add_documents(self, documents_with_embeddings):
        """Add documents with their embeddings to the index"""
        if not documents_with_embeddings:
            return
        
        if self.collection is None:
            raise Exception("Milvus collection not initialized")
        
        # Prepare data for insertion
        embeddings = []
        doc_ids = []
        texts = []
        metadatas = []
        
        new_documents = []
        
        for doc_info in documents_with_embeddings:
            if 'embedding' in doc_info and 'text' in doc_info and 'id' in doc_info:
                embeddings.append(doc_info['embedding'])
                doc_ids.append(doc_info['id'])
                texts.append(doc_info['text'])
                metadatas.append(json.dumps(doc_info['metadata'] if 'metadata' in doc_info else {}))
                new_documents.append(doc_info)
        
        if not embeddings:
            return
        
        # Convert to numpy array
        embeddings_np = np.array(embeddings, dtype=np.float32)
        
        # Insert data into Milvus
        entities = [
            embeddings_np,
            doc_ids,
            texts,
            metadatas
        ]
        
        insert_result = self.collection.insert(entities)
        
        # Flush to ensure data is written
        self.collection.flush()
        
        # Update documents cache
        self.documents.extend(new_documents)
        
        print(f"Added {len(new_documents)} documents to Milvus")
    
    def search(self, query_embedding, top_k=5):
        """Search for similar documents based on query embedding"""
        if not query_embedding:
            return []
        
        if self.collection is None:
            return []
        
        # Convert query embedding to numpy array
        query_np = np.array([query_embedding], dtype=np.float32)
        
        # Search parameters
        search_params = {
            "metric_type": "IP",
            "params": {}
        }
        
        # Perform search
        results = self.collection.search(
            data=query_np,
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["doc_id", "text", "metadata"]
        )
        
        # Process results
        search_results = []
        for hits in results:
            for hit in hits:
                doc = {
                    "id": hit.entity.get("doc_id"),
                    "text": hit.entity.get("text"),
                    "metadata": json.loads(hit.entity.get("metadata")),
                    "similarity": hit.score
                }
                search_results.append(doc)
        
        return search_results
    
    def get_document_count(self):
        """Get total number of documents in the index"""
        if self.collection is None:
            return 0
        return self.collection.num_entities
    
    def clear_all(self):
        """Clear all documents and index"""
        if self.collection is None:
            return
        
        # Drop and recreate collection
        self.collection.drop()
        print(f"Dropped collection: {self.collection_name}")
        
        # Recreate collection
        self.create_collection()
        
        # Clear documents cache
        self.documents = []
        print("Cleared all documents")

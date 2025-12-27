#!/usr/bin/env python3

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class VectorIndexFactory:
    @staticmethod
    def create_index():
        """Create vector index based on configuration"""
        vector_db_type = os.getenv("VECTOR_DB_TYPE", "faiss").lower()
        print(f"Vector DB Type: {vector_db_type}")
        if vector_db_type == "milvus":
            try:
                from milvus_index import MilvusIndex
                return MilvusIndex()
            except ImportError:
                print("Warning: pymilvus not available, falling back to FAISS")
            except Exception as e:
                print(f"Warning: Failed to initialize Milvus index: {e}, falling back to FAISS")
        
        # Default to FAISS (also used as fallback if Milvus fails)
        from faiss_index import VectorIndex
        return VectorIndex()

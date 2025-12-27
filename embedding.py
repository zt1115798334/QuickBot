from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class TextEmbedder:
    def __init__(self):
        self.model_name = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-zh-v1.5")
        self.local_model_path = os.getenv("EMBEDDING_MODEL_LOCAL_PATH", "./models/bge-small-zh-v1.5")

        try:
            # 如果网上加载失败，尝试从本地路径加载
            self.model = SentenceTransformer(self.local_model_path)
            print(f"✓ 成功从本地路径加载模型: {self.local_model_path}")
        except Exception as local_e:
            print(f"✗ 从本地路径加载模型也失败: {local_e}")
            try:
                # 首先尝试从网上加载模型
                self.model = SentenceTransformer(self.model_name)
                print(f"✓ 成功从网上加载模型: {self.model_name}")
            except Exception as e:
                print(f"⚠ 从网上加载模型失败: {e}")

            raise Exception(f"无法加载模型，无论是从网上还是本地路径: {local_e}")


        
    def embed_text(self, text):
        """Embed a single text string into a vector"""
        if not text or not isinstance(text, str):
            return None
        
        # Ensure text is not too long (truncate if needed)
        if len(text) > 512:
            text = text[:512]
            
        return self.model.encode(text, normalize_embeddings=True).tolist()
    
    def embed_texts(self, texts):
        """Embed multiple text strings into vectors"""
        if not texts or not isinstance(texts, list):
            return []
            
        # Filter out empty texts and ensure all are strings
        valid_texts = [t[:512] if len(t) > 512 else t for t in texts if t and isinstance(t, str)]
        
        if not valid_texts:
            return []
            
        embeddings = self.model.encode(valid_texts, normalize_embeddings=True)
        return embeddings.tolist()

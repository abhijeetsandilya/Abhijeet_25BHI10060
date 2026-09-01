from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL


class EmbeddingModel:
    def __init__(self):
    
        self.model = SentenceTransformer(EMBEDDING_MODEL)

    def embed_documents(self, texts):
        
        return self.model.encode(
            texts,
            convert_to_numpy=True
        )

    def embed_query(self, query):
        
        return self.model.encode(
            query,
            convert_to_numpy=True
        )
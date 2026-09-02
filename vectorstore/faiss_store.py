import os
import faiss
import numpy as np


class FAISSStore:

    def __init__(self, embedding_model):

        self.embedding_model = embedding_model
        self.index = None
        self.documents = []

    def build(self, documents):

        if not documents:
            raise ValueError("No documents provided to build the FAISS index.")

        texts = [document.page_content for document in documents]

        embeddings = self.embedding_model.encode(
            texts,
            convert_to_numpy=True
        )

        embeddings = np.asarray(embeddings, dtype="float32")

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)

        self.documents = documents

    def search(self, query, k=3):

        if self.index is None:
            raise ValueError("FAISS index has not been built yet.")

        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True
        )

        query_embedding = np.asarray(
            query_embedding,
            dtype="float32"
        )

        distances, indices = self.index.search(
            query_embedding,
            k
        )

        results = []

        for index in indices[0]:
            if index != -1:
                results.append(self.documents[index])

        return results

    def save(self, path="faiss_index"):
 
        if self.index is None:
            raise ValueError("FAISS index has not been built yet.")

        os.makedirs(path, exist_ok=True)

        faiss.write_index(
            self.index,
            os.path.join(path, "index.faiss")
        )

        np.save(
            os.path.join(path, "documents.npy"),
            np.array(self.documents, dtype=object),
            allow_pickle=True
        )

    def load(self, path="faiss_index"):
        
        index_path = os.path.join(path, "index.faiss")
        documents_path = os.path.join(path, "documents.npy")

        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"FAISS index not found at {index_path}"
            )

        self.index = faiss.read_index(index_path)

        self.documents = np.load(
            documents_path,
            allow_pickle=True
        ).tolist()
class Retriever:

    def __init__(self, vector_store, top_k=3):

        self.vector_store = vector_store
        self.top_k = top_k

    def retrieve(self, query):
        
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        return self.vector_store.search(
            query,
            k=self.top_k
        )
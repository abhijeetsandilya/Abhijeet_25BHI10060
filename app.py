import os

from loaders.document_loader import load_document
from processing.chunker import chunk_documents
from embeddings.embedding_model import EmbeddingModel
from vectorstore.faiss_store import FAISSStore
from retrieval.retriever import Retriever
from generation.generator import Generator


DATA_FILE = "data/sample_documents/sample.pdf"
FAISS_DIR = "data/faiss_index"


def build_index():

    print("Loading documents...")
    documents = load_document(DATA_FILE)

    if not documents:
        raise ValueError("No documents found.")

    print(f"Loaded {len(documents)} characters from the document.")

    print("Chunking documents...")
    chunks = chunk_documents(documents)

    print(f"Created {len(chunks)} chunks.")

    print("Loading embedding model...")
    embedding_model = EmbeddingModel()

    print("Building FAISS index...")
    vector_store = FAISSStore(embedding_model)
    vector_store.build(chunks)

    os.makedirs(FAISS_DIR, exist_ok=True)
    vector_store.save(FAISS_DIR)

    print("FAISS index created successfully.")

    return embedding_model, vector_store


def main():

    embedding_model, vector_store = build_index()

    retriever = Retriever(vector_store, top_k=3)
    generator = Generator()

    print("\nDocSeek is ready!")
    print("Ask questions about your documents.")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("You: ").strip()

        if query.lower() == "exit":
            print("Goodbye!")
            break

        if not query:
            continue

        try:
            documents = retriever.retrieve(query)

            answer = generator.generate(
                query,
                documents
            )

            print(f"\nDocSeek: {answer}\n")

        except Exception as error:
            print(f"\nError: {error}\n")


if __name__ == "__main__":
    main()
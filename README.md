# DocSeek

DocSeek is a document question-answering system built around a
Retrieval-Augmented Generation (RAG) pipeline.

Instead of asking an LLM to answer questions from its general knowledge,
DocSeek first searches the user's documents for relevant information and
then provides that information to an LLM as context. This helps keep
answers grounded in the supplied documents.

## Architecture

``` text
                    User Question
                          |
                          v
                   +--------------+
                   |   Retriever  |
                   +--------------+
                          |
                          v
                   +--------------+
                   |    FAISS     |
                   | Vector Store |
                   +--------------+
                          ^
                          |
                    Query Embedding
                          |
                          v
                   Embedding Model
                          ^
                          |
                    Document Chunks
                          ^
                          |
                   +--------------+
                   |    Chunker   |
                   +--------------+
                          ^
                          |
                   +--------------+
                   |    Loader    |
                   +--------------+
                          ^
                          |
                    PDF / TXT / DOCX

Retrieved Chunks + User Question
                |
                v
          +-------------+
          |    Groq     |
          |     LLM     |
          +-------------+
                |
                v
             Answer
```

## Pipeline

1.  **Document Loading**\
    `loaders/document_loader.py` extracts text from PDF, TXT, or DOCX
    files.

2.  **Chunking**\
    `processing/chunker.py` splits the extracted text into smaller
    overlapping pieces.

3.  **Embedding**\
    `embeddings/embedding_model.py` converts each chunk into a numerical
    vector using a Sentence Transformer.

4.  **Vector Storage**\
    `vectorstore/faiss_store.py` stores the vectors in a FAISS index and
    performs similarity search.

5.  **Retrieval**\
    `retrieval/retriever.py` retrieves the most relevant chunks for a
    user's question.

6.  **Generation**\
    `generation/generator.py` sends the question and retrieved context
    to the Groq-hosted LLM.

7.  **Application**\
    `app.py` connects all components and provides the command-line
    interface.

------------------------------------------------------------------------

# Why This Architecture?

The project is divided into separate modules so that each component has
one main responsibility.

``` text
loaders/       Document input
processing/    Text processing
embeddings/    Text -> vectors
vectorstore/   Vector storage and similarity search
retrieval/     Retrieval logic
generation/    LLM generation
app.py         Application orchestration
```

This separation makes the system easier to understand, debug, and
replace.

For example, the embedding model can be changed without rewriting the
document loader, and the LLM can be changed without changing the FAISS
interface.

# Design Justifications

## 1. Why a chunk size of 1000?

DocSeek currently uses:

``` python
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
```

The unit is approximately **characters**, because
`RecursiveCharacterTextSplitter` is being used with character-based
chunk sizing.

A document that is too large per chunk can contain multiple unrelated
topics. That makes retrieval less precise because a retrieved chunk may
contain a lot of irrelevant material.

A document that is too small can lose the surrounding context needed to
understand an idea.

A size of **1000 characters** is a practical starting point that
provides enough context for many passages while keeping chunks
reasonably focused.

This value is not mathematically optimal for every document. It is a
tunable engineering parameter.

### Why 200 characters of overlap?

Information near a chunk boundary can otherwise be separated from the
context that explains it.

A **200-character overlap** gives neighboring chunks shared context
while avoiding excessive duplication.

The 1000/200 configuration is therefore a reasonable starting point, not
a universal rule.

------------------------------------------------------------------------

## 2. Why RecursiveCharacterTextSplitter?

DocSeek uses LangChain's `RecursiveCharacterTextSplitter`.

Instead of immediately cutting text at arbitrary character positions, it
attempts to preserve larger textual structures first.

The configured separators are:

``` python
[
    "\n\n",
    "\n",
    ". ",
    " ",
    ""
]
```

This gives the splitter a preference roughly equivalent to:

``` text
Paragraph
   ↓
Line
   ↓
Sentence
   ↓
Word
   ↓
Character
```

The goal is to produce chunks that are more semantically coherent than
simply taking every N characters.

------------------------------------------------------------------------

## 3. Why `sentence-transformers/all-MiniLM-L6-v2`?

DocSeek uses:

``` text
sentence-transformers/all-MiniLM-L6-v2
```

This model converts text into dense numerical vectors called
**embeddings**.

The important property for DocSeek is that semantically similar pieces
of text should have similar vector representations.

For example:

``` text
"What is biophysics?"
```

and

``` text
"Biophysics studies biological systems using principles of physics."
```

can be represented in nearby regions of vector space even though they do
not contain exactly the same words.

### Why this model?

`all-MiniLM-L6-v2` was chosen because it provides a good practical
balance for a small project:

-   It is relatively lightweight.
-   It is designed for sentence and short-text embeddings.
-   It can run locally instead of requiring an embedding API call for
    every chunk.
-   It is fast enough to be practical on modest hardware.
-   It is widely used and straightforward to integrate with Sentence
    Transformers.

For DocSeek, this is a practical engineering choice rather than a claim
that it is the best embedding model for every dataset.

------------------------------------------------------------------------

## 4. What is a Sentence Transformer?

A Transformer is a neural-network architecture designed to process
sequences of text using attention mechanisms.

A **Sentence Transformer** adapts Transformer models so that complete
sentences or pieces of text can be converted into useful fixed-size
vector representations.

DocSeek uses those representations for semantic search.

``` text
Text
 |
 v
Sentence Transformer
 |
 v
Embedding vector
 |
 v
FAISS
```

The embedding model is not generating the answer. It helps DocSeek
determine which pieces of the document are relevant to the question.

------------------------------------------------------------------------

## 5. Why FAISS?

DocSeek uses **FAISS (Facebook AI Similarity Search)** as its vector
store.

Once text has been converted into embeddings, ordinary keyword search is
not enough. We want to find vectors that are mathematically close to the
vector representing the user's question.

FAISS is designed specifically for efficient similarity search over
vectors.

``` text
Document chunk
     |
     v
Embedding
     |
     v
FAISS index
```

When the user asks a question:

``` text
Question
   |
   v
Question embedding
   |
   v
FAISS similarity search
   |
   v
Top-k relevant chunks
```

### Why FAISS for this project?

FAISS is a good fit because:

-   It is open source.
-   It runs locally.
-   It is straightforward to integrate.
-   It is efficient for vector similarity search.
-   It avoids requiring a separate hosted vector-database service.
-   It is more than sufficient for the relatively small document
    collection used by this project.

For a much larger production system, other vector databases or
specialized FAISS indexes could be considered.

------------------------------------------------------------------------

## 6. Why `IndexFlatL2`?

DocSeek currently uses:

``` python
faiss.IndexFlatL2(dimension)
```

because it is simple and exact.

It compares the query vector against the stored vectors using L2
distance.

The advantage is simplicity and predictable behavior.

The trade-off is that it performs a brute-force comparison rather than
using an approximate nearest-neighbor index. For the scale of this
project, simplicity is more valuable than premature optimization.

If the number of vectors became very large, a different FAISS index
could be introduced.

------------------------------------------------------------------------

## 7. Why Groq for Generation?

DocSeek uses the Groq API to provide the LLM used for final answer
generation.

The LLM receives:

``` text
Retrieved Context
+
User Question
```

and generates the final answer.

The project keeps the generation layer separate from retrieval so that
the LLM provider or model can be changed without redesigning the rest of
the RAG pipeline.

------------------------------------------------------------------------

## 8. Why retrieve only the top 3 chunks?

The retriever currently uses:

``` python
top_k = 3
```

This means DocSeek sends the three most relevant chunks to the
generator.

A larger value provides more context but can introduce more irrelevant
information and increases the amount of text sent to the LLM.

A smaller value makes the context more focused but risks excluding
useful information.

Three chunks is therefore a practical starting point for this project
and can be tuned depending on retrieval quality.

------------------------------------------------------------------------

# Project Structure

``` text
DocSeek/
│
├── app.py
├── config.py
├── requirements.txt
├── README.md
├── .env
├── .gitignore
│
├── loaders/
│   ├── __init__.py
│   └── document_loader.py
│
├── processing/
│   ├── __init__.py
│   └── chunker.py
│
├── embeddings/
│   ├── __init__.py
│   └── embedding_model.py
│
├── vectorstore/
│   ├── __init__.py
│   └── faiss_store.py
│
├── retrieval/
│   ├── __init__.py
│   └── retriever.py
│
├── generation/
│   ├── __init__.py
│   └── generator.py
│
├── utils/
│   ├── __init__.py
│   └── helpers.py
│
└── data/
    └── sample_documents/
```

# Installation

Create and activate a virtual environment:

``` powershell
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

``` powershell
pip install -r requirements.txt
```

# Environment Variables

Create a `.env` file in the project root:

``` text
GROQ_API_KEY=your_api_key_here
```

Never commit `.env` or expose the API key publicly.

# Running DocSeek

Place a supported document inside:

``` text
data/sample_documents/
```

Set the file path in `app.py`:

``` python
DATA_FILE = "data/sample_documents/sample.pdf"
```

Then run:

``` powershell
python app.py
```

DocSeek will:

1.  Load the document.
2.  Split it into chunks.
3.  Generate embeddings.
4.  Build a FAISS index.
5.  Accept questions from the terminal.
6.  Retrieve relevant chunks.
7.  Generate an answer using the LLM.

Example:

``` text
DocSeek is ready!
Ask questions about your documents.
Type 'exit' to quit.

You: What is biophysics?

DocSeek: ...
```

If relevant information cannot be found in the retrieved context,
DocSeek is instructed not to invent an answer.

# Technology Stack

-   **Python** - Core programming language
-   **PyPDF** - PDF text extraction
-   **python-docx** - DOCX text extraction
-   **LangChain Text Splitters** - Document chunking
-   **Sentence Transformers** - Local text embeddings
-   **FAISS** - Vector similarity search
-   **Groq API** - LLM inference
-   **NumPy** - Numerical array handling
-   **python-dotenv** - Environment variable management

# Limitations

DocSeek is a small educational RAG implementation, so it intentionally
has limitations:

-   The current document loader processes one file at a time.
-   PDF extraction quality depends on the PDF's text layer.
-   The current FAISS setup is designed for relatively small datasets.
-   Retrieval quality depends heavily on chunking and embedding quality.
-   The system does not currently provide source citations in generated
    answers.
-   The LLM can still make mistakes, even when supplied with retrieved
    context.
-   Chunk size, overlap, and `top_k` are fixed configuration values
    rather than dynamically optimized.

# Future Improvements

Possible extensions include:

-   Support for multiple documents at once.
-   Metadata-aware retrieval.
-   Source/page citations in answers.
-   Better chunking strategies for different document types.
-   Hybrid keyword + semantic search.
-   Reranking retrieved chunks.
-   Persistent document metadata.
-   A web interface.
-   Conversation history.
-   Evaluation of retrieval accuracy.
-   Configurable embedding and LLM models.
-   More scalable vector indexes.

# Learning Objective

DocSeek demonstrates the core concepts behind a Retrieval-Augmented
Generation system:

``` text
Documents
   ↓
Text Extraction
   ↓
Chunking
   ↓
Embeddings
   ↓
Vector Database
   ↓
Semantic Retrieval
   ↓
Context Injection
   ↓
LLM Generation
```

The project focuses on understanding how these components interact
rather than hiding the entire pipeline behind a single framework call.

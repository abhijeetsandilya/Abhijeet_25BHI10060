import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

LLM_MODEL = "openai/gpt-oss-20b"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

if not GROQ_API_KEY:

    raise ValueError(
    "GROQ_API_KEY not found. Please add it to your .env file."
    )

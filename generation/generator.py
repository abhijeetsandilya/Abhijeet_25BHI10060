from groq import Groq

from config import GROQ_API_KEY, LLM_MODEL


class Generator:
    

    def __init__(self):
        
        self.client = Groq(api_key=GROQ_API_KEY)

    def generate(self, query, documents):
        
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        if not documents:
            return "I could not find relevant information in the provided documents."

        context = "\n\n".join(
            document.page_content
            for document in documents
        )

        prompt = f"""
You are DocSeek, a document question-answering assistant.

Answer the user's question using only the information provided
in the context below.

If the answer cannot be found in the context, say that you
could not find the answer in the provided documents.

Do not make up information.

Context:
{context}

Question:
{query}

Answer:
"""

        response = self.client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You answer questions using provided document context."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0
        )

        return response.choices[0].message.content
"""
generation.py
-------------
Answer generation using local Ollama model.
Maintains conversation history for multi-turn dialogue.
"""

import requests


class DocumentAgent:
    """
    Conversational agent that answers questions from documents.
    Uses RAG: retrieve relevant chunks, then generate grounded answer.
    """

    def __init__(self, vector_store, model: str = 'gemma3:12b',
                 ollama_url: str = 'http://127.0.0.1:11434'):
        self.vector_store = vector_store
        self.model = model
        self.ollama_url = ollama_url
        self.history = []

    def _build_prompt(self, query: str, context: list) -> str:
        """Build RAG prompt with retrieved context."""
        context_text = '\n\n'.join(context)
        return f"""You are a helpful assistant that answers questions based on the provided document.
Answer only from the context below. If the answer is not in the context, say so clearly.

Context:
{context_text}

Question: {query}
Answer:"""

    def ask(self, query: str, top_k: int = 3) -> dict:
        """
        Ask a question about the loaded document.

        Args:
            query: User question
            top_k: Number of chunks to retrieve

        Returns:
            Dict with answer and retrieved context
        """
        # Retrieve relevant chunks
        context = self.vector_store.retrieve(query, top_k=top_k)

        # Build prompt
        prompt = self._build_prompt(query, context)

        # Add to history
        self.history.append({'role': 'user', 'content': prompt})

        # Generate answer
        response = requests.post(
            f'{self.ollama_url}/api/chat',
            json={
                'model': self.model,
                'messages': self.history,
                'stream': False
            },
            proxies={'http': None, 'https': None}
        )

        answer = response.json()['message']['content']

        # Store assistant response in history
        self.history.append({'role': 'assistant', 'content': answer})

        return {
            'query': query,
            'answer': answer,
            'context': context,
            'history_length': len(self.history)
        }

    def reset(self) -> None:
        """Clear conversation history."""
        self.history = []
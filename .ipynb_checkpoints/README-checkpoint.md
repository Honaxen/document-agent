# Document Agent

An AI agent that answers questions about documents using Claude API and RAG.
Upload a document, ask anything — the agent retrieves and reasons.

---

## What It Does

- Ingests PDF and text documents
- Chunks and embeds content into a vector store
- Retrieves relevant chunks for any query
- Uses Claude API to generate grounded answers
- Maintains conversation context across multiple questions

---

## Architecture

```
Document
   |
   v
Ingestion (chunk + embed)
   |
   v
Vector Store (FAISS)
   |
   v
Query -> Retrieve -> Claude API -> Answer
```

---

## Project Structure

```
document-agent/
├── agent/
│   ├── main.py
│   ├── ingestion.py
│   ├── retrieval.py
│   └── generation.py
├── api/
│   └── main.py
├── tests/
├── data/
├── requirements.txt
├── Dockerfile
└── README.md
```
---

## Getting Started

```bash
pip install -r requirements.txt
```

Run CLI:
```bash
cd agent
python3 main.py --document ../data/your_document.pdf
```

Run API:
```bash
uvicorn api.main:app --reload
```

Run tests:
```bash
pytest tests/test_agent.py -v
```
---

## Stack

Python · Ollama · FAISS · sentence-transformers · FastAPI

---

## What I Learned

Building an agent is different from building a model.
A model predicts. An agent reasons, retrieves, and responds.

Conversation history changes everything.
Without it, every question is isolated.
With it, the agent can answer follow-up questions coherently.

Local models (Ollama) are viable for development and prototyping.
For production, a hosted API gives better reliability and speed.

---

## Author

[Honaxen](https://github.com/Honaxen)
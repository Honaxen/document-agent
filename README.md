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

## Stack

Python · Claude API · FAISS · sentence-transformers · FastAPI

---

## What I Learned

TBD — will be updated after completion.

---

## Author

[Honaxen](https://github.com/Honaxen)
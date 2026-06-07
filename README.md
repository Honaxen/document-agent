# Document Agent

A conversational AI agent that answers questions from any document.
Upload a PDF or text file — ask anything — the agent retrieves and reasons.

---

## What It Does

- Ingests PDF and text documents
- Chunks and embeds content into a vector store
- Retrieves relevant chunks for any query
- Generates grounded answers using a local LLM (Ollama)
- Maintains conversation context across multiple questions
- Web UI via Gradio — no coding required

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
Query -> Retrieve -> Ollama (gemma3:12b) -> Answer
```

---

## Project Structure

```
document-agent/
├── agent/
│   ├── main.py          — CLI interface
│   ├── ingestion.py     — PDF/TXT loading and chunking
│   ├── retrieval.py     — FAISS vector store
│   └── generation.py    — conversational agent with history
├── api/
│   └── main.py          — FastAPI REST API
├── tests/
│   └── test_agent.py    — 6/6 passing
├── app.py               — Gradio web UI
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

Make sure Ollama is running:
```bash
ollama serve
```

### Run Web UI (Gradio)
```bash
python3 app.py
```
Then open http://localhost:7860

### Run CLI
```bash
cd agent
python3 main.py --document ../data/your_document.pdf
```

### Run API
```bash
uvicorn api.main:app --reload
```

### Run Tests
```bash
pytest tests/test_agent.py -v
```

---

## Stack

Python · Ollama · FAISS · sentence-transformers · FastAPI · Gradio

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
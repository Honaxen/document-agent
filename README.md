# Document Agent

A conversational AI agent that answers questions from any document.
Upload a PDF or text file — ask anything — the agent retrieves and reasons.

---

## What It Does

- Ingests PDF and text documents
- Chunks and embeds content into a vector store
- **Hybrid retrieval: FAISS (dense) + BM25 (sparse) + RRF fusion**
- Generates grounded answers using a local LLM (Ollama)
- Maintains conversation context across multiple questions
- Web UI via Gradio — no coding required
- **Built-in evaluation via rag-evaluation-framework**

---

## Architecture

```
Document
   |
   v
Ingestion (chunk + clean)
   |
   v
Hybrid Index (FAISS + BM25)
   |         |
dense      sparse
   |         |
   └────┬────┘
        |
   RRF Fusion
        |
        v
Query -> Retrieve -> Ollama (gemma3:12b) -> Answer
```

**Why hybrid?**
FAISS handles semantic similarity ("what does X mean?").
BM25 handles exact keyword matches ("what is the value of X?").
Combining both with Reciprocal Rank Fusion covers cases neither handles alone.

---

## Project Structure

```
document-agent/
├── agent/
│   ├── main.py          — CLI interface
│   ├── ingestion.py     — PDF/TXT loading and chunking
│   ├── retrieval.py     — Hybrid: FAISS + BM25 + RRF
│   └── generation.py    — conversational agent with history
├── api/
│   └── main.py          — FastAPI REST API
├── tests/
│   └── test_agent.py    — 6/6 passing
├── app.py               — Gradio web UI
├── evaluate.py          — RAG evaluation script
├── data/
│   └── sample_questions.json
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Getting Started

```bash
pip install -r requirements.txt
ollama serve
ollama pull gemma3:12b
```

### Run Web UI (Gradio)
```bash
python3 app.py
```
Open: http://localhost:7860

### Run CLI
```bash
cd agent
python3 main.py --document ../data/your_document.pdf
```

### Run API
```bash
uvicorn api.main:app --reload
```

### Run Evaluation
```bash
python3 evaluate.py --doc data/your_document.pdf --questions data/sample_questions.json
```

Output:
```
────────────────────────────────────────────
  Metric             Score  Bar
────────────────────────────────────────────
  Faithfulness        1.00  ████████████████████
  Relevance           0.30  █████░░░░░░░░░░░░░░░
  Completeness        1.00  ████████████████████
  Precision           0.17  ███░░░░░░░░░░░░░░░░░
────────────────────────────────────────────
  Overall             0.62
```

### Run Tests
```bash
pytest tests/test_agent.py -v
```

---

## Stack

Python · Ollama · FAISS · BM25 · sentence-transformers · FastAPI · Gradio

---

## What I Learned

Building an agent is different from building a model.
A model predicts. An agent reasons, retrieves, and responds.

Hybrid search beats pure semantic search.
FAISS alone misses exact keyword matches. BM25 alone misses semantic relationships.
Reciprocal Rank Fusion combines both rankings without needing to normalize scores.

Evaluation reveals what intuition hides.
High faithfulness (1.00) means no hallucination.
Low precision (0.17) means retrieved chunks contain a lot of noise.
You can't see either of these without measuring them.

---

## Related Projects

- [rag-evaluation-framework](https://github.com/Honaxen/rag-evaluation-framework) — the evaluation tool used here
- [rag-system-from-scratch](https://github.com/Honaxen/rag-system-from-scratch) — RAG pipeline from zero

---

## Author

[Honaxen](https://github.com/Honaxen)
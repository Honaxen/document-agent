"""
api/main.py
-----------
FastAPI wrapper for the Document Agent.

Run with:
    uvicorn api.main:app --reload
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import tempfile
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.ingestion import ingest
from agent.retrieval import VectorStore
from agent.generation import DocumentAgent

app = FastAPI(
    title="Document Agent API",
    description="Upload a document and ask questions about it.",
    version="1.0.0"
)

# Global state
store = VectorStore()
agent = None


@app.get("/")
async def root():
    return {
        "name": "Document Agent API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "document_loaded": agent is not None,
        "chunks": store.size
    }


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a PDF or TXT document.
    Ingests, chunks, and indexes the document for querying.
    """
    global store, agent

    if not file.filename.endswith(('.pdf', '.txt')):
        raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported.")

    # Save to temp file
    suffix = '.pdf' if file.filename.endswith('.pdf') else '.txt'
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        chunks = ingest(tmp_path)
        store = VectorStore()
        store.build(chunks)
        agent = DocumentAgent(vector_store=store)
    finally:
        os.unlink(tmp_path)

    return {
        "message": "Document loaded successfully",
        "filename": file.filename,
        "chunks": store.size
    }


class QueryRequest(BaseModel):
    question: str
    top_k: int = 3


@app.post("/ask")
async def ask(request: QueryRequest):
    """
    Ask a question about the uploaded document.
    """
    if agent is None:
        raise HTTPException(status_code=400, detail="No document loaded. Upload a document first.")

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    result = agent.ask(request.question, top_k=request.top_k)
    return {
        "question": result["query"],
        "answer": result["answer"],
        "context_chunks": len(result["context"])
    }


@app.post("/reset")
async def reset():
    """Reset conversation history."""
    if agent is None:
        raise HTTPException(status_code=400, detail="No document loaded.")
    agent.reset()
    return {"message": "Conversation history cleared."}
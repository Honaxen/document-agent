"""
test_agent.py
-------------
Tests for Document Agent API.
"""

from fastapi.testclient import TestClient
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.main import app

client = TestClient(app)


def test_root():
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["name"] == "Document Agent API"


def test_health_no_document():
    """Test health endpoint before document upload."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["document_loaded"] == False


def test_ask_without_document():
    """Test that asking without a document returns 400."""
    response = client.post(
        "/ask",
        json={"question": "What is this about?", "top_k": 3}
    )
    assert response.status_code == 400


def test_empty_question():
    """Test that empty question returns 400."""
    response = client.post(
        "/ask",
        json={"question": "", "top_k": 3}
    )
    assert response.status_code == 400


def test_upload_txt_document():
    """Test uploading a text document."""
    # Create a simple test document
    test_content = b"Machine learning is a subset of AI. It enables computers to learn from data."
    
    response = client.post(
        "/upload",
        files={"file": ("test.txt", test_content, "text/plain")}
    )
    assert response.status_code == 200
    assert response.json()["message"] == "Document loaded successfully"
    assert response.json()["chunks"] > 0


def test_health_after_upload():
    """Test health endpoint after document upload."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["document_loaded"] == True
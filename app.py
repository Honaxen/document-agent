"""
app.py
------
Gradio demo for Document Agent.
Upload a document and ask questions about it.
Compatible with Gradio 6.x
"""

import gradio as gr
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent.ingestion import ingest
from agent.retrieval import VectorStore
from agent.generation import DocumentAgent

# Global state
store = VectorStore()
agent = None


def load_document(file):
    """Load and index an uploaded document."""
    global store, agent

    if file is None:
        return "Please upload a document first."

    try:
        chunks = ingest(file)
        store = VectorStore()
        store.build(chunks)
        agent = DocumentAgent(vector_store=store)
        return f"Document loaded successfully. {store.size} chunks indexed."
    except Exception as e:
        return f"Error loading document: {str(e)}"


def ask_question(question, history):
    if not question.strip():
        return history, ""

    if agent is None:
        history = history + [{"role": "user", "content": question}, 
                             {"role": "assistant", "content": "Please upload a document first."}]
        return history, ""

    result = agent.ask(question)
    history = history + [{"role": "user", "content": question},
                         {"role": "assistant", "content": result['answer']}]
    return history, ""


def reset_conversation():
    global agent
    if agent:
        agent.reset()
    return []


# Build Gradio UI
with gr.Blocks(title="Document Agent") as demo:
    gr.Markdown("# Document Agent")
    gr.Markdown("Upload a PDF or text file and ask questions about it.")

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(
                label="Upload Document (PDF or TXT)",
                file_types=[".pdf", ".txt"]
            )
            load_btn = gr.Button("Load Document", variant="primary")
            load_status = gr.Textbox(label="Status", interactive=False)
            reset_btn = gr.Button("Reset Conversation")

        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Conversation", height=400, type="messages")
            question_input = gr.Textbox(
                label="Ask a question",
                placeholder="What is this document about?",
            )
            ask_btn = gr.Button("Ask", variant="primary")

    load_btn.click(
        fn=load_document,
        inputs=[file_input],
        outputs=[load_status]
    )

    ask_btn.click(
        fn=ask_question,
        inputs=[question_input, chatbot],
        outputs=[chatbot, question_input]
    )

    question_input.submit(
        fn=ask_question,
        inputs=[question_input, chatbot],
        outputs=[chatbot, question_input]
    )

    reset_btn.click(
        fn=reset_conversation,
        outputs=[chatbot]
    )

if __name__ == "__main__":
    demo.launch()
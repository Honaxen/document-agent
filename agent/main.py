import sys
sys.stdout.reconfigure(line_buffering=True)

from ingestion import ingest
from retrieval import VectorStore
from generation import DocumentAgent
import argparse

def main():
    parser = argparse.ArgumentParser(description='Document Agent')
    parser.add_argument('--document', required=True)
    parser.add_argument('--top_k', type=int, default=3)
    parser.add_argument('--model', default='gemma3:12b')
    args = parser.parse_args()

    print(f"Loading document: {args.document}", flush=True)
    chunks = ingest(args.document)
    print(f"Chunks: {len(chunks)}", flush=True)

    print("Building vector store...", flush=True)
    store = VectorStore()
    store.build(chunks)
    print(f"Ready. {store.size} chunks indexed.", flush=True)

    agent = DocumentAgent(vector_store=store, model=args.model)

    print("\n" + "="*50, flush=True)
    print("Document Agent Ready", flush=True)
    print("Type 'exit' to quit, 'reset' to clear history", flush=True)
    print("="*50 + "\n", flush=True)

    while True:
        query = input("You: ").strip()
        if not query:
            continue
        if query.lower() == 'exit':
            print("Goodbye!")
            break
        if query.lower() == 'reset':
            agent.reset()
            print("Conversation history cleared.\n")
            continue
        result = agent.ask(query, top_k=args.top_k)
        print(f"\nAgent: {result['answer']}\n", flush=True)

if __name__ == '__main__':
    main()
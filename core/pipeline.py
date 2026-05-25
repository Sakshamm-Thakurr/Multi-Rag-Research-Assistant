from core.document_loader import load_and_chunk_pdf
from core.embeddings import build_faiss_index, load_faiss_index
from agents.retriever import retriever_agent
from agents.summariser import summariser_agent
from agents.synthesiser import synthesiser_agent
import os

def run_pipeline(pdf_path: str, query: str) -> dict:
    """
    Full RAG pipeline — connects all 3 agents.
    1. Load & chunk PDF
    2. Build/load FAISS index
    3. Retriever finds relevant chunks
    4. Summariser compresses context
    5. Synthesiser generates final answer
    """
    print(f"\n{'='*50}")
    print(f"📚 RAG Pipeline Starting")
    print(f"{'='*50}")

    # Step 1 — Load and chunk PDF
    chunks = load_and_chunk_pdf(pdf_path)

    # Step 2 — Build or load FAISS index
    index, chunks = build_faiss_index(chunks, pdf_path)

    # Step 3 — Agent 1: Retrieve relevant chunks
    retriever_output = retriever_agent(query, index, chunks, top_k=5)

    # Step 4 — Agent 2: Summarise/compress context
    summariser_output = summariser_agent(
        retriever_output["retrieved_chunks"],
        query
    )

    # Step 5 — Agent 3: Synthesise final answer
    final_output = synthesiser_agent(
        summariser_output["compressed_context"],
        query,
        low_confidence=retriever_output["low_confidence"]
    )

    print(f"\n{'='*50}")
    print(f"✅ Pipeline Complete")
    print(f"{'='*50}\n")

    return {
        "query": query,
        "answer": final_output["answer"],
        "context_used": summariser_output["compressed_context"],
        "low_confidence": retriever_output["low_confidence"]
    }
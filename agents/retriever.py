from core.embeddings import search_similar_chunks

def retriever_agent(query: str, index, chunks: list, top_k: int = 5) -> dict:
    """
    AGENT 1 — Retriever
    Takes the user's question, searches FAISS for the most
    semantically similar chunks, returns them for the next agent.
    """
    print(f"\n🔍 [Retriever Agent] Searching for: '{query}'")

    results = search_similar_chunks(query, index, chunks, top_k=top_k)

    # Pull out just the text of each chunk
    retrieved_chunks = [r["chunk"] for r in results]
    distances = [r["distance"] for r in results]

    # Confidence check — if best match distance is too high, context may be poor
    best_distance = distances[0] if distances else 999
    low_confidence = best_distance > 1.5

    if low_confidence:
        print(f"⚠️  [Retriever Agent] Low confidence — best distance: {best_distance:.2f}. Document may not contain relevant info.")
    else:
        print(f"✅ [Retriever Agent] Retrieved {len(retrieved_chunks)} chunks — best distance: {best_distance:.2f}")

    return {
        "query": query,
        "retrieved_chunks": retrieved_chunks,
        "distances": distances,
        "low_confidence": low_confidence
    }
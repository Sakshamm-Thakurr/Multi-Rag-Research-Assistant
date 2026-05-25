import os
import faiss
import pickle
import hashlib
import numpy as np
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv


FAISS_INDEX_PATH = "faiss_index.bin"
CHUNKS_PATH = "chunks.pkl"
HASH_PATH = "pdf_hash.txt"


def get_pdf_hash(pdf_path: str) -> str:
    """
    Creates a unique fingerprint of the PDF file.
    If the same PDF is uploaded again, we skip re-embedding — saves API calls.
    """
    with open(pdf_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def get_embedding_model():
    """
    Loads the Gemini embedding model.
    """
    load_dotenv()  # force reload here too
    api_key = os.getenv("GOOGLE_API_KEY")
    print(f"🔑 API Key loaded: {api_key is not None}")  # debug line
    
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key
    )


def build_faiss_index(chunks: list, pdf_path: str):
    """
    Embeds all chunks and builds a FAISS index.
    Saves index + chunks to disk for reuse.
    Skips re-embedding if same PDF was already indexed.
    """

    # Check if this PDF was already indexed
    current_hash = get_pdf_hash(pdf_path)
    if os.path.exists(HASH_PATH):
        with open(HASH_PATH, "r") as f:
            saved_hash = f.read().strip()
        if saved_hash == current_hash:
            print("✅ Same PDF detected — loading existing FAISS index (no re-embedding)")
            return load_faiss_index()

    print(f"🔄 Embedding {len(chunks)} chunks with Gemini...")

    embedding_model = get_embedding_model()

    # Embed all chunks — returns list of 768-dim vectors
    vectors = embedding_model.embed_documents(chunks)

    print(f"✅ Embeddings created: {len(vectors)} vectors, {len(vectors[0])} dimensions each")

    # Build FAISS index
    dimension = len(vectors[0])  # 768 for Gemini
    index = faiss.IndexFlatL2(dimension)  # L2 = Euclidean distance search

    import numpy as np
    index.add(np.array(vectors).astype("float32"))

    # Save everything to disk
    faiss.write_index(index, FAISS_INDEX_PATH)

    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)

    with open(HASH_PATH, "w") as f:
        f.write(current_hash)

    print(f"✅ FAISS index saved — {index.ntotal} vectors stored")
    return index, chunks


def load_faiss_index():
    """
    Loads a previously saved FAISS index from disk.
    """
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)
    print(f"✅ FAISS index loaded — {index.ntotal} vectors, {len(chunks)} chunks")
    return index, chunks


def search_similar_chunks(query: str, index, chunks: list, top_k: int = 5):
    """
    Agent 1 (Retriever) uses this.
    Converts query to vector, finds top_k most similar chunks in FAISS.
    Returns the actual text of those chunks.
    """
    embedding_model = get_embedding_model()

    # Embed the query
    query_vector = embedding_model.embed_query(query)

    import numpy as np
    query_array = np.array([query_vector]).astype("float32")

    # Search FAISS for top_k nearest vectors
    distances, indices = index.search(query_array, top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        if idx != -1:  # -1 means no result found
            results.append({
                "chunk": chunks[idx],
                "distance": float(distances[0][i]),
                "chunk_index": int(idx)
            })

    print(f"✅ Retrieved {len(results)} relevant chunks for query")
    return results
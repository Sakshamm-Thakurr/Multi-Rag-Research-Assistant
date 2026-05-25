import streamlit as st
import os
import tempfile
from core.document_loader import load_and_chunk_pdf
from core.embeddings import build_faiss_index
from agents.retriever import retriever_agent
from agents.summariser import summariser_agent
from agents.synthesiser import synthesiser_agent

# ── Page config ──────────────────────────────────────────
st.set_page_config(
    page_title="RAG Research Assistant",
    page_icon="🧠",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stTextInput > div > div > input { background-color: #1e2130; color: white; }
    .answer-box {
        background-color: #1e2130;
        border-left: 4px solid #4f8ef7;
        padding: 20px;
        border-radius: 8px;
        margin-top: 10px;
        font-size: 16px;
        line-height: 1.8;
    }
    .context-box {
        background-color: #1a1f2e;
        border-left: 4px solid #3ecf8e;
        padding: 15px;
        border-radius: 8px;
        font-size: 13px;
        color: #8892a4;
    }
    .agent-step {
        background-color: #151c32;
        border: 1px solid #2a3550;
        padding: 10px 15px;
        border-radius: 6px;
        margin: 5px 0;
        font-size: 13px;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────
st.title("🧠 Multi-Agent RAG Research Assistant")
st.caption("Upload any PDF → Ask questions → Get accurate, document-grounded answers")
st.divider()

# ── Session state ─────────────────────────────────────────
if "index" not in st.session_state:
    st.session_state.index = None
if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ── Layout ────────────────────────────────────────────────
col1, col2 = st.columns([1, 2])

# ── LEFT PANEL — Upload ───────────────────────────────────
with col1:
    st.subheader("📄 Upload Document")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Upload any PDF — textbook, research paper, notes, report"
    )

    if uploaded_file is not None:
        if uploaded_file.name != st.session_state.pdf_name:
            with st.spinner("📖 Reading and indexing PDF..."):
                try:
                    # Save uploaded file to temp location
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name

                    # Load, chunk, embed
                    chunks = load_and_chunk_pdf(tmp_path)
                    index, chunks = build_faiss_index(chunks, tmp_path)

                    # Store in session
                    st.session_state.index = index
                    st.session_state.chunks = chunks
                    st.session_state.pdf_name = uploaded_file.name
                    st.session_state.chat_history = []

                    os.unlink(tmp_path)  # clean up temp file

                    st.success(f"✅ Indexed {len(chunks)} chunks!")

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

    # Show document info if loaded
    if st.session_state.pdf_name:
        st.markdown("---")
        st.markdown("**📚 Current Document**")
        st.info(f"📄 {st.session_state.pdf_name}")
        st.metric("Chunks indexed", len(st.session_state.chunks) if st.session_state.chunks else 0)

    # Agent pipeline explanation
    st.markdown("---")
    st.markdown("**🤖 Agent Pipeline**")
    st.markdown('<div class="agent-step">🔍 <b>Agent 1 — Retriever</b><br>Finds relevant chunks via FAISS vector search</div>', unsafe_allow_html=True)
    st.markdown('<div class="agent-step">📝 <b>Agent 2 — Summariser</b><br>Compresses context, removes noise</div>', unsafe_allow_html=True)
    st.markdown('<div class="agent-step">🧠 <b>Agent 3 — Synthesiser</b><br>Generates grounded final answer</div>', unsafe_allow_html=True)

# ── RIGHT PANEL — Chat ────────────────────────────────────
with col2:
    st.subheader("💬 Ask Questions")

    if st.session_state.index is None:
        st.info("👈 Upload a PDF on the left to get started")
    else:
        # Display chat history
        for chat in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(chat["question"])
            with st.chat_message("assistant"):
                st.markdown(f'<div class="answer-box">{chat["answer"]}</div>', unsafe_allow_html=True)
                if chat.get("low_confidence"):
                    st.warning("⚠️ Low confidence — document may not contain relevant info")
                with st.expander("📋 View compressed context used"):
                    st.markdown(f'<div class="context-box">{chat["context"]}</div>', unsafe_allow_html=True)

        # Query input
        query = st.chat_input("Ask anything about your document...")

        if query:
            with st.chat_message("user"):
                st.write(query)

            with st.chat_message("assistant"):
                with st.spinner("🤖 Agents working..."):
                    try:
                        # Run all 3 agents
                        retriever_out = retriever_agent(query, st.session_state.index, st.session_state.chunks)
                        summariser_out = summariser_agent(retriever_out["retrieved_chunks"], query)
                        synthesiser_out = synthesiser_agent(
                            summariser_out["compressed_context"],
                            query,
                            low_confidence=retriever_out["low_confidence"]
                        )

                        answer = synthesiser_out["answer"]
                        context = summariser_out["compressed_context"]
                        low_conf = retriever_out["low_confidence"]

                        # Display answer
                        st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

                        if low_conf:
                            st.warning("⚠️ Low confidence — document may not contain relevant info")

                        with st.expander("📋 View compressed context used"):
                            st.markdown(f'<div class="context-box">{context}</div>', unsafe_allow_html=True)

                        # Save to history
                        st.session_state.chat_history.append({
                            "question": query,
                            "answer": answer,
                            "context": context,
                            "low_confidence": low_conf
                        })

                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
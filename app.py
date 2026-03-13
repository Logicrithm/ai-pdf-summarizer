"""
AI PDF Summarizer — Streamlit Application
------------------------------------------
Steps 5, 13-16: 4-tab UI with progress, streaming, caching, citations.
Powered by Groq (LLaMA 3.1) + local sentence-transformers embeddings.
"""

import hashlib
import os
import shutil
import streamlit as st
from groq import Groq
from dotenv import load_dotenv

from pdf_reader import extract_pages
from utils import chunk_pages
from embeddings import embed_chunks, build_faiss_index
from cache import load_cache, save_cache, get_file_hash
from summarizer import summarize_document_streaming, verify_summary
from qa_engine import answer_question_streaming

load_dotenv()

# ──────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────
st.set_page_config(page_title="AI PDF Summarizer", page_icon="📄", layout="wide")

# ──────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    font-size: 2.8rem; font-weight: 700; text-align: center; margin-bottom: 0;
}
.sub-header { text-align: center; color: #888; font-size: 1.1rem; margin-bottom: 2rem; }
.stat-card {
    background: rgba(255,255,255,0.05); border-radius: 12px; padding: 1rem;
    text-align: center; border: 1px solid rgba(255,255,255,0.1);
}
.stat-number {
    font-size: 1.8rem; font-weight: 700;
    background: linear-gradient(135deg, #667eea, #764ba2);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.stat-label { color: #888; font-size: 0.85rem; margin-top: 0.2rem; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────
st.markdown('<p class="main-header">📄 AI PDF Summarizer</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Upload a PDF → Get structured summaries → Ask questions with citations</p>', unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Session State Init
# ──────────────────────────────────────────────
for key in ["chunks", "summary", "faiss_index", "file_hash",
            "verification", "section_sources", "key_concepts"]:
    if key not in st.session_state:
        st.session_state[key] = None
if "doc_processed" not in st.session_state:
    st.session_state.doc_processed = False
if "used_cache" not in st.session_state:
    st.session_state.used_cache = False

# ──────────────────────────────────────────────
# Groq Client
# ──────────────────────────────────────────────
@st.cache_resource
def get_client():
    return Groq(api_key=st.secrets["GROQ_API_KEY"])

client = get_client()

# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📤 Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    if uploaded_file:
        st.success(f"✅ **{uploaded_file.name}**")

    # Step 16: Clear button
    if st.button("🗑️ Clear Document"):
        st.session_state.clear()
        st.rerun()

    if st.button("🗂️ Clear Cache"):
        cache_dir = os.path.join(os.path.dirname(__file__), "cache")
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
            os.makedirs(cache_dir, exist_ok=True)
        st.success("Cache cleared!")

    st.markdown("---")
    st.markdown('<div style="text-align:center;color:#666;font-size:0.8rem;">Powered by Groq · FAISS · PyMuPDF</div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Main Content
# ──────────────────────────────────────────────
if uploaded_file:
    pdf_bytes = uploaded_file.read()
    file_hash = get_file_hash(pdf_bytes)

    # ── Step 5: Multi-PDF hash reset ──
    if st.session_state.file_hash != file_hash:
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.session_state.file_hash = file_hash
        st.session_state.doc_processed = False
        st.session_state.used_cache = False
        for key in ["chunks", "summary", "faiss_index", "verification",
                     "section_sources", "key_concepts"]:
            st.session_state[key] = None

    # ── Process PDF ──
    if not st.session_state.doc_processed:
        # Check cache first
        cached = load_cache(pdf_bytes)
        if cached is not None:
            st.session_state.chunks = cached["chunks"]
            st.session_state.faiss_index = cached["faiss_index"]
            st.session_state.summary = cached.get("summary")
            st.session_state.used_cache = True
            st.session_state.doc_processed = True
            st.rerun()

        # ── Step 14: Progress indicators ──
        status = st.empty()
        progress = st.progress(0)

        status.text("📄 Extracting text...")
        pages = extract_pages(pdf_bytes)
        progress.progress(0.2)

        status.text("✂️ Building chunks...")
        chunks = chunk_pages(pages, chunk_size=800, overlap=100)
        st.session_state.chunks = chunks
        progress.progress(0.4)

        status.text("🧠 Generating embeddings...")
        embeddings = embed_chunks(client, chunks, max_workers=4)
        progress.progress(0.7)

        status.text("📦 Building search index...")
        faiss_index = build_faiss_index(embeddings)
        st.session_state.faiss_index = faiss_index
        progress.progress(0.9)

        status.text("💾 Saving to cache...")
        save_cache(pdf_bytes, chunks, embeddings, faiss_index)
        progress.progress(1.0)

        status.text("✅ Ready!")
        progress.empty()
        status.empty()

        st.session_state.doc_processed = True
        st.rerun()

    # ── Document processed ──
    chunks = st.session_state.chunks
    faiss_index = st.session_state.faiss_index

    if st.session_state.used_cache:
        st.success("⚡ Loaded from cache — instant startup!")

    # Stats row
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'<div class="stat-card"><div class="stat-number">{len(chunks)}</div><div class="stat-label">Chunks</div></div>', unsafe_allow_html=True)
    with col2:
        pages_set = sorted(set(c["page"] for c in chunks))
        st.markdown(f'<div class="stat-card"><div class="stat-number">{len(pages_set)}</div><div class="stat-label">Pages</div></div>', unsafe_allow_html=True)
    with col3:
        word_count = sum(len(c["text"].split()) for c in chunks)
        st.markdown(f'<div class="stat-card"><div class="stat-number">{word_count:,}</div><div class="stat-label">Words</div></div>', unsafe_allow_html=True)

    st.markdown("---")

    # ── Step 13: 4-tab layout ──
    tab1, tab2, tab3, tab4 = st.tabs([
        "📄 Summary", "💡 Key Concepts", "🔍 Ask Questions", "✅ Verification"
    ])

    # ═══ Tab 1: Full Summary ═══
    with tab1:
        if st.session_state.summary:
            st.markdown(st.session_state.summary)

            # Step 15: Citations
            if st.session_state.section_sources:
                st.markdown("---")
                st.markdown("#### 📌 Section Sources")
                for section, pages in st.session_state.section_sources.items():
                    if pages:
                        page_str = ", ".join(str(p) for p in pages)
                        st.markdown(f"> **{section.title()}**: Pages {page_str}")

            if st.button("🔄 Re-summarize"):
                st.session_state.summary = None
                st.session_state.verification = None
                st.session_state.section_sources = None
                st.session_state.key_concepts = None
                st.rerun()
        else:
            if st.button("✨ Generate Summary", type="primary", use_container_width=True):
                progress_bar = st.progress(0, text="Retrieving sections...")

                def update_progress(current, total):
                    progress_bar.progress(current / total, text=f"✅ Section {current}/{total} processed")

                summary_placeholder = st.empty()
                full_summary = ""
                section_sources = {}

                for msg_type, content, cur, tot in summarize_document_streaming(
                    client, chunks, faiss_index, progress_callback=update_progress
                ):
                    if msg_type == "token":
                        full_summary += content
                        summary_placeholder.markdown(full_summary + "▌")
                    elif msg_type == "sources":
                        section_sources = content
                    elif msg_type == "done":
                        full_summary = content

                summary_placeholder.markdown(full_summary)
                progress_bar.empty()

                st.session_state.summary = full_summary
                st.session_state.section_sources = section_sources

                # Step 10: Verification pass
                with st.spinner("🔍 Running verification..."):
                    verification = verify_summary(client, chunks, full_summary)
                    st.session_state.verification = verification

                # Update cache with summary
                from embeddings import embed_chunks as _ec
                embeddings = _ec(client, chunks, max_workers=4)
                save_cache(pdf_bytes, chunks, embeddings, faiss_index, full_summary)

                st.rerun()

    # ═══ Tab 2: Key Concepts ═══
    with tab2:
        if st.session_state.summary:
            # Extract Key Technical Concepts section from summary
            summary = st.session_state.summary
            concepts_start = summary.find("## Key Technical Concepts")
            if concepts_start != -1:
                # Find next section
                next_section = summary.find("## ", concepts_start + 1)
                if next_section == -1:
                    concepts_text = summary[concepts_start:]
                else:
                    concepts_text = summary[concepts_start:next_section]
                st.markdown(concepts_text)
            else:
                st.info("Key concepts will appear after generating a summary.")

            if st.session_state.section_sources and "concepts" in st.session_state.section_sources:
                pages = st.session_state.section_sources["concepts"]
                if pages:
                    st.markdown(f"\n> Source: Pages {', '.join(str(p) for p in pages)}")
        else:
            st.info("📝 Generate a summary first to see key concepts.")

    # ═══ Tab 3: Ask Questions ═══
    with tab3:
        st.markdown("#### 💬 Ask anything about your document")
        question = st.text_input(
            "Your question",
            placeholder="e.g. What are the limitations of this approach?",
            label_visibility="collapsed",
        )

        if question:
            answer_placeholder = st.empty()
            full_answer = ""
            source_pages = []

            for msg_type, content in answer_question_streaming(
                client, question, chunks, faiss_index
            ):
                if msg_type == "token":
                    full_answer += content
                    answer_placeholder.markdown(full_answer + "▌")
                elif msg_type == "sources":
                    source_pages = content
                elif msg_type == "no_match":
                    answer_placeholder.markdown(content)

            if full_answer:
                answer_placeholder.markdown(full_answer)

            # Step 15: Show sources below answer
            if source_pages:
                st.markdown(f"\n> **Source:** Pages {', '.join(str(p) for p in source_pages)}")

    # ═══ Tab 4: Verification ═══
    with tab4:
        if st.session_state.verification:
            is_clean = "all claims supported" in st.session_state.verification.lower()
            icon = "✅" if is_clean else "⚠️"
            st.markdown(f"### {icon} Verification Report")
            with st.expander("View Full Verification Report", expanded=True):
                st.markdown(st.session_state.verification)
        else:
            st.info("📝 Generate a summary first to see the verification report.")

else:
    # ── Landing state ──
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("### 📄 Summary")
        st.markdown("Structured summaries with page citations.")
    with col2:
        st.markdown("### 💡 Concepts")
        st.markdown("Key technical concepts grouped by category.")
    with col3:
        st.markdown("### 🔍 Q&A")
        st.markdown("Ask questions with grounded, cited answers.")
    with col4:
        st.markdown("### ✅ Verify")
        st.markdown("Hallucination checking on every summary.")

    st.markdown("---")
    st.info("👈 Upload a PDF from the sidebar to get started.")

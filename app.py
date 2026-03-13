"""
AI Research Assistant — Streamlit Application
----------------------------------------------
Production-ready PDF summarizer and Q&A tool.
Powered by Groq (LLaMA 3.3) + FAISS + sentence-transformers.
"""

import hashlib
import os
import shutil
import time
from datetime import date
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

# ══════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════
FREE_DAILY_LIMIT = 3  # Free PDFs per day
PRODUCT_NAME = "AI Research Assistant"

# ──────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title=PRODUCT_NAME,
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Custom CSS — Premium Dark Theme
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
.sub-header {
    text-align: center; color: #888; font-size: 1.15rem; margin-bottom: 0.5rem;
}
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

/* Feature cards on landing page */
.feature-card {
    background: linear-gradient(145deg, rgba(102,126,234,0.08), rgba(118,75,162,0.05));
    border: 1px solid rgba(102,126,234,0.2);
    border-radius: 16px; padding: 1.5rem; text-align: center;
    transition: transform 0.2s, border-color 0.2s;
}
.feature-card:hover {
    transform: translateY(-2px);
    border-color: rgba(102,126,234,0.5);
}
.feature-icon { font-size: 2.2rem; margin-bottom: 0.5rem; }
.feature-title { font-weight: 600; font-size: 1.05rem; margin-bottom: 0.3rem; }
.feature-desc { color: #999; font-size: 0.88rem; line-height: 1.4; }

/* Usage badge */
.usage-badge {
    background: linear-gradient(135deg, #667eea20, #764ba220);
    border: 1px solid #667eea40;
    border-radius: 8px; padding: 0.5rem 0.8rem;
    text-align: center; font-size: 0.85rem; margin-bottom: 1rem;
}

/* Upgrade banner */
.upgrade-banner {
    background: linear-gradient(135deg, #667eea, #764ba2);
    border-radius: 12px; padding: 1rem 1.2rem;
    text-align: center; color: white; font-size: 0.9rem;
    margin-top: 0.5rem;
}
.upgrade-banner a { color: #fff; font-weight: 600; text-decoration: underline; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────
st.markdown(f'<p class="main-header">🧠 {PRODUCT_NAME}</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Upload research papers & PDFs → Get instant summaries → Ask questions with page citations</p>', unsafe_allow_html=True)

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

# ── Usage tracking (per-day limits) ──
if "usage_date" not in st.session_state:
    st.session_state.usage_date = str(date.today())
if "pdf_count" not in st.session_state:
    st.session_state.pdf_count = 0

# Reset counter if it's a new day
if st.session_state.usage_date != str(date.today()):
    st.session_state.usage_date = str(date.today())
    st.session_state.pdf_count = 0

# ──────────────────────────────────────────────
# Groq Client
# ──────────────────────────────────────────────
@st.cache_resource
def get_client():
    # Works with both secrets.toml (local) and env vars (Render)
    api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
    if not api_key:
        st.error("⚠️ GROQ_API_KEY not found. Set it in `.streamlit/secrets.toml` or as an environment variable.")
        st.stop()
    return Groq(api_key=api_key)

client = get_client()

# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📤 Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    if uploaded_file:
        st.success(f"✅ **{uploaded_file.name}**")

    # Usage counter
    remaining = max(0, FREE_DAILY_LIMIT - st.session_state.pdf_count)
    st.markdown(
        f'<div class="usage-badge">📊 <b>{remaining}/{FREE_DAILY_LIMIT}</b> free PDFs remaining today</div>',
        unsafe_allow_html=True,
    )

    # Clear buttons
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🗑️ Clear Doc", use_container_width=True):
            st.session_state.clear()
            st.rerun()
    with col_b:
        if st.button("🗂️ Clear Cache", use_container_width=True):
            cache_dir = os.path.join(os.path.dirname(__file__), "cache")
            if os.path.exists(cache_dir):
                shutil.rmtree(cache_dir)
                os.makedirs(cache_dir, exist_ok=True)
            st.success("Cache cleared!")

    st.markdown("---")

    # Upgrade CTA
    if remaining <= 1:
        st.markdown(
            '<div class="upgrade-banner">'
            '🚀 Need more? <b>Upgrade to Pro</b> for unlimited PDFs.<br>'
            '<small>₹199/month · Cancel anytime</small>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown("")

    st.markdown(
        '<div style="text-align:center;color:#555;font-size:0.75rem;margin-top:0.5rem;">'
        'Powered by Groq · FAISS · PyMuPDF<br>'
        '© 2026 AI Research Assistant'
        '</div>',
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════
# Main Content
# ══════════════════════════════════════════════
if uploaded_file:
    pdf_bytes = uploaded_file.read()
    file_hash = get_file_hash(pdf_bytes)

    # ── Multi-PDF hash reset ──
    if st.session_state.file_hash != file_hash:
        old_hash = st.session_state.file_hash
        for key in list(st.session_state.keys()):
            if key not in ("usage_date", "pdf_count"):
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

        # ── Usage limit check ──
        if st.session_state.pdf_count >= FREE_DAILY_LIMIT:
            st.warning(
                f"⚠️ **Free daily limit reached** ({FREE_DAILY_LIMIT} PDFs/day).\n\n"
                "Upgrade to **Pro** for unlimited PDFs, or come back tomorrow!"
            )
            st.stop()

        # Count this PDF
        st.session_state.pdf_count += 1

        # ── Progress indicators ──
        status = st.empty()
        progress = st.progress(0)

        status.text("📄 Extracting text from PDF...")
        progress.progress(10)
        pages = extract_pages(pdf_bytes)

        status.text("✂️ Splitting into chunks...")
        progress.progress(25)
        chunks = chunk_pages(pages, chunk_size=800, overlap=100)
        st.session_state.chunks = chunks

        status.text("🔢 Generating embeddings...")
        progress.progress(45)
        embeddings = embed_chunks(client, chunks, max_workers=4)

        status.text("🗂️ Building search index...")
        progress.progress(65)
        faiss_index = build_faiss_index(embeddings)
        st.session_state.faiss_index = faiss_index

        status.text("💾 Saving to cache...")
        progress.progress(85)
        save_cache(pdf_bytes, chunks, embeddings, faiss_index)

        status.text("✅ Done! Your document is ready.")
        progress.progress(100)
        time.sleep(1.5)
        status.empty()
        progress.empty()

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

    # ── 4-tab layout ──
    tab1, tab2, tab3, tab4 = st.tabs([
        "📄 Summary", "💡 Key Concepts", "🔍 Ask Questions", "✅ Verification"
    ])

    # ═══ Tab 1: Full Summary ═══
    with tab1:
        if st.session_state.summary:
            st.markdown(st.session_state.summary)

            # Citations
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

                # Verification pass
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
            summary = st.session_state.summary
            concepts_start = summary.find("## Key Technical Concepts")
            if concepts_start != -1:
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

            # Show sources below answer
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
    # ══════════════════════════════════════════
    # Landing Page — No PDF uploaded
    # ══════════════════════════════════════════
    st.markdown("")

    # Feature cards
    col1, col2, col3, col4 = st.columns(4)
    features = [
        ("📄", "Instant Summaries", "5-section structured summaries with page citations in under 30 seconds."),
        ("💡", "Key Concepts", "Auto-extracted concepts grouped by category with source pages."),
        ("🔍", "Smart Q&A", "Ask any question — get grounded, cited answers from your document."),
        ("✅", "Hallucination Check", "Every summary is verified against source text automatically."),
    ]
    for col, (icon, title, desc) in zip([col1, col2, col3, col4], features):
        with col:
            st.markdown(
                f'<div class="feature-card">'
                f'<div class="feature-icon">{icon}</div>'
                f'<div class="feature-title">{title}</div>'
                f'<div class="feature-desc">{desc}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("")
    st.markdown("")

    # How it works
    st.markdown("### ⚡ How It Works")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("**1. Upload** 📤\n\nDrop any research paper or PDF into the sidebar.")
    with col_b:
        st.markdown("**2. Analyze** 🧠\n\nAI reads every page, extracts key information, and builds a search index.")
    with col_c:
        st.markdown("**3. Explore** 🔍\n\nRead the summary, explore concepts, or ask specific questions with citations.")

    st.markdown("---")

    # Pricing teaser
    col_free, col_pro = st.columns(2)
    with col_free:
        st.markdown(
            "### 🆓 Free Plan\n"
            f"- **{FREE_DAILY_LIMIT} PDFs per day**\n"
            "- Full summaries with citations\n"
            "- Q&A with page references\n"
            "- Verification reports\n"
        )
    with col_pro:
        st.markdown(
            "### 🚀 Pro Plan — ₹199/month\n"
            "- **Unlimited PDFs**\n"
            "- Priority processing\n"
            "- Longer documents supported\n"
            "- Email support\n"
        )

    st.markdown("---")
    st.info("👈 Upload a PDF from the sidebar to get started — it's free!")

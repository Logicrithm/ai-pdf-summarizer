# 📄 AI PDF Summarizer

A production-quality AI-powered PDF summarizer and Q&A tool built with **Streamlit**, **Groq (LLaMA 3.3)**, **FAISS**, and **Sentence Transformers**.

Upload any PDF → Get a structured summary with citations → Ask questions with grounded, cited answers.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red?logo=streamlit)
![Groq](https://img.shields.io/badge/LLM-Groq_LLaMA_3.3-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## ✨ Features

| Feature | Description |
|---|---|
| 📄 **Structured Summaries** | 5-section summaries: Overview, Key Concepts, Findings, Applications, Limitations |
| 📌 **Page Citations** | Every section cites the exact source pages used |
| 🔍 **RAG Q&A** | Ask questions with grounded answers pulled from the document |
| ⚠️ **Out-of-scope Rejection** | Questions unrelated to the PDF are rejected, not hallucinated |
| ✅ **Verification** | Automatic hallucination checking on every summary |
| ⚡ **Smart Caching** | Repeat uploads load instantly from disk cache |
| 🔄 **Multi-PDF Support** | Upload a new PDF → full pipeline reset, clean processing |
| 💡 **Concept Grouping** | Dual-query retrieval for comprehensive concept extraction |
| 🛡️ **Rate Limit Handling** | Auto-retry with exponential backoff for Groq API limits |

---

## 🏗️ Architecture

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  PDF Upload  │───▶│  Text + Page │───▶│   Chunking   │
│  (PyMuPDF)   │    │  Extraction  │    │  (800 chars) │
└──────────────┘    └──────────────┘    └──────┬───────┘
                                               │
                    ┌──────────────┐    ┌───────▼───────┐
                    │  FAISS Index │◀───│  Embeddings   │
                    │  (L2 Search) │    │ (MiniLM-L6)   │
                    └──────┬───────┘    └───────────────┘
                           │
              ┌────────────▼────────────┐
              │  Section-Aware Summary  │
              │  (5 FAISS queries →     │
              │   per-section LLM call  │
              │   → merge → verify)     │
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │    RAG Q&A Engine       │
              │  (Embed → Search →      │
              │   Confidence Filter →   │
              │   Stream Answer)        │
              └─────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/ai-pdf-summarizer.git
cd ai-pdf-summarizer
```

### 2. Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up your Groq API key

Get a free key at [console.groq.com](https://console.groq.com)

```bash
# Create .env file
echo GROQ_API_KEY=your_key_here > .env

# Create Streamlit secrets
mkdir .streamlit
echo GROQ_API_KEY = "your_key_here" > .streamlit/secrets.toml
```

### 5. Run the app

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`. First run downloads the embedding model (~90MB).

---

## 📁 Project Structure

```
ai-pdf-summarizer/
├── app.py              # Streamlit UI — 4-tab layout with streaming
├── pdf_reader.py       # PDF text extraction with PyMuPDF
├── utils.py            # Text chunking with overlap
├── embeddings.py       # Local embeddings (sentence-transformers)
├── vector_store.py     # FAISS index building and search
├── summarizer.py       # Section-aware summarization pipeline
├── qa_engine.py        # RAG Q&A with confidence filtering
├── prompts.py          # All LLM prompt templates
├── cache.py            # MD5-based disk caching
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variable template
├── .gitignore          # Git ignore rules
└── README.md           # This file
```

---

## 🔧 Configuration

| Parameter | File | Default | Description |
|---|---|---|---|
| `CONFIDENCE_THRESHOLD` | `qa_engine.py` | `1.2` | L2 distance cutoff for Q&A relevance |
| `MAX_CHUNK_CHARS` | `summarizer.py` | `400` | Max chars per chunk sent to LLM |
| `CHUNK_MODEL` | `summarizer.py` | `llama-3.3-70b-versatile` | Groq model for summarization |
| `chunk_size` | `app.py` | `800` | Characters per text chunk |
| `overlap` | `app.py` | `100` | Overlap between chunks |

---

## 📝 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- [Groq](https://groq.com) — Ultra-fast LLM inference
- [Streamlit](https://streamlit.io) — Python web framework
- [FAISS](https://github.com/facebookresearch/faiss) — Vector similarity search
- [Sentence Transformers](https://www.sbert.net/) — Local embedding models
- [PyMuPDF](https://pymupdf.readthedocs.io/) — PDF text extraction

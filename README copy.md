# 🔬 AI Skin Disease Detection System

A production-grade AI-powered skin disease detection and risk assessment system built with **Streamlit**, **LangChain**, **HuggingFace**, and **Groq** APIs.

> ⚠️ **Disclaimer**: This system is for educational purposes only and does not replace professional dermatological consultation.

---

## ✨ Features

- **Image Classification** — Upload a skin lesion photo for AI-powered disease prediction via HuggingFace Inference API
- **Multi-Modal Risk Assessment** — Combines AI confidence with patient metadata using weighted medical scoring
- **LLM-Powered Analysis** — LangChain agent (Groq LLaMA3) provides structured medical explanations
- **RAG System** — FAISS vector store with trusted medical knowledge (WHO, NIH, Mayo Clinic)
- **Research Tools** — Tavily search, PubMed, and Wikipedia for evidence-based retrieval
- **PDF Reports** — Downloadable diagnostic reports
- **Safety First** — Confidence thresholds, cancer warnings, and always-visible disclaimers

---

## 🏗️ Architecture

```
skin-disease-detection/
├── models/
│   ├── classifier.py          # HuggingFace Inference API classification
│   └── risk_assessment.py     # Weighted medical risk scoring
├── agent/
│   ├── llm_agent.py           # LangChain agent (Groq/HuggingFace)
│   └── tools.py               # Search & retrieval tools
├── rag/
│   ├── vector_store.py        # FAISS vector store
│   └── medical_knowledge.py   # Trusted medical text corpus
├── frontend/
│   └── app.py                 # Streamlit UI
├── utils/
│   ├── config.py              # Configuration management
│   ├── logger.py              # Structured logging
│   └── pdf_report.py          # PDF report generator
├── requirements.txt
├── .env.example
├── run.py
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone and install

```bash
cd skin-disease-detection
pip install -r requirements.txt
```

### 2. Set up API keys

```bash
cp .env.example .env
# Edit .env with your keys
```

Or enter them directly in the Streamlit sidebar.

**Required API keys:**

| Key | Source | Purpose |
|-----|--------|---------|
| `HUGGINGFACE_API_KEY` | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) | Image classification & embeddings |
| `GROQ_API_KEY` | [console.groq.com](https://console.groq.com) | LLM analysis (LLaMA3) |
| `TAVILY_API_KEY` | [tavily.com](https://tavily.com) | Web search for medical info |

### 3. Run

```bash
python run.py
# or
streamlit run frontend/app.py
```

---

## 🔒 Safety Rules

- Never claims definitive diagnosis
- Cancer probability > 70% → **Urgent consultation warning**
- Model confidence < 60% → **Low certainty warning**
- Disclaimer visible on every page

---

## 📋 How It Works

1. **Upload** a skin lesion image
2. **AI classifies** the lesion using a cloud-hosted dermatology model
3. **Enter patient info** (age, symptoms, history)
4. **Risk engine** combines AI output with clinical metadata
5. **LLM agent** generates a structured explanation with cited sources
6. **Download** a PDF report of the full analysis

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **LLM**: Groq LLaMA3 / HuggingFace Inference API
- **Image AI**: HuggingFace Inference API (no local weights)
- **RAG**: FAISS + HuggingFace Embeddings API
- **Tools**: Tavily, PubMed (NCBI E-utilities), Wikipedia
- **PDF**: FPDF2

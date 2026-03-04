# 🔬 AI Skin Disease Detection System — Complete Architecture Guide

> **Tech Stack:** Python · Streamlit · LangChain · Groq API · HuggingFace · FAISS · Tavily · FPDF2  
> **AI Models:** Anwarkh1/Skin_Cancer-Image_Classification (HF Classifier) · LLaMA 4 Scout 17B (Vision) · LLaMA 3.3 70B (Analysis LLM) · all-MiniLM-L6-v2 (Embeddings)

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Directory Structure](#2-directory-structure)
3. [End-to-End Pipeline Workflow](#3-end-to-end-pipeline-workflow)
4. [AI Models — What's Used & Where](#4-ai-models--whats-used--where)
5. [Module Deep Dive](#5-module-deep-dive)
   - [5.1 Image Classifier (`models/classifier.py`)](#51-image-classifier)
   - [5.2 Risk Assessment Engine (`models/risk_assessment.py`)](#52-risk-assessment-engine)
   - [5.3 Vision-Enhanced LLM Agent (`agent/llm_agent.py`)](#53-vision-enhanced-llm-agent)
   - [5.4 Agent Tools (`agent/tools.py`)](#54-agent-tools)
   - [5.5 RAG System (`rag/`)](#55-rag-system)
   - [5.6 Frontend (`frontend/app.py`)](#56-frontend)
   - [5.7 Utilities (`utils/`)](#57-utilities)
6. [Algorithms & Scoring Logic](#6-algorithms--scoring-logic)
7. [API Keys & Configuration](#7-api-keys--configuration)
8. [Data Flow Diagram](#8-data-flow-diagram)
9. [Supported Skin Conditions](#9-supported-skin-conditions)
10. [Architecture Evolution Note](#10-architecture-evolution-note)
11. [Safety & Disclaimer](#11-safety--disclaimer)

---

## 1. System Overview

This project is an **AI-powered skin disease detection and analysis system** that uses a **dual-model architecture** — combining a **dedicated image classifier** with a **vision-capable LLM** to produce highly accurate and comprehensive skin disease analysis.

### How It Works (High-Level)

A user uploads a photo of a skin lesion + fills in patient metadata (age, symptoms, history), and the system runs a **multi-stage AI pipeline**:

1. **Classifies** the lesion using the `Anwarkh1/Skin_Cancer-Image_Classification` model (a fine-tuned HuggingFace image classifier trained specifically on skin cancer images).
2. **Assesses risk** via a weighted scoring algorithm combining classifier output + patient data.
3. **Gets an independent vision diagnosis** from LLaMA 4 Scout — not constrained to fixed labels, can detect rare/unusual conditions.
4. **Searches the internet** (Tavily) for the identified condition.
5. **Retrieves curated medical knowledge** from a RAG corpus (WHO, NIH, Mayo Clinic).
6. **Combines BOTH model results** — the classifier predictions AND the vision diagnosis — through a large language model (LLaMA 3.3 70B) which synthesizes everything into a comprehensive analysis.
7. **Generates a downloadable PDF report.**
8. **Supports follow-up chat** for user questions.

### The Dual-Model Design Philosophy

```
                 ┌──────────────────────────────┐
                 │       USER'S SKIN IMAGE       │
                 └───────────┬──────────────────-─┘
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
┌──────────────────────────┐   ┌───────────────────────────────┐
│  MODEL 1: HF Classifier  │   │  MODEL 2: LLaMA 4 Scout      │
│  (Anwarkh1/Skin_Cancer)  │   │  (Open-ended Vision Diagnosis)│
│                          │   │                               │
│  • Fixed 9-label set     │   │  • No label restriction       │
│  • Fine-tuned on skin    │   │  • Can identify rare diseases │
│    cancer dataset        │   │  • Visual description         │
│  • Confidence scores     │   │  • Differential diagnoses     │
│  • Cancer probability    │   │  • Severity assessment        │
└────────────┬─────────────┘   └──────────────┬────────────────┘
             │                                │
             └────────────┬───────────────────┘
                          ▼
             ┌───────────────────────────┐
             │  MODEL 3: LLaMA 3.3 70B  │
             │  (Analysis LLM)           │
             │                           │
             │  Compares & synthesizes:  │
             │  • Classifier results     │
             │  • Vision diagnosis       │
             │  • Internet search        │
             │  • Medical knowledge      │
             │  • Patient data           │
             └───────────────────────────┘
```

**Why two models?**
- The **HF Classifier** (`Anwarkh1/Skin_Cancer-Image_Classification`) is a purpose-built model fine-tuned specifically on the HAM10000 skin cancer dataset — it gives accurate, structured predictions with confidence scores for the 9 most common skin conditions.
- **LLaMA 4 Scout** acts as a general-purpose dermatology vision AI — it's not limited to those 9 labels and can identify rare conditions (e.g., Kaposi sarcoma, porokeratosis, mycosis fungoides) that the classifier would never predict.
- The **Analysis LLM** (LLaMA 3.3 70B) then **compares** both outputs. If they agree, confidence is high. If they disagree, the LLM explains both perspectives and weighs which is more likely correct.

---

## 2. Directory Structure

```
skin-disease-detection/
├── run.py                      # Entry point — launches Streamlit app
├── requirements.txt            # Python dependencies
├── test_pipeline.py            # End-to-end test script
├── .env / .env.example         # API keys (Groq, HuggingFace, Tavily)
│
├── frontend/
│   ├── __init__.py
│   └── app.py                  # Streamlit UI (upload, results, chat, PDF download)
│
├── models/
│   ├── __init__.py
│   ├── classifier.py           # SkinClassifier — image classification
│   └── risk_assessment.py      # RiskAssessor — weighted medical scoring algorithm
│
├── agent/
│   ├── __init__.py
│   ├── llm_agent.py            # 3-step vision-enhanced LLM analysis pipeline
│   └── tools.py                # LangChain tools (PubMed, Wikipedia, Tavily, RAG)
│
├── rag/
│   ├── __init__.py
│   ├── medical_knowledge.py    # Curated medical corpus (14+ documents)
│   └── vector_store.py         # FAISS vector store + HuggingFace embeddings
│
├── utils/
│   ├── __init__.py
│   ├── config.py               # Config dataclass — API keys + model settings
│   ├── logger.py               # Structured logging (console + file)
│   └── pdf_report.py           # FPDF2-based PDF report generator
│
├── logs/
│   └── app.log                 # Runtime logs
│
└── test_images/                # Sample test images for testing
```

---

## 3. End-to-End Pipeline Workflow

When a user clicks **"Analyze Image"**, this is the exact sequence that runs:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        USER UPLOADS IMAGE                              │
│               + fills patient info in sidebar (age,                    │
│                 symptoms, duration, family history)                    │
└─────────────────────┬───────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 1: IMAGE CLASSIFICATION  (models/classifier.py)                  │
│                                                                         │
│  Model: Anwarkh1/Skin_Cancer-Image_Classification (HuggingFace)        │
│  Currently using: LLaMA 4 Scout via Groq (fallback — see note below)   │
│                                                                         │
│  • Image resized to 224×224, converted to base64 JPEG                  │
│  • Sent to classification model                                         │
│  • Model returns top 3-5 predictions with confidence scores            │
│  • Labels mapped to cancer risk metadata via LABEL_MAP                  │
│  • Cancer probability calculated (weighted sum of all predictions)      │
│  • Warnings generated if low confidence or high cancer risk             │
│                                                                         │
│  OUTPUT: { label, confidence, cancer_probability, is_cancerous,         │
│            all_predictions, warnings }                                  │
└─────────────────────┬───────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 2: RISK ASSESSMENT  (models/risk_assessment.py)                  │
│                                                                         │
│  No model — pure algorithmic scoring                                    │
│                                                                         │
│  • 4-component weighted scoring:                                        │
│    - Model Score (35%): confidence × disease severity weight            │
│    - Cancer Score (25%): direct cancer probability                      │
│    - Symptom Score (25%): bleeding, size change, itching, pain, etc.   │
│    - History Score (15%): age, gender, family history                   │
│  • Final score → Low / Moderate / High / Critical                       │
│  • Human-readable reasoning text generated                              │
│                                                                         │
│  OUTPUT: { risk_level, risk_score, reasoning, components }              │
└─────────────────────┬───────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 3: VISION-ENHANCED LLM ANALYSIS  (agent/llm_agent.py)           │
│                                                                         │
│  3a: INDEPENDENT VISION DIAGNOSIS                                      │
│      Model: LLaMA 4 Scout 17B via Groq Vision API                     │
│      • Same image (resized to 512×512) sent to vision model            │
│      • Open-ended prompt — can diagnose ANY condition                  │
│      • Returns: diagnosis, confidence, severity, differential list      │
│                                                                         │
│  3b: INTERNET SEARCH                                                   │
│      API: Tavily Search                                                 │
│      • Searches for the vision model's diagnosis                       │
│      • Returns top 3 results + AI summary                             │
│                                                                         │
│  3c: MEDICAL KNOWLEDGE LOOKUP (RAG)                                    │
│      • Keyword search of curated corpus (14 documents)                 │
│      • Matches both classifier AND vision model diagnoses              │
│      • Always includes prevention + guidance docs                      │
│                                                                         │
│  3d: COMBINED ANALYSIS                                                 │
│      Model: LLaMA 3.3 70B Versatile via Groq (LangChain ChatGroq)     │
│      • Gets ALL data: classifier + vision + search + RAG + patient     │
│      • COMPARES classifier vs vision model results                     │
│      • If they disagree → explains both, weighs evidence               │
│      • If vision finds rare condition → prioritizes over classifier    │
│      • Generates comprehensive markdown analysis                       │
│                                                                         │
│  OUTPUT: Markdown-formatted medical analysis text                       │
└─────────────────────┬───────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 4: RESULTS + INTERACTION  (frontend/app.py)                      │
│                                                                         │
│  • Metric cards (condition, confidence, cancer prob, risk level)        │
│  • Confidence bars for all predictions                                  │
│  • Color-coded risk assessment card                                     │
│  • Full AI analysis (expandable)                                        │
│  • PDF report download (utils/pdf_report.py)                            │
│  • Follow-up chat with the LLM about the analysis                      │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. AI Models — What's Used & Where

| # | Model | Provider | File | Purpose |
|:-:|-------|----------|------|---------|
| 1 | **Anwarkh1/Skin_Cancer-Image_Classification** | HuggingFace Inference API | `models/classifier.py` | **Primary image classifier** — fine-tuned on skin cancer dataset, classifies into 9 conditions with confidence scores |
| 2 | **LLaMA 4 Scout 17B** (`meta-llama/llama-4-scout-17b-16e-instruct`) | Groq Vision API | `models/classifier.py` (fallback) + `agent/llm_agent.py` (vision diagnosis) | **Vision model** — used for classification fallback AND independent open-ended diagnosis |
| 3 | **LLaMA 3.3 70B Versatile** (`llama-3.3-70b-versatile`) | Groq API via LangChain `ChatGroq` | `agent/llm_agent.py` | **Analysis LLM** — synthesizes all data sources into comprehensive medical analysis + handles follow-up chat |
| 4 | **all-MiniLM-L6-v2** (`sentence-transformers/all-MiniLM-L6-v2`) | HuggingFace Inference API | `rag/vector_store.py` | **Text embeddings** — embeds medical knowledge docs for FAISS similarity search |

### Model Roles Explained

**Model 1 — HF Classifier (Anwarkh1/Skin_Cancer-Image_Classification):**
- A **fine-tuned image classification model** specifically trained on the HAM10000 skin cancer dataset
- Defined in `config.py` as `classification_model`
- Produces structured predictions: `[{label: "melanoma", confidence: 0.82}, ...]`
- Constrained to 9 known skin conditions — very accurate for common cases
- Cancer probability and risk metadata are derived from its output

**Model 2 — LLaMA 4 Scout (Vision):**
- A **multimodal vision-language model** from Meta, accessed via Groq's API
- Used for **open-ended diagnosis** — the prompt does NOT restrict it to fixed labels
- Can identify **rare conditions** like Epidermodysplasia verruciformis, Kaposi sarcoma, cutaneous horns, etc.
- Also provides: visual description, severity rating, differential diagnoses, and a search query

**Model 3 — LLaMA 3.3 70B (Analysis):**
- A high-quality **text-only LLM** for reasoning and synthesis
- Receives the outputs of BOTH models + internet search + RAG + patient data
- Its key job: **compare** the classifier and vision model outputs, resolve disagreements, and produce a safe, comprehensive analysis
- Runs at `temperature=0.1` for factual consistency

**Model 4 — all-MiniLM-L6-v2 (Embeddings):**
- A lightweight **sentence-transformer** for embedding medical text chunks
- Used to build the FAISS vector store for semantic search over the medical knowledge corpus
- Runs via HuggingFace cloud API — no local model download

---

## 5. Module Deep Dive

### 5.1 Image Classifier

**File:** `models/classifier.py`  
**Class:** `SkinClassifier`  
**Config:** `config.classification_model` = `Anwarkh1/Skin_Cancer-Image_Classification`

**What it does:**
1. Takes a PIL image, resizes to **224×224**, converts to base64 JPEG
2. Sends to the classification model
3. Model returns top 3-5 predictions with confidence scores (must sum to ~1.0)
4. Labels are mapped through `LABEL_MAP` for display names and cancer risk metadata
5. Cancer probability is calculated as a weighted sum across all predictions
6. Warnings are generated for low confidence or high cancer probability

**The `LABEL_MAP` (9 conditions + HAM10000 aliases):**

| Condition | Cancerous? | Cancer Risk Weight | Aliases |
|-----------|:----------:|:------------------:|---------|
| Melanoma | ✅ | 0.95 | `mel` |
| Squamous Cell Carcinoma | ✅ | 0.90 | — |
| Basal Cell Carcinoma | ✅ | 0.85 | `bcc` |
| Actinic Keratosis | ❌ (precancer) | 0.35 | `akiec` |
| Pigmented Benign Keratosis | ❌ | 0.05 | `bkl` |
| Melanocytic Nevus (Mole) | ❌ | 0.05 | `nv`, `nevus` |
| Dermatofibroma | ❌ | 0.05 | `df` |
| Vascular Lesion | ❌ | 0.05 | `vasc` |
| Seborrheic Keratosis | ❌ | 0.03 | — |

**Cancer Probability Algorithm:**
```
cancer_prob = (top_label_cancer_risk × top_confidence)
            + Σ (secondary_cancer_risk × secondary_confidence × 0.5)
cancer_prob = min(cancer_prob, 1.0)
```

**Warning Thresholds (configurable):**
- Confidence < `0.60` → "Low certainty" warning
- Cancer probability > `0.70` → "URGENT: High cancer probability" warning

---

### 5.2 Risk Assessment Engine

**File:** `models/risk_assessment.py`  
**Class:** `RiskAssessor`

A **pure algorithmic** module (no AI model) — computes risk using a weighted scoring formula.

**4-Component Weighted Formula:**

```
final_score = 0.35 × model_score      (Model confidence × disease severity)
            + 0.25 × cancer_score     (Direct cancer probability)
            + 0.25 × symptom_score    (ABCDE-inspired patient symptoms)
            + 0.15 × history_score    (Demographics + family history)
```

**Symptom Score Breakdown (ABCDE-inspired):**

| Factor | Points | Rationale |
|--------|:------:|-----------|
| Bleeding | +0.30 | Strong melanoma indicator |
| Size change | +0.25 | "Evolving" in ABCDE rule |
| Pain (scaled) | +0.20 × (pain/10) | Severity indicator |
| Itching | +0.10 | Common symptom |
| Duration | +0.02 to +0.15 | Longer = more concern |

**History Score Breakdown:**

| Factor | Points |
|--------|:------:|
| Family history of skin cancer | +0.40 |
| Age > 50 | +0.20 |
| Age > 65 (additional) | +0.10 |
| Male gender | +0.10 |

**Risk Level Thresholds:**

| Score | Level |
|:-----:|:-----:|
| ≥ 0.70 | 🔴 Critical |
| 0.45–0.69 | 🟠 High |
| 0.25–0.44 | 🟡 Moderate |
| < 0.25 | 🟢 Low |

---

### 5.3 Vision-Enhanced LLM Agent

**File:** `agent/llm_agent.py`

This is the **core intelligence module** — runs after classification and risk assessment. It has 4 sub-steps:

#### Step 3a: Independent Vision Diagnosis (`_vision_diagnosis`)
- **Model:** LLaMA 4 Scout 17B via Groq Vision API
- Image resized to **512×512** (larger than classifier's 224×224 for more detail)
- Uses an **open-ended prompt** — explicitly lists rare conditions to consider
- Returns structured JSON: `diagnosis`, `confidence`, `severity`, `is_rare`, `differential_diagnoses`, `visual_description`, `reasoning`, `search_query`

#### Step 3b: Internet Search (`_search_condition`)
- **API:** Tavily
- Uses the vision model's `search_query` to find information
- Returns AI-generated summary + top 3 web results with URLs
- Skipped if Tavily API key not configured

#### Step 3c: Medical Knowledge Lookup (`_fetch_medical_context`)
- **Keyword-based** search of the curated 14-document corpus
- Searches for BOTH the classifier's prediction AND the vision model's diagnosis
- Always includes "Skin Cancer Prevention" and "When to See a Doctor" documents

#### Step 3d: Combined Analysis (`analyze_prediction`)
- **Model:** LLaMA 3.3 70B Versatile via LangChain `ChatGroq`
- Builds a mega-prompt combining **ALL** data sources:
  - Classifier predictions + confidence + cancer probability + warnings
  - Vision AI independent diagnosis + differential diagnoses
  - Internet search results
  - Medical knowledge base content
  - Patient info (age, gender, symptoms, duration, family history)
  - Risk assessment score and reasoning
- **System prompt enforces critical rules:**
  - Compare classifier vs vision model — if they disagree, explain both
  - For rare conditions, prioritize vision model over fixed-label classifier
  - Never claim definitive diagnosis (use "suggests", "may indicate")
  - Always recommend dermatologist consultation
  - If cancer risk is high, URGENTLY recommend consultation
- **Output:** 7-section markdown analysis (Vision Assessment, Comparison, Condition Overview, Risk, Observations, Internet Research, Recommended Actions)

#### Follow-Up Chat (`chat_followup`)
- Takes user question + previous analysis context
- Sends to the same LLM with the same safety-focused system prompt

---

### 5.4 Agent Tools

**File:** `agent/tools.py`

LangChain `@tool`-decorated functions for extended information retrieval:

| Tool | API/Source | Purpose |
|------|-----------|---------|
| `search_medical_knowledge` | FAISS vector store (internal) | Primary RAG search over curated medical corpus |
| `search_pubmed` | PubMed E-Utilities API | Medical research articles — titles + abstracts (top 3) |
| `search_wikipedia` | Wikipedia API | General medical context and definitions |
| `search_web_medical` | Tavily API | Recent web info about skin conditions (optional) |

Created via `create_tools(config)` factory. `search_web_medical` only available if Tavily API key is set.

---

### 5.5 RAG System

#### Medical Knowledge Corpus (`rag/medical_knowledge.py`)

**14 curated documents** from trusted medical sources, embedded directly in code:

| Topic | Source | Category |
|-------|--------|----------|
| Melanoma | NIH/NCI | Cancer |
| Melanoma Treatment | NIH/NCI | Cancer |
| Basal Cell Carcinoma | Mayo Clinic | Cancer |
| Squamous Cell Carcinoma | Mayo Clinic | Cancer |
| Actinic Keratosis | WHO | Precancerous |
| Dermatofibroma | NIH/NIAMS | Benign |
| Melanocytic Nevus | NIH/NCI | Benign |
| Seborrheic Keratosis | Mayo Clinic | Benign |
| Vascular Lesions | NIH/NHLBI | Benign |
| Pigmented Benign Keratosis | WHO | Benign |
| Skin Cancer Prevention | WHO | Prevention |
| ABCDE Rule | American Cancer Society | Diagnosis |
| Psoriasis | NIH/NIAMS | Chronic |
| Eczema | NIH/NIAID | Chronic |
| When to See a Doctor | AAD | Guidance |

#### Vector Store (`rag/vector_store.py`)

- **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` via HuggingFace Inference API (cloud — no local download)
- **Vector Database:** FAISS (in-memory)
- Documents converted to LangChain `Document` objects, then embedded and indexed
- Module-level cache — built once, reused across requests
- Retriever returns top-k results (default `k=3`, configurable)

---

### 5.6 Frontend

**File:** `frontend/app.py`  
**Framework:** Streamlit

**Sidebar:**
- API key status indicators (✅/❌) — auto-loaded from `.env`
- Manual override expander for API keys
- Patient information form: age, gender, duration, symptoms (itching, bleeding, size change, family history), pain level

**Main Area Flow:**
1. Image upload (JPG, PNG, BMP, WebP)
2. Image preview
3. "Analyze Image" button → runs full pipeline
4. Results display:
   - 4 metric cards (Predicted Condition, Model Confidence, Cancer Probability, Risk Level)
   - Confidence progress bars for all predictions
   - Color-coded risk assessment card (green/yellow/orange/red)
   - Expandable AI analysis section
   - PDF report download button
   - Follow-up chat interface

**Styling:** Custom CSS with Inter font, glassmorphism gradients, pulse animation for critical warnings.

---

### 5.7 Utilities

#### Config (`utils/config.py`)
```python
@dataclass
class Config:
    groq_api_key          # Groq API (vision + LLM)
    huggingface_api_key   # HuggingFace (classifier + embeddings)
    tavily_api_key        # Tavily (web search)

    classification_model = "Anwarkh1/Skin_Cancer-Image_Classification"  # HF classifier
    embedding_model      = "sentence-transformers/all-MiniLM-L6-v2"     # RAG embeddings
    llm_model            = "llama-3.3-70b-versatile"                    # Analysis LLM
    llm_temperature      = 0.1

    rag_top_k                 = 3
    low_confidence_threshold  = 0.60
    high_cancer_threshold     = 0.70
```

Supports loading from `.env` or manual override via Streamlit sidebar.

#### Logger (`utils/logger.py`)
- Dual output: console (INFO) + file (DEBUG → `logs/app.log`)
- Format: `timestamp | LEVEL | module | message`

#### PDF Report (`utils/pdf_report.py`)
- Uses **FPDF2** with Helvetica font
- Sections: AI Prediction, Risk Assessment, Patient Info, AI Analysis
- Header with blue accent line, footer with disclaimer + page numbers
- Returns PDF as bytes for Streamlit download

---

## 6. Algorithms & Scoring Logic

### 6.1 Cancer Probability (from Classifier)
```
cancer_prob = top_cancer_risk × top_confidence
            + Σ(secondary_cancer_risk × secondary_confidence × 0.5)
cancer_prob = min(cancer_prob, 1.0)
```

### 6.2 Risk Assessment (4-component weighted score)
```
final = 0.35 × (confidence × disease_risk_weight)
      + 0.25 × (cancer_probability)
      + 0.25 × (symptom_score)              ← ABCDE-inspired
      + 0.15 × (history_score)              ← demographics
```

### 6.3 Dual-Model Analysis Pipeline
```
Image → HF Classifier (9 labels) ──────┐
  │                                     │
  └→ LLaMA 4 Scout (open-ended) ──┐    │
                                   │    │
     Tavily Search ←── from Scout  │    │
                                   │    │
     Medical RAG ←── both labels   │    │
                                   │    │
     ALL combined ─────────────────┴────┘
           │
           ▼
     LLaMA 3.3 70B → Final Analysis
```

---

## 7. API Keys & Configuration

| Key | Required | Used For |
|-----|:--------:|----------|
| `GROQ_API_KEY` | ✅ Yes | LLaMA 4 Scout (vision) + LLaMA 3.3 70B (analysis LLM) |
| `HUGGINGFACE_API_KEY` | ✅ Yes | `Anwarkh1/Skin_Cancer-Image_Classification` (classifier) + `all-MiniLM-L6-v2` (embeddings) |
| `TAVILY_API_KEY` | ⚡ Optional | Internet search for condition info (enhances rare condition analysis) |

**Setup:** Copy `.env.example` to `.env` and add your API keys.

---

## 8. Data Flow Diagram

```
                    ┌──────────────┐
                    │  User Image  │
                    │ + Patient    │
                    │   Metadata   │
                    └──────┬───────┘
                           │
              ┌────────────┼────────────────┐
              ▼            │                ▼
     ┌────────────────┐    │     ┌────────────────────┐
     │ HF Classifier  │    │     │ LLaMA 4 Scout      │
     │ (Anwarkh1/     │    │     │ (Open-ended Vision  │
     │  Skin_Cancer)  │    │     │  Diagnosis)         │
     │                │    │     │                     │
     │ 9 labels +     │    │     │ • Any condition     │
     │ confidence     │    │     │ • Differentials     │
     │ scores         │    │     │ • Severity          │
     └───────┬────────┘    │     └──────────┬──────────┘
             │             │                │
             │             │                ▼
             │             │     ┌────────────────────┐
             │             │     │ Tavily Search      │
             │             │     │ (Internet info)     │
             │             │     └──────────┬──────────┘
             │             │                │
             ▼             │                │
     ┌────────────────┐    │     ┌──────────▼──────────┐
     │ Risk Assessor  │    │     │ Medical RAG         │
     │ (Weighted      │    │     │ (FAISS + HF         │
     │  Algorithm)    │    │     │  Embeddings)        │
     └───────┬────────┘    │     └──────────┬──────────┘
             │             │                │
             └─────────────┼────────────────┘
                           │
                           ▼
              ┌──────────────────────┐
              │  LLaMA 3.3 70B      │
              │  Combined Analysis   │
              │                      │
              │  Compares both       │
              │  models, synthesizes │
              │  all data sources    │
              └──────────┬───────────┘
                         │
              ┌──────────┼──────────────┐
              ▼          ▼              ▼
        ┌──────────┐ ┌──────────┐ ┌─────────────┐
        │ Streamlit│ │ PDF      │ │ Follow-up   │
        │ Results  │ │ Report   │ │ Chat        │
        └──────────┘ └──────────┘ └─────────────┘
```

---

## 9. Supported Skin Conditions

### From the HF Classifier (9 fixed labels)

| # | Condition | Type | Cancer Risk |
|:-:|-----------|------|:-----------:|
| 1 | Melanoma | Malignant | 0.95 |
| 2 | Squamous Cell Carcinoma | Malignant | 0.90 |
| 3 | Basal Cell Carcinoma | Malignant | 0.85 |
| 4 | Actinic Keratosis | Precancerous | 0.35 |
| 5 | Pigmented Benign Keratosis | Benign | 0.05 |
| 6 | Melanocytic Nevus (Mole) | Benign | 0.05 |
| 7 | Dermatofibroma | Benign | 0.05 |
| 8 | Vascular Lesion | Benign | 0.05 |
| 9 | Seborrheic Keratosis | Benign | 0.03 |

### From LLaMA 4 Scout (unlimited — examples of rare conditions it can detect)

- Epidermodysplasia verruciformis
- Cutaneous horns
- Porokeratosis
- Mycosis fungoides
- Kaposi sarcoma
- Harlequin ichthyosis
- Tinea corporis (as seen in logs)
- Target lesion
- And any other dermatological condition

---

## 10. Architecture Evolution Note

The logs reveal the system's evolution:

1. **Original Design:** Used `Anwarkh1/Skin_Cancer-Image_Classification` via HuggingFace Inference API as the primary classifier.
2. **HF Model Deprecation:** The model returned `410 Gone` errors — it was deprecated by the HuggingFace inference provider.
3. **Groq Fallback:** The classifier was updated to use LLaMA 4 Scout via Groq Vision API as a replacement, with a structured prompt constraining it to the same 9 labels.
4. **Vision Enhancement:** The independent vision diagnosis pipeline was added (Step 3a), using LLaMA 4 Scout with an open-ended prompt for rare condition detection.

The `config.classification_model` field still holds `"Anwarkh1/Skin_Cancer-Image_Classification"`, reflecting the intended design. The current code uses Groq Vision as a fallback implementation.

---

## 11. Safety & Disclaimer

Multiple layers of medical safety are enforced:

1. **Prompt-level:** LLM must use hedging language, never claim definitive diagnoses
2. **Automated warnings:** Triggered for low confidence or high cancer probability
3. **UI disclaimers:** Footer on every Streamlit page + every PDF page
4. **Critical alerts:** Pulsing red banner for high-risk predictions
5. **Dual-model validation:** Two independent AI opinions reduce error risk
6. **Always recommends:** Professional dermatologist consultation

> ⚕️ **This system is for educational purposes only.** It does not provide medical advice, diagnosis, or treatment recommendations. Always consult a qualified healthcare provider.

---

*Generated on: 2026-02-13 • AI Skin Disease Detection System*

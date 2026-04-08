# SupportSage Pro

### Multi-Tenant Agentic RAG — Private Knowledge, Zero Compromise

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)](https://python.org)
[![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-orange?style=flat-square)](https://www.trychroma.com/)
[![BGE-M3](https://img.shields.io/badge/Embeddings-BAAI%2FBGE--M3-green?style=flat-square)](https://huggingface.co/BAAI/bge-m3)
[![Groq](https://img.shields.io/badge/LLM-Groq%20%7C%20LLaMA--3.3--70B-purple?style=flat-square)](https://groq.com)
[![Gradio](https://img.shields.io/badge/UI-Gradio-red?style=flat-square)](https://gradio.app)
[![CUDA](https://img.shields.io/badge/Hardware-CUDA%20Accelerated-76B900?style=flat-square&logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)

---

## Overview

SupportSage Pro is a production-grade, multi-tenant Retrieval-Augmented Generation (RAG) system built for real-world customer support scenarios. It allows any organization — or any individual — to upload their private documentation and instantly deploy a fully isolated, intelligent support agent over that data.

Every user gets their own secured vector database. Every answer is grounded in source-cited documents. Every response passes through an agentic reflection loop before it reaches the user.

This is not a tutorial project. This is a deployable system.

---

## The Problem It Solves

Generic LLMs hallucinate. Cloud embedding APIs leak your private data. Shared vector databases mean one user's documents can bleed into another's context.

SupportSage Pro addresses all three:

- **Hallucination** → Agentic reflection loop scores every answer before delivery
- **Data Privacy** → All embeddings run locally via BAAI/BGE-M3 on-device; no document ever leaves your machine
- **Data Isolation** → Every user gets a cryptographically authenticated, fully separate ChromaDB collection

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER UPLOADS DOCS                        │
└────────────────────────────┬────────────────────────────────────┘
                             │
                    ┌────────▼────────┐
                    │   chunking.py   │
                    │                 │
                    │  Semantic Split │  ← NLTK sentence tokenization
                    │  (Parent Docs)  │  ← Cosine similarity grouping
                    │                 │  ← 15th percentile threshold
                    │  Child Chunks   │  ← RecursiveCharacterSplitter
                    │  (400 tokens)   │     on top of semantic parents
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  embedding.py   │
                    │                 │
                    │  BAAI/BGE-M3    │  ← 1024-dim dense embeddings
                    │  (Local, CUDA)  │  ← Pulse-mode: 1 doc at a time
                    │                 │     to prevent GPU PSU spikes
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  database.py    │
                    │                 │
                    │  ChromaDB       │  ← {user_id}_parents
                    │  Multi-Tenant   │  ← {user_id}_children
                    │                 │  ← {user_id}_memory
                    │  SHA-256 Auth   │  ← Password-gated access
                    └────────┬────────┘
                             │
              ┌──────────────▼──────────────┐
              │          agent.py           │
              │                             │
              │  1. Intent Router (LLM)     │  ← STRIPE_DOCS vs CHAT
              │  2. Long-Term Memory Check  │  ← Semantic similarity < 0.15
              │  3. Vector Retrieval        │  ← Top-K child chunks
              │  4. Parent Expansion        │  ← Fetch full parent context
              │  5. Cross-Encoder Re-Rank   │  ← ms-marco-MiniLM-L-6-v2
              │  6. Response Generation     │  ← LLaMA-3.3-70B via Groq
              │  7. Reflection Loop         │  ← Score/correct up to 2x
              │  8. Memory Storage          │  ← Save high-quality answers
              └──────────────┬──────────────┘
                             │
                    ┌────────▼────────┐
                    │    app.py       │
                    │  Gradio UI      │
                    │  Data Forge Tab │
                    │  Chat Tab       │
                    └─────────────────┘
```

---

## Key Features

### Semantic Parent-Child Chunking
Documents are not split by fixed token counts. Instead, sentences are embedded and grouped by semantic similarity — topic boundaries are detected using cosine distance at the 15th percentile threshold. These semantic blocks become **parent documents**. Each parent is then further split into smaller **child chunks** (400 tokens) for precision retrieval. At query time, children are retrieved, but the full parent is passed to the LLM — giving the model rich context while keeping retrieval precise.

### Fully Local Embeddings
BAAI/BGE-M3 runs entirely on-device via CUDA. No document content is ever sent to an external embedding API. This is critical for enterprise use cases where data cannot leave the local environment. The embedding engine uses a "pulse-mode" loop — encoding one document at a time with controlled GPU cache clearing — to ensure stable operation on consumer-grade hardware.

### Multi-Tenant Isolated Vector Storage
Each user registers with a unique ID and password. Their data is stored in three isolated ChromaDB collections. Authentication is enforced at every access point. No query from User A can ever touch User B's data. Users can permanently delete their own database at any time via the Data Forge tab.

### Agentic Intent Routing
Before touching the vector database, the agent uses the LLM to classify the query. Conversational queries (greetings, general questions) are handled directly. Only queries that require private document knowledge trigger the full retrieval pipeline. This reduces latency and unnecessary vector database calls.

### Cross-Encoder Re-Ranking
After retrieving the top-K candidates from ChromaDB, results are passed through a cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`). Unlike bi-encoder similarity (which compares embeddings independently), a cross-encoder reads the query and each document *together*, producing a more accurate relevance score. The top 5 re-ranked results are passed to the LLM.

### Source-Cited Responses
Every factual claim in the agent's response is tied back to the source document. The retrieval pipeline embeds filename metadata into the context block, and the LLM is instructed to cite sources explicitly: *"According to stripe_api_v3.pdf..."*. Answers without grounding are not acceptable.

### Agentic Reflection Loop
After generation, a second LLM call scores the answer from 1–10 against the retrieved context. If the score is below 9, the agent corrects the answer and tries again — up to 2 correction cycles. High-scoring answers are automatically stored in the long-term memory collection for future reuse.

### Long-Term Memory
Verified, high-quality answers are stored as embeddings in the user's `_memory` collection. On future queries, the system checks memory first. If a semantically similar query has been answered before (distance < 0.15), the stored answer is returned immediately — no retrieval, no generation, no latency.

---

## Project Structure

```
supportsage-pro/
│
├── chunking.py          # Semantic parent-child document processing
├── embedding.py         # Local BGE-M3 embedding engine (CUDA)
├── database.py          # ChromaDB multi-tenant manager + auth
├── agent.py             # Full agentic pipeline (routing → retrieval → reflection)
├── app.py               # Gradio web UI
│
├── requirements.txt     # Dependencies
├── .env                 # GROQ_API_KEY (not committed)
└── README.md
```

---

## Tech Stack

| Component | Technology | Reason |
|-----------|-----------|--------|
| Embeddings | BAAI/BGE-M3 (local) | Privacy-first; no external API calls; 1024-dim multilingual |
| Vector Database | ChromaDB (persistent) | Lightweight, local, supports per-collection isolation |
| LLM | LLaMA-3.3-70B via Groq | High-quality reasoning at low latency |
| Re-Ranker | ms-marco-MiniLM-L-6-v2 | Fast cross-encoder for precision post-retrieval scoring |
| Sentence Splitting | NLTK punkt tokenizer | Reliable sentence boundaries before semantic grouping |
| UI | Gradio | Rapid, clean interface; Tunnel-deployable |
| Hardware Target | NVIDIA RTX 3060 Ti | CUDA acceleration for embedding and re-ranking |

---

## Setup & Installation

**1. Clone the repository**
```bash
git clone https://github.com/yourusername/supportsage-pro.git
cd supportsage-pro
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Configure environment**
```bash
# Create a .env file in the project root
GROQ_API_KEY=your_groq_api_key_here
```

**4. Run the application**
```bash
python app.py
```

The Gradio interface will launch at `http://localhost:7860`. Use `share=True` in `app.py` to get a public tunnel URL.

---

## Usage

### Step 1 — Build Your Knowledge Base
1. Open the **Data Forge** tab
2. Enter a unique User ID and password
3. Upload your documents (PDF, TXT, MD, PY, CSV)
4. Click **Build Vector Intelligence**
5. Wait for the pipeline to complete — your isolated database is ready

### Step 2 — Query Your Agent
1. Open the **Intelligence Console** tab
2. Enter your User ID and password
3. Ask questions about your documents
4. The agent will retrieve, re-rank, generate, and cite its answer

### Step 3 — Delete Your Data
If you need to permanently remove your database, use the **Delete My Database** button in the Data Forge tab. This requires your ID and password and is irreversible.

---

## Supported Document Types

| Extension | Handler |
|-----------|---------|
| `.pdf` | PyPDFLoader |
| `.txt` | TextLoader |
| `.md` | TextLoader |
| `.py` | PythonLoader |
| `.csv` | TextLoader (with column hint injection) |

---

## Performance Notes

- Embedding runs in **pulse-mode** (one document at a time) to prevent voltage spikes on consumer PSUs during CUDA inference
- The reflection loop adds up to 2 additional LLM calls per query on the `STRIPE_DOCS` path — average added latency is 1–3 seconds on Groq
- Long-term memory lookups happen before all other processing and add near-zero latency (single vector query)
- For large document sets (500+ pages), initial ingestion may take several minutes depending on GPU speed

---

## What Makes This Different

Most RAG demonstrations are single-tenant, use cloud embeddings, and stop at basic retrieval. SupportSage Pro was built to answer a harder question: *what does a RAG system look like when it needs to actually work in production?*

The answer involved multi-tenancy with real auth, fully local embedding for data privacy, a re-ranking stage for precision, source citations for trust, and a reflection loop for accuracy. Each of these decisions exists because a real support system fails without them.

---

## Author

Built by **Zain** — running on an RTX 3060 Ti, refusing to use cloud embeddings.

---

## License

MIT License — see `LICENSE` for details.

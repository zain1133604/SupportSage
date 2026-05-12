# SupportSage Pro
### Multi-Tenant Agentic RAG — Private Knowledge, Zero Compromise

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Persistent-green?style=flat-square)
![BGE-M3](https://img.shields.io/badge/BGE--M3-Local%20Embeddings-orange?style=flat-square)
![Groq](https://img.shields.io/badge/Groq-LLaMA%203.3%2070B-purple?style=flat-square)
![Gradio](https://img.shields.io/badge/Gradio-UI-ff7043?style=flat-square)
![LangSmith](https://img.shields.io/badge/LangSmith-Traced-yellow?style=flat-square)
![MySQL](https://img.shields.io/badge/MySQL-Order%20DB-00758f?style=flat-square)
![CUDA](https://img.shields.io/badge/CUDA-RTX%203060%20Ti-76b900?style=flat-square)
![SQLite](https://img.shields.io/badge/SQLite-Audit%20Store-003b57?style=flat-square)
![CMMC](https://img.shields.io/badge/CMMC-Compliance-red?style=flat-square)

---

## Overview

SupportSage Pro is a production-grade, multi-tenant Agentic RAG system built for real-world customer support scenarios. It goes far beyond retrieval — it understands intent, takes live actions against a real database, handles complaints with automated escalation, and reflects on its own answers before delivering them.

Any organization or individual can upload their private documentation and instantly deploy a fully isolated, intelligent support agent over that data. The agent can also process live order actions (cancel, track, refund, modify), escalate complaints via email, and accumulate knowledge over time through long-term memory.

Every user gets their own secured vector database. Every answer is grounded in source-cited documents. Every response passes through an agentic reflection loop. Every agent call is traced and observable via LangSmith. **Every action is persisted to an immutable audit store — survives restarts and queryable by auditors without replaying the agent.**

**This is not a tutorial project. This is a deployable system.**

---

## The Problem It Solves

| Problem | Solution |
|---|---|
| Generic LLMs hallucinate | Agentic reflection loop scores every answer before delivery |
| Cloud embedding APIs leak private data | All embeddings run locally via BAAI/BGE-M3 on-device — zero data leaves your machine |
| Shared vector DBs bleed context across users | Every user gets a cryptographically isolated ChromaDB collection |
| RAG systems can't take real actions | Live MySQL integration handles order cancellations, refunds, address changes, modifications |
| Complaints fall through the cracks | Automated SMTP escalation emails routed to support team on complaint detection |
| No production visibility | Full LangSmith tracing on every agent call — reasoning, latency, chain execution |
| Audit trails only live in memory | Persistent SQLite audit store — every event written on commit, survives restarts |
| No way to undo agent actions | Every state-changing action stores exact rollback SQL — retrievable anytime without touching the agent |
| AI maps wrong evidence to compliance controls | Compliance agent maps uploaded docs to CMMC, NIST, SOC 2 with source provenance and confidence scores |
| Human has no control over AI decisions | Real human-in-the-loop approval — PENDING actions wait for human Approve/Reject before executing |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         USER UPLOADS DOCS                           │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                      ┌────────▼────────┐
                      │   chunking.py   │
                      │                 │
                      │ Semantic Split  │  ← NLTK sentence tokenization
                      │ (Parent Docs)   │  ← Cosine similarity grouping
                      │                 │  ← 15th percentile threshold
                      │ Child Chunks    │  ← RecursiveCharacterSplitter
                      │ (400 tokens)    │     on semantic parents
                      └────────┬────────┘
                               │
                      ┌────────▼────────┐
                      │  embedding.py   │
                      │                 │
                      │  BAAI/BGE-M3    │  ← 1024-dim dense embeddings
                      │  (Local, CUDA)  │  ← Pulse-mode: 1 doc at a time
                      │                 │     prevents GPU PSU spikes
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
         ┌─────────────────────▼──────────────────────┐
         │             agent_traced.py                 │
         │                                             │
         │   @traceable decorator                      │  ← LangSmith span per call
         │   Latency + trace capture                   │  ← Chain execution logged
         │                                             │
         │                agent.py                     │
         │                                             │
         │   1. Intent Router (LLM)                    │  ← CHAT / KNOWLEDGE_QUERY /
         │                                             │     ORDER_ACTION / COMPLAINT /
         │                                             │     UNKNOWN
         │   2. Long-Term Memory Check                 │  ← Semantic similarity < 0.15
         │   3. ORDER_ACTION Handler                   │  ← Live MySQL: cancel / track /
         │                                             │     refund / modify / change address
         │   4. COMPLAINT Handler                      │  ← SMTP escalation email
         │   5. Vector Retrieval                       │  ← Top-K child chunks
         │   6. Parent Expansion                       │  ← Fetch full parent context
         │   7. Cross-Encoder Re-Rank                  │  ← ms-marco-MiniLM-L-6-v2
         │   8. Response Generation                    │  ← LLaMA-3.3-70B via Groq
         │   9. Reflection Loop                        │  ← Score/correct up to 2x
         │  10. Memory Storage                         │  ← Save high-quality answers
         │  11. Audit Logging (_log_audit)             │  ← Every action → audit_db.py
         └─────────────────────┬──────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │     audit_db.py     │
                    │                     │
                    │  Immutable SQLite   │  ← Every event INSERT only — never deleted
                    │  Audit Store        │  ← actor, timestamp, SHA-256 hashes
                    │                     │  ← control_ref, approval_state
                    │  log_event()        │  ← rollback_sql, needs_human_review
                    │  update_approval()  │  ← PENDING → APPROVED / REJECTED
                    │  explain_control()  │  ← "show me why AC.1.001 passed"
                    │  get_rollback_info()│  ← exact SQL to undo any action
                    └──────────┬──────────┘
                               │
         ┌─────────────────────▼──────────────────────┐
         │           compliance_agent.py               │
         │                                             │
         │   _search_evidence()                        │  ← BGE-M3 search per control
         │   _evaluate_control()                       │  ← LLM: SATISFIED/PARTIAL/GAP
         │   run_compliance_check()                    │  ← Full framework loop
         │   summarise()                               │  ← Score calculation
         │                                             │
         │   Frameworks: CMMC / NIST / SOC 2          │  ← 20 controls each
         └─────────────────────┬──────────────────────┘
                               │
                      ┌────────▼────────┐
                      │    app.py       │
                      │  Gradio UI      │
                      │  Data Forge     │  ← Upload + build knowledge base
                      │  Chat Tab       │  ← Full agentic chat interface
                      │  Live Trace     │  ← Real-time session metadata panel
                      │  Compliance Tab │  ← Run CMMC / NIST / SOC 2 checks
                      │  Audit Trail    │  ← Query persistent audit store
                      │  Review Queue   │  ← Human approve / reject panel
                      └─────────────────┘
```

---

## Key Features

### Agentic Intent Routing
Before touching any database, the agent uses the LLM to classify the query into one of five intents: `CHAT`, `KNOWLEDGE_QUERY`, `ORDER_ACTION`, `COMPLAINT`, or `UNKNOWN`. The router extracts entities (order IDs, email addresses, new addresses) and returns structured JSON with a confidence score. Queries below 0.3 confidence fall back to `UNKNOWN`. This eliminates unnecessary vector DB calls and routes each query to the correct execution path.

### Live Order Management (MySQL Integration)
Order action queries trigger a live MySQL pipeline. The agent handles the full lifecycle: tracking, cancellation (with status guards blocking cancellation of shipped orders), address changes (blocked in transit), refund initiation, item modification, and detailed order summaries. All SQL uses parameterized queries. Connections are always released in `finally` blocks. Business rules are enforced at the code level, not delegated to the LLM.

### Automated Complaint Escalation
When the intent router detects a `COMPLAINT`, the system immediately logs the event and sends an SMTP escalation email to the support team containing the user query and all extracted entities. The user receives an empathetic confirmation. No complaint is silently dropped.

### Persistent Immutable Audit Store
Every action the agent takes is written as an immutable record to a SQLite database (`audit_store.db`) that survives server restarts. Each record captures:

| Field | Purpose |
|---|---|
| `actor` | Who triggered the action (user_id) |
| `timestamp_utc` | Exact UTC time of the event |
| `tool_input_hash` | SHA-256 fingerprint of what went in |
| `tool_output_hash` | SHA-256 fingerprint of what came out |
| `control_ref` | Compliance control this event maps to |
| `approval_state` | AUTO / PENDING / APPROVED / REJECTED |
| `rollback_sql` | Exact SQL to undo any state change |
| `needs_human_review` | Flagged for human review queue |

An auditor can call `explain_control("CMMC.AC.1.001")` and get the full evidence trail directly from the database — without replaying the agent at all.

### Rollback Path
Every state-changing order action (cancel, address change, modify) generates and stores the exact rollback SQL before executing. If a decision needs to be undone, an ops team calls `get_rollback_info(event_id)` to retrieve the SQL and run it manually — no agent involvement required. The rollback SQL is stored permanently in the audit record and is visible in the Audit Trail tab of the UI.

### Human-in-the-Loop Approval
Refund requests and low-confidence answers are never executed automatically. They are written as `PENDING` events in the audit store with `needs_human_review=True`. A reviewer opens the **Review Queue** tab, loads all pending actions, copies the Event ID, enters their name, and clicks **Approve** or **Reject**.

Two things happen on every human decision:
1. The original record's `approval_state` is updated from `PENDING` to `APPROVED` or `REJECTED` — the original content is never deleted
2. A brand new `HUMAN_DECISION` event is written to the audit store recording the reviewer's name, their decision, and exactly which event they reviewed — permanently and queryably

### Compliance Agent (CMMC / NIST / SOC 2)
Upload your organization's policy documents and run a full compliance check against any supported framework. For each control the agent:

1. Encodes the control description with BGE-M3 and searches the user's vector DB for relevant evidence
2. Sends the evidence to Groq LLM with a structured prompt requesting: status (SATISFIED / PARTIAL / GAP), confidence score, exact evidence quote, and source document name
3. Persists the full result to the audit store with `control_ref` so it is queryable later

**Supported frameworks:**

| Framework | Controls |
|---|---|
| CMMC | 20 controls (AC, IA, AU, CM, IR, RM, SC, SI, MP, PE) |
| NIST SP 800-171 | 20 controls (3.1 through 3.14) |
| SOC 2 | 20 controls (CC1 through A1) |

### LangSmith Observability
Agent calls are wrapped with a custom `@traceable` decorator via `trace_wrapper.py`. Every query creates a named LangSmith span (`SupportSage-Agent-Call`) capturing the full execution trace — reasoning steps, chain latency, and intermediate outputs. The `langsmith_config.py` module configures tracing at startup, connecting to the `SupportSage-Pro` project automatically.

### Live Logic Trace Panel
The Gradio UI includes a real-time JSON trace panel in the Intelligence Console tab. After every query it displays the session user, GPU hardware, agent memory depth, and last query — giving a live window into system state during a session.

### Semantic Parent-Child Chunking
Documents are not split by fixed token counts. Sentences are embedded and grouped by cosine similarity — topic boundaries are detected at the 15th percentile similarity threshold. These semantic blocks become parent documents. Each parent is further split into 400-token child chunks for precision retrieval. At query time, children are retrieved but the full parent is passed to the LLM, giving rich context while keeping retrieval precise.

### Fully Local Embeddings
BAAI/BGE-M3 runs entirely on-device via CUDA. No document content is ever sent to an external embedding API. The engine uses pulse-mode encoding — one document at a time with controlled GPU cache clearing and sleep intervals — to ensure stable operation on consumer-grade hardware without PSU voltage spikes.

### Multi-Tenant Isolated Vector Storage
Each user registers with a unique ID and password (SHA-256 hashed). Their data lives in three isolated ChromaDB collections: `{user_id}_parents`, `{user_id}_children`, and `{user_id}_memory`. Authentication is enforced at every access point. No query from User A can touch User B's data. Users can permanently delete their entire database at any time via the Data Forge tab.

### Cross-Encoder Re-Ranking
After retrieving top-K candidates from ChromaDB, results pass through `cross-encoder/ms-marco-MiniLM-L-6-v2`. Unlike bi-encoder similarity (which compares embeddings independently), a cross-encoder reads the query and each document together, producing a more accurate relevance score. The top 5 re-ranked results are passed to the LLM.

### Source-Cited Responses
Every factual claim is tied back to its source document. The retrieval pipeline embeds filename metadata into the context block, and the LLM is instructed to cite sources explicitly: `According to stripe_api_v3.pdf...`. Uncited answers are not acceptable by design.

### Agentic Reflection Loop
After generation, a second LLM call scores the answer from 1–10 against the retrieved context. If the score is below 9, the agent corrects and retries — up to 2 correction cycles. Answers scoring 7+ are automatically stored in long-term memory for future reuse.

### Long-Term Memory
Verified high-quality answers are stored as embeddings in the user's `_memory` collection. On future queries, the system checks memory first. If a semantically similar query has been answered before (distance < 0.4), the stored answer is returned immediately — no retrieval, no generation, no latency.

---

## Project Structure

```
supportsage-pro/
│
├── langsmith_config.py      # LangSmith tracing setup (project + API key)
├── trace_wrapper.py         # @traceable decorator for agent observability
├── agent_traced.py          # TracedAgent wrapper — connects tracing to agent
├── chunking.py              # Semantic parent-child document processing
├── embedding.py             # Local BGE-M3 embedding engine (CUDA, pulse-mode)
├── database.py              # ChromaDB multi-tenant manager + SHA-256 auth
├── agent.py                 # Full agentic pipeline (routing → retrieval → reflection)
├── audit_db.py              # Persistent immutable SQLite audit store
├── compliance_agent.py      # Evidence-to-control mapping (CMMC / NIST / SOC 2)
├── compliance_framework.py  # Control definitions for all 3 frameworks
├── app.py                   # Gradio web UI + audit trail + compliance + review queue
│
├── requirements.txt         # All dependencies
├── .env                     # GROQ_API_KEY + LANGCHAIN_API_KEY (never committed)
└── README.md
```

---

## Tech Stack

| Component | Technology | Reason |
|---|---|---|
| Embeddings | BAAI/BGE-M3 (local) | Privacy-first; no external API; 1024-dim multilingual |
| Vector Database | ChromaDB (persistent) | Lightweight, local, per-collection isolation |
| LLM | LLaMA-3.3-70B via Groq | High-quality reasoning at low latency |
| Re-Ranker | ms-marco-MiniLM-L-6-v2 | Fast cross-encoder for precision post-retrieval scoring |
| Order Database | MySQL | Live order lifecycle management with parameterized SQL |
| Audit Store | SQLite (audit_db.py) | Immutable persistent event log — survives restarts |
| Complaint Routing | SMTP (Gmail) | Automated escalation emails to support team |
| Observability | LangSmith | Full agent tracing, latency monitoring, chain visibility |
| Sentence Splitting | NLTK punkt tokenizer | Reliable sentence boundaries before semantic grouping |
| UI | Gradio | Rapid, clean interface; tunnel-deployable |
| Hardware Target | NVIDIA RTX 3060 Ti | CUDA acceleration for embedding and re-ranking |

---

## Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/supportsage-pro.git
cd supportsage-pro
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure environment
```env
# .env
GROQ_API_KEY=your_groq_api_key_here
LANGCHAIN_API_KEY=your_langsmith_api_key_here
```

### 4. Set up MySQL database
```sql
CREATE DATABASE supportsage_db;
USE supportsage_db;

CREATE TABLE orders (
    order_id VARCHAR(50) PRIMARY KEY,
    product_name VARCHAR(255),
    status VARCHAR(100),
    address TEXT
);
```

### 5. Run the application
```bash
python app.py
```

The Gradio interface launches at `http://localhost:7860`. Use `share=True` in `app.py` for a public tunnel URL. Agent traces appear automatically in your LangSmith project dashboard at [smith.langchain.com](https://smith.langchain.com).

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
3. Ask questions about your documents, place order actions, or report issues
4. The agent routes your intent, retrieves, re-ranks, generates, reflects, and cites
5. Watch the **Live Logic Trace** panel update in real time

### Step 3 — Run a Compliance Check
1. Open the **Compliance** tab
2. Enter your User ID and password
3. Select a framework: CMMC, NIST, or SOC 2
4. Click **Run Compliance Check**
5. The agent maps your uploaded documents to every control and returns a full report with status, confidence score, evidence quote, and source document per control

### Step 4 — Review the Audit Trail
1. Open the **Audit Trail** tab
2. Search by Event ID, actor, or compliance control reference
3. Every event is queryable without replaying the agent
4. Call `explain_control("CMMC.AC.1.001")` to see exactly why a control passed

### Step 5 — Approve or Reject Pending Actions
1. Open the **Review Queue** tab
2. Click **Load Review Queue** to see all pending actions
3. Copy the full Event ID from the table
4. Enter your reviewer name
5. Click **Approve** or **Reject**
6. The original record is never deleted — your decision is written as a permanent `HUMAN_DECISION` event in the audit store

### Step 6 — Monitor in LangSmith
1. Open [smith.langchain.com](https://smith.langchain.com) → **SupportSage-Pro** project
2. Every agent call appears as: `SupportSage-Agent-Call`
3. Inspect reasoning paths, chain latency, and intermediate outputs per query

### Step 7 — Delete Your Data
Use the **Delete My Database** button in the Data Forge tab. Requires your ID and password. Irreversible.

---

## Audit Event Types

| Event Type | When It Is Written |
|---|---|
| `ROUTING` | Every query the agent receives |
| `RAG_RETRIEVAL` | Every knowledge base search |
| `SQL_ACTION` | Every order change (cancel, address, modify) |
| `REFLECTION_SCORE` | Every answer quality check |
| `COMPLAINT` | Every complaint detected |
| `COMPLIANCE_CHECK` | Every compliance control evaluated |
| `HUMAN_DECISION` | Every human approval or rejection |

---

## Intent Routing Reference

| Intent | Trigger Examples | Execution Path |
|---|---|---|
| `CHAT` | "Hello", "How are you?" | Direct LLM response, no retrieval |
| `KNOWLEDGE_QUERY` | "What is the refund policy?", "How do I integrate webhooks?" | Vector retrieval → Re-rank → Generate → Reflect |
| `ORDER_ACTION` | "Cancel order 1234", "Track my order", "Change address" | MySQL lookup → Business rule check → Execute |
| `COMPLAINT` | "This is the worst service", "I'm very unhappy" | Log → SMTP escalation email → Empathetic response |
| `UNKNOWN` | Unclear or low-confidence queries | Graceful fallback message |

---

## Order Action Reference

| Action | Business Rules |
|---|---|
| `track_order` | Returns current status for any order |
| `cancel_order` | Blocked if status is `Shipped` or `Delivered` |
| `change_address` | Blocked if status is `Shipped` or `Delivered` |
| `refund_request` | Written as PENDING — waits for human approval before any money moves |
| `modify_order` | Only allowed for `Pending` or `In Cart` status |
| `view_order_details` | Returns product, status, and shipping address |
| `payment_issue` | Redirects to secure portal — card info never accepted in chat |

---

## Supported Document Types

| Extension | Handler |
|---|---|
| `.pdf` | PyPDFLoader |
| `.txt` | TextLoader |
| `.md` | TextLoader |
| `.py` | PythonLoader |
| `.csv` | TextLoader (with column hint injection) |

---

## Performance Notes

- Embedding runs in pulse-mode (one document at a time) to prevent voltage spikes on consumer PSUs during CUDA inference
- The reflection loop adds up to 2 additional LLM calls per `KNOWLEDGE_QUERY` — average added latency is 1–3 seconds on Groq
- Long-term memory lookups happen before all other processing and add near-zero latency (single vector query)
- LangSmith tracing adds negligible overhead — spans are submitted asynchronously and do not block the response path
- MySQL order actions are connection-pooled with a 5-second timeout and always released in `finally` blocks
- Audit store writes are committed immediately on every event — no batching to preserve immutability
- Compliance checks loop through 20 controls per framework — each control makes one embedding query and one LLM call
- For large document sets (500+ pages), initial ingestion may take several minutes depending on GPU speed

---

## What Makes This Different

Most RAG demonstrations are single-tenant, use cloud embeddings, stop at basic retrieval, and have zero production observability. SupportSage Pro answers a harder question: **what does a support system look like when it needs to actually work?**

The answer involved multi-tenancy with real auth, fully local embedding for data privacy, a re-ranking stage for precision, source citations for trust, a reflection loop for accuracy, live database integration for real actions, automated complaint escalation, LangSmith tracing so you can see exactly what the agent is doing on every single call, a persistent immutable audit store so every decision survives restarts and is queryable by auditors, rollback SQL so any action can be undone without touching the agent, human-in-the-loop approval so no money moves automatically, and a compliance agent that maps your documents to CMMC, NIST, and SOC 2 controls with full source provenance.

Each of these decisions exists because a real support system fails without them.

---

## Author

Built by **Zain** — running on an RTX 3060 Ti, refusing to use cloud embeddings, tracing every agent call, actually connecting to the database, and persisting every decision to an audit store that answers "show me why this control passed" without replaying the agent.

---

## License

MIT License — see `LICENSE` for details.

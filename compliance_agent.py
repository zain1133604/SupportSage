"""
compliance_agent.py
--------------------
Maps uploaded documents to compliance framework controls.

For each control it:
  1. Searches the user's vector DB for relevant evidence
  2. Asks the LLM: does this evidence satisfy the control? Yes / Partial / No
  3. Saves the result to the persistent audit store with full provenance
  4. Returns a structured report the UI can display

Works with CMMC, NIST SP 800-171, and SOC 2.
"""

import logging
import uuid
from typing import Dict, List
from groq import Groq
from database import ChromaVectorDB
from embedding import EmbeddingEngine
from audit_db import AuditDatabase
from compliance_framework import FRAMEWORKS
import os
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class ComplianceAgent:

    def __init__(self, db_path: str, user_id: str, password: str,
                 audit_db_path: str = "./audit_store.db"):

        # Auth
        self.db_manager = ChromaVectorDB(persist_dir=db_path)
        self.db_manager.authenticate(user_id, password)

        self.chroma_client = self.db_manager.client
        self.user_id       = user_id
        self.session_id    = uuid.uuid4().hex

        # Collections
        self.child_col  = self.chroma_client.get_collection(f"{user_id}_children")
        self.parent_col = self.chroma_client.get_collection(f"{user_id}_parents")

        # Models
        self.embedder = EmbeddingEngine()
        self.llm      = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model    = "llama-3.3-70b-versatile"

        # Persistent audit store
        self.audit_db = AuditDatabase(db_path=audit_db_path)

        logger.info(f"[ComplianceAgent] Ready for user: {user_id}")

    # ------------------------------------------------------------------
    # STEP 1 — Find evidence for a single control
    # ------------------------------------------------------------------

    def _search_evidence(self, control_description: str, top_k: int = 8) -> List[Dict]:
        """
        Search the user's vector DB for documents relevant to a control.
        Returns a list of {text, source} dicts.
        """
        query_emb = self.embedder.model.encode(
            [control_description], normalize_embeddings=True
        )[0].tolist()

        results = self.child_col.query(
            query_embeddings=[query_emb], n_results=top_k
        )

        if not results or not results["metadatas"] or not results["metadatas"][0]:
            return []

        parent_ids = list(set(
            m["parent_ref"] for m in results["metadatas"][0] if "parent_ref" in m
        ))
        if not parent_ids:
            return []

        parent_data = self.parent_col.get(ids=parent_ids)
        return [
            {"text": doc, "source": meta.get("source", "Unknown")}
            for doc, meta in zip(parent_data["documents"], parent_data["metadatas"])
        ]

    # ------------------------------------------------------------------
    # STEP 2 — Ask LLM if evidence satisfies the control
    # ------------------------------------------------------------------

    def _evaluate_control(self, control: Dict, evidence_docs: List[Dict]) -> Dict:
        """
        Ask the LLM: does the retrieved evidence satisfy this control?
        Returns: { status, confidence, reasoning, evidence_snippet, source }
        """
        if not evidence_docs:
            return {
                "status":           "GAP",
                "confidence":       0.0,
                "reasoning":        "No relevant documents found in the knowledge base.",
                "evidence_snippet": None,
                "source":           None,
            }

        # Build context block
        context = "\n---\n".join(
            f"[Source: {d['source']}]\n{d['text'][:600]}" for d in evidence_docs[:4]
        )

        prompt = f"""You are a compliance auditor.

Control ID   : {control['id']}
Control Name : {control['name']}
Requirement  : {control['description']}

Evidence from the organisation's documents:
{context}

Your task:
1. Decide if the evidence SATISFIES, PARTIALLY satisfies, or shows a GAP for this control.
2. Quote the single most relevant sentence from the evidence (max 30 words).
3. State which source document the quote came from.
4. Give a confidence score 0.0–1.0.
5. Write one sentence of reasoning.

Respond ONLY in this exact JSON format — no markdown, no extra text:
{{
  "status": "SATISFIED | PARTIAL | GAP",
  "confidence": 0.0,
  "reasoning": "one sentence here",
  "evidence_snippet": "quoted sentence here or null",
  "source": "filename here or null"
}}"""

        try:
            response = self.llm.chat.completions.create(
                model       = self.model,
                messages    = [{"role": "user", "content": prompt}],
                temperature = 0.0,
            )
            raw = response.choices[0].message.content.strip()

            # Clean any accidental markdown fences
            import re, json
            raw = re.sub(r"```json|```", "", raw).strip()
            result = json.loads(raw)
            return result

        except Exception as e:
            logger.error(f"LLM evaluation failed for {control['id']}: {e}")
            return {
                "status":           "GAP",
                "confidence":       0.0,
                "reasoning":        f"Evaluation error: {str(e)}",
                "evidence_snippet": None,
                "source":           None,
            }

    # ------------------------------------------------------------------
    # STEP 3 — Run full framework check
    # ------------------------------------------------------------------

    def run_compliance_check(self, framework_name: str) -> List[Dict]:
        """
        Run a full compliance check for the given framework.
        Returns a list of result dicts — one per control.
        Each result is also persisted to the audit store.
        """
        if framework_name not in FRAMEWORKS:
            raise ValueError(f"Unknown framework: {framework_name}. Choose from {list(FRAMEWORKS.keys())}")

        controls = FRAMEWORKS[framework_name]
        results  = []

        logger.info(f"[ComplianceAgent] Starting {framework_name} check — {len(controls)} controls")

        for i, control in enumerate(controls):
            logger.info(f"  Checking {control['id']} ({i+1}/{len(controls)})...")

            # 1. Retrieve evidence
            evidence_docs = self._search_evidence(control["description"])

            # 2. Evaluate
            evaluation = self._evaluate_control(control, evidence_docs)

            # 3. Build full result record
            record = {
                "framework":        framework_name,
                "control_id":       control["id"],
                "control_name":     control["name"],
                "requirement":      control["description"],
                "status":           evaluation.get("status", "GAP"),
                "confidence":       evaluation.get("confidence", 0.0),
                "reasoning":        evaluation.get("reasoning", ""),
                "evidence_snippet": evaluation.get("evidence_snippet"),
                "source":           evaluation.get("source"),
            }

            # 4. Persist to audit store
            self.audit_db.log_event(
                session_id  = self.session_id,
                actor       = self.user_id,
                event_type  = "COMPLIANCE_CHECK",
                details     = record,
                source_file = record["source"],
                tool_input  = {"control_id": control["id"], "description": control["description"]},
                tool_output = {"status": record["status"], "confidence": record["confidence"]},
                control_ref = f"{framework_name}.{control['id']}",
                approval_state = "AUTO",
            )

            results.append(record)

        logger.info(f"[ComplianceAgent] {framework_name} check complete.")
        return results

    # ------------------------------------------------------------------
    # STEP 4 — Summary statistics
    # ------------------------------------------------------------------

    @staticmethod
    def summarise(results: List[Dict]) -> Dict:
        total     = len(results)
        satisfied = sum(1 for r in results if r["status"] == "SATISFIED")
        partial   = sum(1 for r in results if r["status"] == "PARTIAL")
        gaps      = sum(1 for r in results if r["status"] == "GAP")
        score     = round((satisfied + partial * 0.5) / total * 100, 1) if total else 0

        return {
            "total":     total,
            "satisfied": satisfied,
            "partial":   partial,
            "gaps":      gaps,
            "score":     score,
        }
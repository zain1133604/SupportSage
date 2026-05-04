import os
import logging
import uuid
from typing import List, Dict, Any, Tuple, Optional
from groq import Groq
import chromadb
from embedding import EmbeddingEngine
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder
from database import ChromaVectorDB
from audit_db import AuditDatabase
import json
import smtplib
from email.mime.text import MIMEText
import mysql.connector


load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class AgenticStripeScout:
    def __init__(self, db_path: str, user_id: str, password: str,
                 audit_db_path: str = "./audit_store.db"):

        # --- Auth & Vector DB ---
        self.db_manager = ChromaVectorDB(persist_dir=db_path)
        self.db_manager.authenticate(user_id, password)

        self.chroma_client = self.db_manager.client
        self.parent_col    = self.chroma_client.get_collection(f"{user_id}_parents")
        self.child_col     = self.chroma_client.get_collection(f"{user_id}_children")
        self.memory_col    = self.chroma_client.get_or_create_collection(f"{user_id}_memory")

        # --- Identity ---
        self.user_id    = user_id
        self.session_id = uuid.uuid4().hex   # unique per login session

        # --- Hardware & Models ---
        self.embedder  = EmbeddingEngine()
        logger.info("Loading Re-Ranker: cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.reranker  = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cuda')
        self.llm       = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model_name = "llama-3.3-70b-versatile"

        # --- State ---
        self.history   = []
        self.audit_log = []   # in-memory list kept for backward compat with trace_wrapper

        # --- Persistent Audit Store (Randy's requirement) ---
        self.audit_db  = AuditDatabase(db_path=audit_db_path)
        logger.info(f"[Agent] Session {self.session_id} started for actor: {user_id}")

    # -----------------------------------------------------------------------
    # AUDIT LOGGING  — writes to BOTH in-memory list AND persistent DB
    # -----------------------------------------------------------------------

    def _log_audit(
        self,
        event_type: str,
        details: Dict[str, Any],
        *,
        source_file:        Optional[str] = None,
        tool_input:         Optional[Any] = None,
        tool_output:        Optional[Any] = None,
        control_ref:        Optional[str] = None,
        approval_state:     str = "AUTO",
        rollback_sql:       Optional[str] = None,
        needs_human_review: bool = False,
    ) -> str:
        """
        Central audit sink.
        Writes to in-memory audit_log (for trace_wrapper) AND to the
        persistent SQLite AuditDatabase (survives restarts).
        Returns the event_id so callers can reference it later.
        """
        import datetime

        # 1. In-memory entry — keeps trace_wrapper/app.py working unchanged
        in_mem_entry = {
            "timestamp":  datetime.datetime.utcnow().isoformat() + "Z",
            "event_type": event_type,
            **details,
        }
        self.audit_log.append(in_mem_entry)
        logger.info(f"[AUDIT] {event_type}: {details}")

        # 2. Persist to SQLite
        event_id = self.audit_db.log_event(
            session_id          = self.session_id,
            actor               = self.user_id,
            event_type          = event_type,
            details             = details,
            source_file         = source_file,
            tool_input          = tool_input,
            tool_output         = tool_output,
            control_ref         = control_ref,
            approval_state      = approval_state,
            rollback_sql        = rollback_sql,
            needs_human_review  = needs_human_review,
        )
        return event_id

    # -----------------------------------------------------------------------
    # PUBLIC AUDIT QUERY HELPERS
    # An auditor can call these directly — no agent replay required.
    # -----------------------------------------------------------------------

    def get_full_audit_trail(self) -> List[Dict]:
        """Complete chronological trail for this session."""
        return self.audit_db.get_session_trail(self.session_id)

    def get_pending_reviews(self) -> List[Dict]:
        """All events flagged for human review."""
        return self.audit_db.get_pending_review_queue()

    def explain_why_control_passed(self, control_ref: str) -> List[Dict]:
        """Answer 'show me why control X passed' without replaying the agent."""
        return self.audit_db.explain_control(control_ref)

    def get_rollback_sql(self, event_id: str) -> str:
        """Return the exact SQL to undo a state-changing action by event_id."""
        sql = self.audit_db.get_rollback_info(event_id)
        return sql or "No rollback path recorded for this event."

    def approve_pending_action(self, event_id: str, reviewer: str) -> bool:
        """Approve a PENDING action (e.g. a refund waiting for sign-off)."""
        return self.audit_db.update_approval(event_id, "APPROVED", reviewer)

    def reject_pending_action(self, event_id: str, reviewer: str) -> bool:
        """Reject a PENDING action."""
        return self.audit_db.update_approval(event_id, "REJECTED", reviewer)

    # -----------------------------------------------------------------------
    # ROUTING
    # -----------------------------------------------------------------------

    def safe_json_parse(self, text: str) -> Dict:
        try:
            return json.loads(text)
        except Exception:
            import re
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                return json.loads(match.group())
            raise ValueError("Invalid router output")

    def determine_strategy(self, query: str) -> Dict:
        """AGENTIC ROUTING: High-Precision Intent Analysis."""
        logger.info("Agent Logic: Analysing Query Intent...")

        router_prompt = f"""
        You are an INTENT ROUTER for a customer support AI system.

        You MUST output ONLY valid JSON.

        No explanation. No markdown. No extra text.

        If output is not valid JSON, it is considered failure.

        ---

        STRICT OUTPUT SCHEMA:
        {{
        "intent": "CHAT | KNOWLEDGE_QUERY | ORDER_ACTION | COMPLAINT | UNKNOWN",
        "action_type": "cancel_order | change_address | track_order | refund_request | payment_issue | modify_order | view_order_details | null",
        "confidence": 0.0,
        "entities": {{
            "order_id": null,
            "email": null,
            "address": null,
            "product_id": null
        }}
        }}

        ---

        INTENT DEFINITIONS:

        CHAT:
        - greetings
        - casual conversation
        - "how are you"

        KNOWLEDGE_QUERY:
        - policies
        - documentation
        - informational questions

        ORDER_ACTION:
        - any action related to orders:
        cancel, refund, track, modify, change address, view_order_details, payment issues

        COMPLAINT:
        - anger, dissatisfaction, escalation, negative experience

        UNKNOWN:
        - unclear or unrelated queries

        ---

        RULES:

        1. ALWAYS infer intent even if user does not use exact keywords
        2. If multiple intents exist, choose the most important one
        3. Extract entities if present, otherwise keep null
        4. confidence must be between 0.0 and 1.0
        5. If unsure => intent = UNKNOWN
        6. action_type MUST be null unless intent = ORDER_ACTION

        ---

        EXAMPLES:

        User: "cancel my order 1234"
        Output:
        {{
        "intent": "ORDER_ACTION",
        "action_type": "cancel_order",
        "confidence": 0.95,
        "entities": {{
            "order_id": "1234",
            "email": null,
            "address": null,
            "product_id": null
        }}
        }}

        User: "this is worst service ever"
        Output:
        {{
        "intent": "COMPLAINT",
        "action_type": null,
        "confidence": 0.92,
        "entities": {{
            "order_id": null,
            "email": null,
            "address": null,
            "product_id": null
        }}
        }}

        User Query:
        "{query}"
        """

        response = self.llm.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a strict JSON router. You ONLY output valid JSON as defined by the schema."},
                {"role": "user",   "content": router_prompt},
            ],
            temperature=0.0,
        )

        try:
            raw  = response.choices[0].message.content
            data = self.safe_json_parse(raw)

            self._log_audit(
                "ROUTING",
                {
                    "intent":      data.get("intent"),
                    "action_type": data.get("action_type"),
                    "confidence":  data.get("confidence"),
                    "entities":    data.get("entities"),
                },
                tool_input  = {"query": query},
                tool_output = data,
                control_ref = "ROUTING_DECISION",
            )
            return data

        except Exception:
            return {"intent": "UNKNOWN", "action_type": None, "confidence": 0.0, "entities": {}}

    # -----------------------------------------------------------------------
    # ORDER ACTIONS
    # -----------------------------------------------------------------------

    def handle_order_action(self, action_type: str, entities: Dict[str, Any]) -> str:
        """
        Enterprise-grade handler for all order lifecycle actions.
        Every state-changing SQL call is logged with a full rollback path
        and persisted to the audit DB before execution.
        """
        order_id    = entities.get("order_id")
        new_address = entities.get("address")

        if order_id:
            order_id = str(order_id).strip().upper()

        if not order_id:
            logger.warning(f"Routing failed: {action_type} requested without ID.")
            return "I'm ready to help with that! Could you please provide your Order ID first?"

        conn = None
        try:
            conn = mysql.connector.connect(
                host            = "localhost",
                user            = "root",
                password        = os.getenv("MYSQL_PASSWORD", "Zain@1144"),
                database        = "supportsage_db",
                connect_timeout = 5,
            )
            cursor = conn.cursor(dictionary=True)

            cursor.execute("SELECT * FROM orders WHERE order_id = %s", (order_id,))
            order = cursor.fetchone()

            if not order:
                return f"I couldn't find an order matching ID **{order_id}**. Please verify and try again."

            # --- TRACK ---
            if action_type == "track_order":
                self._log_audit(
                    "SQL_ACTION",
                    {"action": "track_order", "order_id": order_id, "status": order["status"]},
                    source_file    = "orders",
                    tool_input     = {"order_id": order_id},
                    tool_output    = {"status": order["status"]},
                    control_ref    = "ORDER_READ",
                    approval_state = "AUTO",
                )
                return f"Tracking Update: Your order for {order['product_name']} is currently **{order['status']}**."

            # --- CHANGE ADDRESS ---
            elif action_type == "change_address":
                if not new_address:
                    return f"I've found order {order_id}, but I need the new delivery address to proceed."

                if order["status"].lower() in ["shipped", "delivered"]:
                    return f"Update Blocked: Order {order_id} is already **{order['status']}** and cannot be rerouted."

                rollback = f"UPDATE orders SET address = '{order.get('address')}' WHERE order_id = '{order_id}'"
                event_id = self._log_audit(
                    "SQL_ACTION",
                    {
                        "action":               "change_address",
                        "order_id":             order_id,
                        "old_address":          order.get("address"),
                        "new_address":          new_address,
                        "order_status_at_time": order.get("status"),
                        "sql":                  "UPDATE orders SET address = %s WHERE order_id = %s",
                    },
                    source_file    = "orders",
                    tool_input     = {"order_id": order_id, "new_address": new_address},
                    tool_output    = {"old_address": order.get("address")},
                    control_ref    = "ORDER_WRITE",
                    approval_state = "AUTO",
                    rollback_sql   = rollback,
                )
                cursor.execute("UPDATE orders SET address = %s WHERE order_id = %s", (new_address, order_id))
                conn.commit()
                return (f"Address Updated: Shipping for {order_id} changed to: {new_address}.\n"
                        f"_(Audit ref: `{event_id}`)_")

            # --- CANCEL ---
            elif action_type == "cancel_order":
                if order["status"].lower() in ["shipped", "delivered"]:
                    return f"Cancellation Declined: Order {order_id} is already **{order['status']}**. Please initiate a return."

                if order["status"].lower() == "cancelled":
                    return f"Order {order_id} is already marked as 'Cancelled'."

                rollback = f"UPDATE orders SET status = '{order.get('status')}' WHERE order_id = '{order_id}'"
                event_id = self._log_audit(
                    "SQL_ACTION",
                    {
                        "action":          "cancel_order",
                        "order_id":        order_id,
                        "previous_status": order.get("status"),
                        "sql":             "UPDATE orders SET status = 'Cancelled' WHERE order_id = %s",
                    },
                    source_file    = "orders",
                    tool_input     = {"order_id": order_id},
                    tool_output    = {"new_status": "Cancelled"},
                    control_ref    = "ORDER_WRITE",
                    approval_state = "AUTO",
                    rollback_sql   = rollback,
                )
                cursor.execute("UPDATE orders SET status = 'Cancelled' WHERE order_id = %s", (order_id,))
                conn.commit()
                return (f"Order Cancelled: Order {order_id} has been stopped. "
                        f"You will receive a refund confirmation via email.\n"
                        f"_(Audit ref: `{event_id}`)_")

            # --- REFUND --- (requires human approval)
            elif action_type == "refund_request":
                if order["status"].lower() == "cancelled":
                    return f"Refund Status: A refund for order {order_id} is already being processed."

                event_id = self._log_audit(
                    "SQL_ACTION",
                    {
                        "action":   "refund_request",
                        "order_id": order_id,
                        "status":   order.get("status"),
                        "note":     "Refund pending human approval before processing.",
                    },
                    source_file         = "orders",
                    tool_input          = {"order_id": order_id},
                    control_ref         = "ORDER_REFUND",
                    approval_state      = "PENDING",    # human must approve
                    needs_human_review  = True,
                )
                return (f"Refund Initiated: Refund for order {order_id} is now **PENDING** specialist approval.\n"
                        f"A support team member will review within 24 hours.\n"
                        f"_(Audit ref: `{event_id}` — status: PENDING)_")

            # --- MODIFY ---
            elif action_type == "modify_order":
                valid_states = ["pending", "in cart", "cart"]
                if order["status"].lower() not in valid_states:
                    return (f"Modification Period Expired: Order {order_id} is in the "
                            f"'{order['status']}' phase and cannot be changed.")

                new_item = entities.get("product_name") or entities.get("item")
                if not new_item:
                    return f"I've accessed order {order_id}. What item would you like to change it to?"

                rollback = f"UPDATE orders SET product_name = '{order.get('product_name')}' WHERE order_id = '{order_id}'"
                event_id = self._log_audit(
                    "SQL_ACTION",
                    {
                        "action":               "modify_order",
                        "order_id":             order_id,
                        "old_product":          order.get("product_name"),
                        "new_product":          new_item,
                        "order_status_at_time": order.get("status"),
                        "sql":                  "UPDATE orders SET product_name = %s WHERE order_id = %s",
                    },
                    source_file    = "orders",
                    tool_input     = {"order_id": order_id, "new_product": new_item},
                    tool_output    = {"old_product": order.get("product_name")},
                    control_ref    = "ORDER_WRITE",
                    approval_state = "AUTO",
                    rollback_sql   = rollback,
                )
                cursor.execute("UPDATE orders SET product_name = %s WHERE order_id = %s", (new_item, order_id))
                conn.commit()

                if cursor.rowcount > 0:
                    return (f"Order Modified: Order {order_id} updated to **{new_item}**.\n"
                            f"_(Audit ref: `{event_id}`)_")
                return f"No changes made. Order {order_id} already contains '{new_item}'."

            # --- PAYMENT ISSUE ---
            elif action_type == "payment_issue":
                self._log_audit(
                    "SQL_ACTION",
                    {"action": "payment_issue_inquiry", "order_id": order_id},
                    source_file    = "orders",
                    tool_input     = {"order_id": order_id},
                    control_ref    = "ORDER_READ",
                    approval_state = "AUTO",
                )
                return "Payment Support: Please use our encrypted portal to update payment details. Do not share card info here."

            # --- VIEW DETAILS ---
            elif action_type == "view_order_details":
                self._log_audit(
                    "SQL_ACTION",
                    {"action": "view_order_details", "order_id": order_id},
                    source_file    = "orders",
                    tool_input     = {"order_id": order_id},
                    tool_output    = {k: order.get(k) for k in ["product_name", "status", "address"]},
                    control_ref    = "ORDER_READ",
                    approval_state = "AUTO",
                )
                return (
                    f"Order Summary for {order_id}:\n"
                    f"Product: {order['product_name']}\n"
                    f"Status: {order['status']}\n"
                    f"Shipping to: {order['address']}"
                )

            return (f"I've recognised your request as '{action_type.replace('_', ' ')}', "
                    f"but I need to consult a human agent for order {order_id}.")

        except mysql.connector.Error as err:
            logger.error(f"SYSTEM DATABASE ERROR: {err}")
            return "Service Interruption: Unable to reach the order database. Please try again shortly."

        finally:
            if conn and conn.is_connected():
                cursor.close()
                conn.close()
                logger.info("Database connection released.")

    # -----------------------------------------------------------------------
    # COMPLAINT HANDLER
    # -----------------------------------------------------------------------

    def send_complaint_email(self, query: str, entities: Dict) -> bool:
        sender_email   = os.getenv("SENDER_EMAIL")
        receiver_email = os.getenv("RECIEVER_EMAIL")
        password       = os.getenv("GMAIL_PASSWORD")

        msg = MIMEText(f"User Query: {query}\n\nExtracted Entities: {entities}")
        msg["Subject"] = "New SupportSage Complaint"
        msg["From"]    = sender_email
        msg["To"]      = receiver_email

        try:
            server = smtplib.SMTP("smtp.gmail.com", 587, timeout=10)
            server.starttls()
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, msg.as_string())
            server.quit()
            return True
        except Exception as e:
            logger.error(f"Email connection failed: {e}")
            return False

    def handle_complaint(self, query: str, entities: Dict) -> str:
        logger.info(f"Complaint detected: {query}")
        self._log_audit(
            "COMPLAINT",
            {"query_preview": query[:120], "entities": entities},
            needs_human_review = True,
            approval_state     = "PENDING",
            control_ref        = "ESCALATION",
        )
        success = self.send_complaint_email(query, entities)
        if success:
            return "I have forwarded your complaint to our support team. They will contact you soon."
        return "I've noted your complaint. Rest assured, it's being processed."

    # -----------------------------------------------------------------------
    # MEMORY
    # -----------------------------------------------------------------------

    def check_long_term_memory(self, query_embedding: List[float]) -> Optional[str]:
        results = self.memory_col.query(query_embeddings=[query_embedding], n_results=1)
        if results["distances"] and results["distances"][0] and results["distances"][0][0] < 0.4:
            logger.info("Brain: Found a match in Long-Term Memory!")
            return results["documents"][0][0]
        return None

    def store_in_memory(self, query: str, answer: str, embedding: List[float]):
        logger.info("Learning: Saving experience to Long-Term Memory...")
        self.memory_col.add(
            ids        = [str(uuid.uuid4())],
            embeddings = [embedding],
            documents  = [answer],
            metadatas  = [{"query": query}],
        )

    # -----------------------------------------------------------------------
    # RETRIEVAL & GENERATION
    # -----------------------------------------------------------------------

    def rerank_context(self, query: str, docs_with_sources: List[Dict]) -> str:
        if not docs_with_sources:
            return ""

        logger.info(f"Re-Ranking {len(docs_with_sources)} documents for maximum precision...")
        pairs  = [[query, d["text"]] for d in docs_with_sources]
        scores = self.reranker.predict(pairs)
        scored = sorted(zip(scores, docs_with_sources), key=lambda x: x[0], reverse=True)

        formatted_context = []
        audit_snippets    = []
        for score, data in scored[:5]:
            formatted_context.append(f"[Source: {data['source']}]\n{data['text']}")
            audit_snippets.append({
                "source":          data["source"],
                "rerank_score":    round(float(score), 4),
                "snippet_preview": data["text"][:120] + ("..." if len(data["text"]) > 120 else ""),
            })

        self._log_audit(
            "RAG_RETRIEVAL",
            {"num_sources": len(audit_snippets), "top_sources": audit_snippets},
            source_file = audit_snippets[0]["source"] if audit_snippets else None,
            tool_input  = {"query": query},
            tool_output = audit_snippets,
            control_ref = "KNOWLEDGE_RETRIEVAL",
        )
        return "\n---\n".join(formatted_context)

    def retrieve_context(self, query: str, query_embedding: List[float], top_k: int = 15) -> str:
        logger.info("Searching Intelligence Core (Vector DB)...")
        results = self.child_col.query(query_embeddings=[query_embedding], n_results=top_k)

        if not results or not results["metadatas"] or not results["metadatas"][0]:
            logger.warning("No matching child chunks found.")
            return ""

        raw_parent_ids    = [m["parent_ref"] for m in results["metadatas"][0] if "parent_ref" in m]
        unique_parent_ids = list(set(raw_parent_ids))
        if not unique_parent_ids:
            return ""

        parent_data       = self.parent_col.get(ids=unique_parent_ids)
        docs_with_sources = [
            {"text": doc, "source": meta.get("source", "Unknown Source")}
            for doc, meta in zip(parent_data["documents"], parent_data["metadatas"])
        ]
        return self.rerank_context(query, docs_with_sources)

    def generate_response(self, query: str, context: str = None) -> str:
        if context:
            system_prompt = (
                "You are a Technical Stripe Expert. "
                "Use the provided context to answer. "
                "CRITICAL: For every factual claim, state: 'According to [source name]...' "
                "If the context doesn't have the answer, say you don't know.\n\n"
                f"Context:\n{context}"
            )
        else:
            system_prompt = (
                "You are SupportSage Pro, a high-performance RAG Agent. "
                "Assist with general queries when the database is not needed."
            )

        messages = [{"role": "system", "content": system_prompt}, *self.history[-4:], {"role": "user", "content": query}]
        response = self.llm.chat.completions.create(model=self.model_name, messages=messages, temperature=0.3)
        return response.choices[0].message.content

    def reflect_and_score(self, query: str, context: str, answer: str) -> Tuple[int, str]:
        logger.info("Reflection: Verifying technical accuracy...")
        prompt = (
            f"Query: {query}\nContext: {context}\nAnswer: {answer}\n"
            f"Rate 1-10. If <10, correct it. FORMAT: SCORE: [num] | FINAL_ANSWER: [text]"
        )
        response = self.llm.chat.completions.create(
            model       = self.model_name,
            messages    = [{"role": "system", "content": prompt}],
            temperature = 0.1,
        )
        content = response.choices[0].message.content
        try:
            score      = int(content.split("|")[0].replace("SCORE:", "").strip())
            final_text = content.split("|")[1].replace("FINAL_ANSWER:", "").strip()
            return score, final_text
        except (ValueError, IndexError) as e:
            logger.error(f"Reflection parse failed: {e}")
            return 10, answer

    # -----------------------------------------------------------------------
    # MAIN CHAT ENTRY POINT
    # -----------------------------------------------------------------------

    def _return(self, user_query: str, answer: str) -> str:
        self.history.append({"role": "user",      "content": user_query})
        self.history.append({"role": "assistant",  "content": answer})
        return answer

    def chat(self, user_query: str) -> str:

        # 1. Encode query
        query_emb = self.embedder.model.encode(
            [user_query], normalize_embeddings=True
        )[0].tolist()

        # 2. Long-term memory check
        learned_answer = self.check_long_term_memory(query_emb)
        if learned_answer:
            self._log_audit(
                "MEMORY_HIT",
                {"query_preview": user_query[:80]},
                tool_input  = {"query": user_query},
                tool_output = {"answer_source": "long_term_memory"},
                control_ref = "KNOWLEDGE_RETRIEVAL",
            )
            return self._return(user_query, learned_answer)

        # 3. Routing
        route       = self.determine_strategy(user_query)
        intent      = route.get("intent")
        action_type = route.get("action_type")
        entities    = route.get("entities")
        confidence  = route.get("confidence", 0.0)

        # 4. Low-confidence fallback
        if confidence < 0.3:
            intent = "UNKNOWN"

        # 5. CHAT
        if intent == "CHAT":
            return self._return(user_query, self.generate_response(user_query))

        # 6. ORDER ACTION
        if intent == "ORDER_ACTION":
            return self._return(user_query, self.handle_order_action(action_type, entities))

        # 7. COMPLAINT
        if intent == "COMPLAINT":
            return self._return(user_query, self.handle_complaint(user_query, entities))

        # 8. KNOWLEDGE QUERY
        if intent == "KNOWLEDGE_QUERY":
            context        = self.retrieve_context(user_query, query_emb)
            current_answer = self.generate_response(user_query, context)
            score          = 10

            for _ in range(2):
                score, improved_answer = self.reflect_and_score(user_query, context, current_answer)
                current_answer = improved_answer
                logger.info(f"Quality Score: {score}/10")
                if score >= 9:
                    break

            needs_review = score < 7
            self._log_audit(
                "REFLECTION_SCORE",
                {
                    "reflection_score":  score,
                    "needs_human_review": needs_review,
                    "query_preview":     user_query[:80],
                },
                tool_input         = {"query": user_query},
                tool_output        = {"score": score},
                control_ref        = "ANSWER_QUALITY",
                approval_state     = "PENDING" if needs_review else "AUTO",
                needs_human_review = needs_review,
            )

            if score >= 7:
                self.store_in_memory(user_query, current_answer, query_emb)

            return self._return(user_query, current_answer)

        # 9. UNKNOWN fallback
        return self._return(user_query, "Sorry, I couldn't understand your request clearly.")


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    DB_PATH = r"D:\project dataset\RAG project\chromadb"

    print("--- SECURE AGENT LOGIN ---")
    input_id = input("Enter User ID: ")
    input_pw = input("Enter Password: ")

    try:
        agent = AgenticStripeScout(db_path=DB_PATH, user_id=input_id, password=input_pw)
        print(f"\nConnection established for {input_id}. Workspace loaded.\n")

        while True:
            user_input = input("\nUser: ")
            if user_input.lower() in ["exit", "quit"]:
                break
            print(f"\nAgent: {agent.chat(user_input)}")

    except Exception as e:
        print(f"\nAccess Denied: {str(e)}")
import os
import logging
from typing import List, Dict, Any, Tuple
from groq import Groq 
import chromadb
from embedding import EmbeddingEngine 
from dotenv import load_dotenv
import uuid
from sentence_transformers import CrossEncoder
# IMPORT the database manager we created
from database import ChromaVectorDB
import json
import smtplib
from email.mime.text import MIMEText
import mysql.connector


load_dotenv()

# --- PROFESSIONAL LOGGING ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class AgenticStripeScout:
    def __init__(self, db_path: str, user_id: str, password: str):
        # ... (Your existing DB and Auth code) ...
        self.db_manager = ChromaVectorDB(persist_dir=db_path)
        self.db_manager.authenticate(user_id, password)
        
        self.chroma_client = self.db_manager.client
        self.parent_col = self.chroma_client.get_collection(f"{user_id}_parents")
        self.child_col = self.chroma_client.get_collection(f"{user_id}_children")
        self.memory_col = self.chroma_client.get_or_create_collection(f"{user_id}_memory")
        
        # Hardware & Brain
        self.embedder = EmbeddingEngine() 
        
        # NEW: Load the Re-Ranker Model (Runs on your 3060 Ti)
        logger.info("⚡ Loading Re-Ranker Intelligence: cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cuda')
        
        self.llm = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model_name = "llama-3.3-70b-versatile"
        self.history = []


    def safe_json_parse(self, text):
        try:
            return json.loads(text)
        except:
            import re
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                return json.loads(match.group())
            raise ValueError("Invalid router output")
        
    def determine_strategy(self, query: str) -> str:
        """AGENTIC ROUTING: High-Precision Intent Analysis."""
        logger.info("🚦 Agent Logic: Analyzing Query Intent...")
        
        # We give the LLM 'Reasoning' instructions
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
        5. If unsure → intent = UNKNOWN
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
                {"role": "user", "content": router_prompt}
            ],
            temperature=0.0 # CRITICAL: 0.0 for consistent routing
        )
        
        try:
            raw = response.choices[0].message.content
            data = self.safe_json_parse(raw)
            return data
        except:
            return {
                "intent": "UNKNOWN",
                "action_type": None,
                "confidence": 0.0,
                "entities": {}
            }# score


    def handle_order_action(self, action_type: str, entities: Dict[str, Any]) -> str:
        """
        Enterprise-grade handler for all order-related lifecycle actions.
        Implements state validation, resource cleanup, and parameterized SQL.
        """
        # 1. Extraction & Sanitization
        order_id = entities.get("order_id")
        new_address = entities.get("address")
        
        # Professional Step: Normalize input to avoid ID typos
        if order_id:
            order_id = str(order_id).strip().upper()

        # 2. Early Guard Clause
        if not order_id:
            logger.warning(f"Routing failed: {action_type} requested without ID.")
            return "I'm ready to help with that! Could you please provide your Order ID first?"

        conn = None
        try:
            # 3. Secure Database Connection
            conn = mysql.connector.connect(
                host="localhost",
                user="root", 
                password="Zain@1144",
                database="supportsage_db",
                connect_timeout=5
            )
            cursor = conn.cursor(dictionary=True)

            # 4. Fetch the 'Source of Truth' (The current order record)
            query_fetch = "SELECT * FROM orders WHERE order_id = %s"
            cursor.execute(query_fetch, (order_id,))
            order = cursor.fetchone()

            # Guard: Check if order even exists
            if not order:
                logger.info(f"Order lookup failed for ID: {order_id}")
                return f"❌ I couldn't find an order matching ID **{order_id}**. Please verify the number and try again."

            # 5. Routing Logic (Handling all cases from your Router Prompt)
            
            # --- ACTION: TRACK ORDER ---
            if action_type == "track_order":
                return f"📦 **Tracking Update**: Your order for {order['product_name']} is currently **{order['status']}**."

            # --- ACTION: CHANGE ADDRESS ---
            elif action_type == "change_address":
                if not new_address:
                    return f"I've found order {order_id}, but I need the new delivery address to proceed."
                
                # Business Rule: Block if already in transit
                if order['status'].lower() in ['shipped', 'delivered']:
                    return f"⚠️ **Update Blocked**: Order {order_id} is already **{order['status']}** and cannot be rerouted at this stage."

                cursor.execute("UPDATE orders SET address = %s WHERE order_id = %s", (new_address, order_id))
                conn.commit()
                return f"✅ **Address Updated**: The shipping destination for {order_id} has been changed to: {new_address}."

            # --- ACTION: CANCEL ORDER ---
            elif action_type == "cancel_order":
                # Business Rule: Cannot cancel if it's already left the building
                if order['status'].lower() in ['shipped', 'delivered']:
                    return f"🚫 **Cancellation Declined**: Order {order_id} is already **{order['status']}**. Please initiate a return once it arrives."
                
                if order['status'].lower() == 'cancelled':
                    return f"ℹ️ Order {order_id} is already marked as 'Cancelled'."

                cursor.execute("UPDATE orders SET status = 'Cancelled' WHERE order_id = %s", (order_id,))
                conn.commit()
                return f"🛑 **Order Cancelled**: Order {order_id} has been successfully stopped. You will receive a refund confirmation via email."

            # --- ACTION: REFUND REQUEST ---
            elif action_type == "refund_request":
                # Logic: Refund is automatic for cancelled, otherwise requires review
                if order['status'].lower() == 'cancelled':
                    return f"💰 **Refund Status**: A refund for order {order_id} is already being processed to your original payment method."
                
                return f"💸 **Refund Initiated**: I have opened a refund request for order {order_id}. A support specialist will review this within 24 hours."

            # --- ACTION: MODIFY ORDER ---
            # --- CASE: MODIFY ORDER ---
            elif action_type == "modify_order":
                # 1. Business Rule: Only 'Pending' or 'In Cart' orders can be modified
                valid_states = ['pending', 'in cart', 'cart']
                if order['status'].lower() not in valid_states:
                    return (f"⚠️ **Modification Period Expired**: Order {order_id} is already in the "
                            f"'{order['status']}' phase. We cannot change items once processing begins.")

                # 2. Extraction: See if the LLM extracted a new product/item
                # Note: We look for 'product_name' or 'item' in the entities
                new_item = entities.get("product_name") or entities.get("item")

                if not new_item:
                    return f"✏️ I've accessed order {order_id}. What specific item or product would you like to change it to?"

                # 3. Execution: Update the product_name in the database
                update_sql = "UPDATE orders SET product_name = %s WHERE order_id = %s"
                cursor.execute(update_sql, (new_item, order_id))
                conn.commit()

                if cursor.rowcount > 0:
                    logger.info(f"MODIFICATION SUCCESS: Order {order_id} item changed to {new_item}")
                    return (f"✅ **Order Modified**: Order {order_id} has been updated. "
                            f"The new item is: **{new_item}**.")
                else:
                    return f"ℹ️ No changes made. Order {order_id} already contains '{new_item}'."

            # --- ACTION: PAYMENT ISSUE ---
            elif action_type == "payment_issue":
                return f"💳 **Payment Support**: I see your inquiry regarding payment for {order_id}. For your security, please use our encrypted portal [link] to update your details. **Do not share card info here.**"
            
            elif action_type == "view_order_details":
                return (
                    f"📝 **Order Summary for {order_id}**:\n"
                    f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"📦 **Product:** {order['product_name']}\n"
                    f"🚦 **Status:** {order['status']}\n"
                    f"📍 **Shipping to:** {order['address']}\n"
                    f"━━━━━━━━━━━━━━━━━━━━━━━━"
                )

            # 6. Fallback for undefined actions
            return f"I've recognized your request as '{action_type.replace('_', ' ')}', but I need to consult a human agent to finish this specific task for order {order_id}."

        except mysql.connector.Error as err:
            logger.error(f"SYSTEM DATABASE ERROR: {err}")
            return "🛡️ Service Interruption: I'm currently unable to sync with the order database. Please try again in a few moments."
        
        finally:
            # Resource Management: Always close connections
            if conn and conn.is_connected():
                cursor.close()
                conn.close()
                logger.info("Database connection released.")
    

    def send_complaint_email(self, query, entities):
        sender_email = os.getenv("SENDER_EMAIL")
        receiver_email = os.getenv("RECIEVER_EMAIL")
        password = os.getenv("GMAIL_PASSWORD") 

        msg = MIMEText(f"User Query: {query}\n\nExtracted Entities: {entities}")
        msg['Subject'] = f"New SupportSage Complaint"
        msg['From'] = sender_email
        msg['To'] = receiver_email

        try:
            # Use Port 587 and starttls() if 465 hangs
            server = smtplib.SMTP("smtp.gmail.com", 587, timeout=10)
            server.starttls() 
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, msg.as_string())
            server.quit()
            return True
        except Exception as e:
            logger.error(f"Email connection failed: {e}")
            return False

    def handle_complaint(self, query, entities):
        # Log it locally for the terminal/trace
        logger.info(f"Complaint detected: {query}")
        
        # Send the email (we will define this function next)
        success = self.send_complaint_email(query, entities)
        
        if success:
            return "I have forwarded your complaint to our support team. They will contact you at your registered email soon."
        else:
            return "I've noted your complaint, but I'm having trouble reaching the support server right now. Rest assured, it's being processed."
    
    def check_long_term_memory(self, query_embedding: List[float]) -> str:
        """MEMORY RETRIEVAL: Checks if we have learned this before."""
        results = self.memory_col.query(query_embeddings=[query_embedding], n_results=1)
        if results['distances'] and results['distances'][0] and results['distances'][0][0] < 0.4: 
            logger.info("Brain: Found a match in Long-Term Memory! Using learned experience.")
            return results['documents'][0][0]
        return None

    def store_in_memory(self, query: str, answer: str, embedding: List[float]):
        """INTELLIGENCE ACCUMULATION: Saves high-quality answers for future use."""
        logger.info("Learning: Saving this experience to Long-Term Memory...")
        self.memory_col.add(
            ids=[str(uuid.uuid4())],
            embeddings=[embedding],
            documents=[answer],
            metadatas=[{"query": query}]
        )

    def _return(self, user_query, answer):
        self.history.append({"role": "user", "content": user_query})
        self.history.append({"role": "assistant", "content": answer})
        return answer

    def rerank_context(self, query: str, docs_with_sources: List[Dict]) -> str:
        """RE-RANKING: Finds best matches and formats citations."""
        if not docs_with_sources:
            return ""
            
        logger.info(f"Re-Ranking {len(docs_with_sources)} document for maximum precision...")
        
        # Re-rank based on the 'text' key
        pairs = [[query, d['text']] for d in docs_with_sources]
        scores = self.reranker.predict(pairs)
        
        # Zip scores with the full dictionaries
        scored_data = sorted(zip(scores, docs_with_sources), key=lambda x: x[0], reverse=True)
        
        # Take top 5 and format with [Source: name]
        formatted_context = []
        for score, data in scored_data[:5]:
            formatted_context.append(f"[Source: {data['source']}]\n{data['text']}")
        
        return "\n---\n".join(formatted_context)


    def retrieve_context(self, query: str, query_embedding: List[float], top_k: int = 15) -> str:
        """DOCUMENT RETRIEVAL: Pulls text AND source metadata."""
        logger.info("🔍 Searching Intelligence Core (Vector DB)...")
        
        results = self.child_col.query(query_embeddings=[query_embedding], n_results=top_k)
        
        # --- ADD THIS SAFETY CHECK HERE ---
        if not results or not results['metadatas'] or not results['metadatas'][0]:
            logger.warning("No matching child chunks found.")
            return ""

        raw_parent_ids = [meta['parent_ref'] for meta in results['metadatas'][0] if 'parent_ref' in meta]
        unique_parent_ids = list(set(raw_parent_ids)) 

        if not unique_parent_ids:
            return ""
        
        # Pull parents
        parent_data = self.parent_col.get(ids=unique_parent_ids)
        
        # Create a list of dictionaries containing text + source
        # Assuming your metadata has a key called 'source' or 'filename'
        docs_with_sources = []
        for doc, meta in zip(parent_data['documents'], parent_data['metadatas']):
            source_name = meta.get('source', 'Unknown Source')
            docs_with_sources.append({"text": doc, "source": source_name})
        
        # Step 2: Precision Re-Ranking (passing the list of dicts)
        return self.rerank_context(query, docs_with_sources)
    
    def generate_response(self, query: str, context: str = None) -> str:
        """COGNITIVE GENERATION: Now with mandatory citation rules."""
        if context:
            system_prompt = f"""
            You are a Technical Stripe Expert. 
            Use the provided context to answer. 
            CRITICAL: You must cite your sources. For every factual claim, state: 'According to [source name]...' 
            If the context doesn't have the answer, say you don't know.
            
            Context:
            {context}
            """
        else:
            system_prompt = "You are SupportSage Pro, a high-performance RAG Agent running on Zain's RTX 3060 Ti hardware. You are here to assist with general queries when the database is not needed."

        messages = [{"role": "system", "content": system_prompt}, *self.history[-4:], {"role": "user", "content": query}]
        response = self.llm.chat.completions.create(model=self.model_name, messages=messages, temperature=0.3)
        return response.choices[0].message.content

    def reflect_and_score(self, query: str, context: str, answer: str) -> Tuple[int, str]:
        """RECURSIVE CRITIQUE: Self-correction gate."""
        logger.info("Reflection: Verifying technical accuracy...")
        reflection_prompt = f"Query: {query}\nContext: {context}\nAnswer: {answer}\nRate 1-10. If <10, correct it. FORMAT: SCORE: [num] | FINAL_ANSWER: [text]"
        response = self.llm.chat.completions.create(model=self.model_name, messages=[{"role": "system", "content": reflection_prompt}], temperature=0.1)
        content = response.choices[0].message.content
        try:
            score = int(content.split('|')[0].replace("SCORE:", "").strip())
            final_text = content.split('|')[1].replace("FINAL_ANSWER:", "").strip()
            return score, final_text
        except (ValueError, IndexError) as e:
            logger.error(f"Reflection parse failed: {e}")
            return 10, answer

    def chat(self, user_query: str):

        # 1. Encode query
        query_emb = self.embedder.model.encode(
            [user_query],
            normalize_embeddings=True
        )[0].tolist()

        # 2. Long-term memory check
        learned_answer = self.check_long_term_memory(query_emb)
        if learned_answer:
            return self._return(user_query, learned_answer)

        # 3. ROUTING
        route = self.determine_strategy(user_query)
        intent = route.get("intent")
        action_type = route.get("action_type")
        entities = route.get("entities")
        confidence = route.get("confidence", 0.0)

        # 4. LOW CONFIDENCE fallback
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
            context = self.retrieve_context(user_query, query_emb)
            current_answer = self.generate_response(user_query, context)

            for i in range(2):
                score, improved_answer = self.reflect_and_score(
                    user_query, context, current_answer
                )
                current_answer = improved_answer
                logger.info(f"Quality Score: {score}/10")
                if score >= 9:
                    break

            if score >= 7:
                self.store_in_memory(user_query, current_answer, query_emb)
            return self._return(user_query, current_answer)

        # 9. UNKNOWN fallback
        return self._return(user_query, "Sorry, I couldn't understand your request clearly.")
    


if __name__ == "__main__":
    DB_PATH = r"D:\project dataset\RAG project\chromadb"
    
    print("--- 🔐 SECURE AGENT LOGIN ---")
    input_id = input("Enter User ID: ")
    input_pw = input("Enter Password: ")

    try:
        # Initialize agent with security
        agent = AgenticStripeScout(db_path=DB_PATH, user_id=input_id, password=input_pw)
        print(f"\nConnection established for {input_id}. Workspace loaded.\n")
        
        while True:
            user_input = input("\n👤 User: ")
            if user_input.lower() in ['exit', 'quit']: break
            print(f"\nAgent: {agent.chat(user_input)}")
            
    except Exception as e:
        print(f"\nAccess Denied: {str(e)}")
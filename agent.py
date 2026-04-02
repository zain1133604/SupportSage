import os
import logging
from typing import List, Dict, Any, Tuple
from groq import Groq 
import chromadb
from embedding import EmbeddingEngine 
from dotenv import load_dotenv
import uuid

load_dotenv()

# --- PROFESSIONAL LOGGING ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class AgenticStripeScout:
    def __init__(self, db_path: str):
        # 1. Database Connections
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        self.parent_col = self.chroma_client.get_collection("parent_chunks")
        self.child_col = self.chroma_client.get_collection("child_chunks")
        
        # 2. NEW: LONG-TERM MEMORY COLLECTION (Intelligence Accumulation)
        self.memory_col = self.chroma_client.get_or_create_collection("long_term_memory")
        
        # 3. Hardware & Brain
        self.embedder = EmbeddingEngine() # Powering the RTX 3060 Ti
        self.llm = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model_name = "llama-3.3-70b-versatile"
        self.history = []

    def determine_strategy(self, query: str) -> str:
        """AGENTIC ROUTING: Decides the workflow path."""
        logger.info("🚦 Thinking: Selecting optimized execution path...")
        router_prompt = f"""
        Analyze: "{query}"
        Route to:
        1. STRIPE_DOCS: Technical Stripe/Payment queries.
        2. CHAT: Greetings or general conversation.
        Return ONLY the word: STRIPE_DOCS or CHAT.
        """
        response = self.llm.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "system", "content": router_prompt}],
            temperature=0.0
        )
        return response.choices[0].message.content.strip()

    def check_long_term_memory(self, query_embedding: List[float]) -> str:
        """MEMORY RETRIEVAL: Checks if we have learned this before."""
        results = self.memory_col.query(query_embeddings=[query_embedding], n_results=1)
        if results['distances'][0] and results['distances'][0][0] < 0.15: # High similarity threshold
            logger.info("🧠 Brain: Found a match in Long-Term Memory! Using learned experience.")
            return results['documents'][0][0]
        return None

    def store_in_memory(self, query: str, answer: str, embedding: List[float]):
        """INTELLIGENCE ACCUMULATION: Saves high-quality answers for future use."""
        logger.info("💾 Learning: Saving this experience to Long-Term Memory...")
        self.memory_col.add(
            ids=[str(uuid.uuid4())],
            embeddings=[embedding],
            documents=[answer],
            metadatas=[{"query": query}]
        )

    def retrieve_context(self, query_embedding: List[float], top_k: int = 10) -> str:
        """DOCUMENT RETRIEVAL: Search optimized for ChromaDB unique parent IDs."""
        logger.info("🔍 Searching Intelligence Core (Vector DB)...")
        results = self.child_col.query(query_embeddings=[query_embedding], n_results=top_k)
        raw_parent_ids = [meta['parent_ref'] for meta in results['metadatas'][0]]
        unique_parent_ids = list(set(raw_parent_ids)) 
        parent_docs = self.parent_col.get(ids=unique_parent_ids)
        return "\n---\n".join(parent_docs['documents'])

    def generate_response(self, query: str, context: str = None) -> str:
        """COGNITIVE GENERATION: Drafts the initial response."""
        system_prompt = f"You are a Technical Stripe Expert. Context: {context}" if context else "You are a helpful AI Assistant."
        messages = [{"role": "system", "content": system_prompt}, *self.history[-4:], {"role": "user", "content": query}]
        response = self.llm.chat.completions.create(model=self.model_name, messages=messages, temperature=0.3)
        return response.choices[0].message.content

    def reflect_and_score(self, query: str, context: str, answer: str) -> Tuple[int, str]:
        """RECURSIVE CRITIQUE: Self-correction gate."""
        logger.info("🧠 Reflection: Verifying technical accuracy...")
        reflection_prompt = f"Query: {query}\nContext: {context}\nAnswer: {answer}\nRate 1-10. If <10, correct it. FORMAT: SCORE: [num] | FINAL_ANSWER: [text]"
        response = self.llm.chat.completions.create(model=self.model_name, messages=[{"role": "system", "content": reflection_prompt}], temperature=0.1)
        content = response.choices[0].message.content
        try:
            score = int(content.split('|')[0].replace("SCORE:", "").strip())
            final_text = content.split('|')[1].replace("FINAL_ANSWER:", "").strip()
            return score, final_text
        except: return 10, answer

    def chat(self, user_query: str):
        # Generate embedding once for both Memory and Doc search
        query_emb = self.embedder.model.encode([user_query], normalize_embeddings=True)[0].tolist()
        
        # 1. Check Long-Term Memory first (LEARNING)
        learned_answer = self.check_long_term_memory(query_emb)
        if learned_answer:
            return learned_answer

        # 2. Strategy Routing (DECISION MAKING)
        strategy = self.determine_strategy(user_query)
        context = self.retrieve_context(query_emb) if strategy == "STRIPE_DOCS" else None
        
        # 3. Initial Generation
        current_answer = self.generate_response(user_query, context)

        # 4. Dynamic Reflection (REASONING)
        if strategy == "STRIPE_DOCS":
            for i in range(2):
                score, improved_answer = self.reflect_and_score(user_query, context, current_answer)
                logger.info(f"📊 Quality Score: {score}/10")
                if score >= 9:
                    # 5. Store in Memory if high quality (EVOLUTION)
                    self.store_in_memory(user_query, improved_answer, query_emb)
                    current_answer = improved_answer
                    break
                current_answer = improved_answer
        
        self.history.append({"role": "user", "content": user_query})
        self.history.append({"role": "assistant", "content": current_answer})
        return current_answer

if __name__ == "__main__":
    DB_PATH = r"D:\project dataset\RAG project\chromadb"
    agent = AgenticStripeScout(db_path=DB_PATH)
    while True:
        user_input = input("\n👤 User: ")
        if user_input.lower() in ['exit', 'quit']: break
        print(f"\n🤖 Agent: {agent.chat(user_input)}")
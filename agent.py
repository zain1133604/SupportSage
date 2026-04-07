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

    def determine_strategy(self, query: str) -> str:
        """AGENTIC ROUTING: High-Precision Intent Analysis."""
        logger.info("🚦 Agent Logic: Analyzing Query Intent...")
        
        # We give the LLM 'Reasoning' instructions
        router_prompt = f"""
        You are the Gatekeeper for a Secure Knowledge Base. 
        Your task is to classify the incoming Query: "{query}"

        CLASSIFICATION RULES:
        - [STRIPE_DOCS]: If the query asks for facts, technical details, people, specific incidents, or data that would be stored in a private company database. 
        - [CHAT]: Only for pure social interaction (e.g., "How are you?", "Who made you?", "Hello").

        DECISION RUBRIC:
        - Does the query contain a Proper Noun (e.g., 'Zain')? -> STRIPE_DOCS
        - Does the query ask 'Did X happen?' or 'Is Y true?' -> STRIPE_DOCS
        - Is this a technical Stripe integration question? -> STRIPE_DOCS

        Response MUST be exactly one word: STRIPE_DOCS or CHAT.
        """
        
        response = self.llm.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a logical routing module. You only output single labels."},
                {"role": "user", "content": router_prompt}
            ],
            temperature=0.0 # CRITICAL: 0.0 for consistent routing
        )
        
        strategy = response.choices[0].message.content.strip().upper()
        # Clean up any extra words the LLM might add
        if "STRIPE_DOCS" in strategy: return "STRIPE_DOCS"
        return "CHAT"

    def check_long_term_memory(self, query_embedding: List[float]) -> str:
        """MEMORY RETRIEVAL: Checks if we have learned this before."""
        results = self.memory_col.query(query_embeddings=[query_embedding], n_results=1)
        if results['distances'] and results['distances'][0] and results['distances'][0][0] < 0.15: 
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

    def rerank_context(self, query: str, docs_with_sources: List[Dict]) -> str:
        """RE-RANKING: Finds best matches and formats citations."""
        if not docs_with_sources:
            return ""
            
        logger.info(f"🎯 Re-Ranking {len(docs_with_sources)} documents for maximum precision...")
        
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
        raw_parent_ids = [meta['parent_ref'] for meta in results['metadatas'][0]]
        unique_parent_ids = list(set(raw_parent_ids)) 
        
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
            system_prompt = "You are a helpful AI Assistant."

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
        query_emb = self.embedder.model.encode([user_query], normalize_embeddings=True)[0].tolist()
        
        learned_answer = self.check_long_term_memory(query_emb)
        if learned_answer:
            return learned_answer

        strategy = self.determine_strategy(user_query)
        
        # We pass the raw user_query now so the Re-Ranker can use the actual text
        context = self.retrieve_context(user_query, query_emb) if strategy == "STRIPE_DOCS" else None
        
        current_answer = self.generate_response(user_query, context)

        if strategy == "STRIPE_DOCS":
            for i in range(2):
                score, improved_answer = self.reflect_and_score(user_query, context, current_answer)
                logger.info(f"📊 Quality Score: {score}/10")
                if score >= 9:
                    self.store_in_memory(user_query, improved_answer, query_emb)
                    current_answer = improved_answer
                    break
                current_answer = improved_answer
        
        self.history.append({"role": "user", "content": user_query})
        self.history.append({"role": "assistant", "content": current_answer})
        return current_answer

if __name__ == "__main__":
    DB_PATH = r"D:\project dataset\RAG project\chromadb"
    
    print("--- 🔐 SECURE AGENT LOGIN ---")
    input_id = input("Enter User ID: ")
    input_pw = input("Enter Password: ")

    try:
        # Initialize agent with security
        agent = AgenticStripeScout(db_path=DB_PATH, user_id=input_id, password=input_pw)
        print(f"\n✅ Connection established for {input_id}. Workspace loaded.\n")
        
        while True:
            user_input = input("\n👤 User: ")
            if user_input.lower() in ['exit', 'quit']: break
            print(f"\n🤖 Agent: {agent.chat(user_input)}")
            
    except Exception as e:
        print(f"\n❌ Access Denied: {str(e)}")
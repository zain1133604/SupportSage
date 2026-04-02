import os
import logging
from typing import List, Dict, Any
from groq import Groq  # Grok models are often accessed via Groq or X.AI API
import chromadb
from embedding import EmbeddingEngine # Reuse your safe local engine
from dotenv import load_dotenv

load_dotenv()

# --- PROFESSIONAL LOGGING ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class ReflectiveRAGAgent:
    def __init__(self, db_path: str):
        # 1. Connect to your local Intelligence Core
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        self.parent_col = self.chroma_client.get_collection("parent_chunks")
        self.child_col = self.chroma_client.get_collection("child_chunks")
        
        # 2. Local Embedding Engine (RTX 3060 Ti)
        self.embedder = EmbeddingEngine()
        
        # 3. The Brain (Grok via Groq/X.AI)
        self.llm = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model_name = "llama-3.3-70b-versatile" # Or your preferred Grok/Llama model
        
        # 4. Simple Session Memory
        self.history = []

    def retrieve_context(self, query: str, top_k: int = 10) -> str:
        """Professional Semantic Search: Search Children -> Get Unique Parent Context"""
        logger.info(f"🔍 Searching database for: {query}")
        
        # Use your local GPU to embed the query
        query_embedding = self.embedder.model.encode([query], normalize_embeddings=True)[0].tolist()
        
        # 1. Find the most relevant 'Child' chunks
        results = self.child_col.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        # 2. Extract Parent IDs and REMOVE DUPLICATES using set()
        # This prevents the DuplicateIDError
        raw_parent_ids = [meta['parent_ref'] for meta in results['metadatas'][0]]
        unique_parent_ids = list(set(raw_parent_ids)) 
        
        logger.info(f"🎯 Found {len(raw_parent_ids)} child matches across {len(unique_parent_ids)} parent documents.")

        # 3. Fetch the unique Parent docs
        parent_docs = self.parent_col.get(ids=unique_parent_ids)
        
        context = "\n---\n".join(parent_docs['documents'])
        return context

    def generate_answer(self, query: str, context: str) -> str:
        """The initial draft generation."""
        system_prompt = f"""
        You are a Professional Technical Assistant. Use the provided context to answer the user.
        If the answer is not in the context, say you don't know. 
        Context: {context}
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            *self.history[-4:], # Include last 2 turns of history
            {"role": "user", "content": query}
        ]
        
        response = self.llm.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.2
        )
        return response.choices[0].message.content

    def reflect_and_correct(self, query: str, context: str, initial_answer: str) -> str:
        """The Reflection Loop: Check for hallucinations and quality."""
        logger.info("🧠 Agent is reflecting on the answer...")
        
        reflection_prompt = f"""
        Review the following AI response for accuracy based ONLY on the context provided.
        Query: {query}
        Context: {context}
        Initial Answer: {initial_answer}

        CRITIQUE RULES:
        1. Ensure the technical event names (like webhooks) match the context exactly.
        2. If the initial answer is perfect, return ONLY the initial answer text.
        3. If not, provide the CORRECTED version.
        4. DO NOT include meta-commentary like "The answer is supported" or "Here is a corrected version."
        5. YOUR OUTPUT MUST BE ONLY THE FINAL RESPONSE FOR THE USER.
        """
        
        response = self.llm.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "system", "content": reflection_prompt}],
            temperature=0.1
        )
        return response.choices[0].message.content

    def chat(self, user_query: str):
        # Step 1: Semantic Search
        context = self.retrieve_context(user_query)
        
        # Step 2: Initial Draft
        current_answer = self.generate_answer(user_query, context)
        
        # --- THE REFLECTION LOOP ---
        max_retries = 3
        iteration = 0
        is_perfect = False

        while not is_perfect and iteration < max_retries:
            iteration += 1
            logger.info(f"🧠 Reflection Loop: Iteration {iteration}/{max_retries}")
            
            # Ask the LLM to critique and potentially fix the answer
            reflection_result = self.reflect_and_correct(user_query, context, current_answer)
            
            # Logic: If the reflection_result is identical to current_answer, 
            # it means the model thinks the answer is now "Perfect".
            if reflection_result.strip() == current_answer.strip():
                logger.info("✅ Reflection complete: Answer verified as accurate.")
                is_perfect = True
            else:
                logger.warning("⚠️ Correction identified. Updating answer and re-reflecting...")
                current_answer = reflection_result
        
        # Final Step: Update History
        self.history.append({"role": "user", "content": user_query})
        self.history.append({"role": "assistant", "content": current_answer})
        
        return current_answer

# --- RUN IT ---
if __name__ == "__main__":
    DB_PATH = r"D:\project dataset\RAG project\chromadb"
    agent = ReflectiveRAGAgent(db_path=DB_PATH)
    
    while True:
        user_input = input("\n👤 User: ")
        if user_input.lower() in ['exit', 'quit']: break
        
        response = agent.chat(user_input)
        print(f"\n🤖 Agent: {response}")
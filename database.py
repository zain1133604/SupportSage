import logging
import chromadb
import os
import hashlib
import json
import uuid
from typing import List, Dict
from dotenv import load_dotenv

# Import your custom classes
from chunking import AscendedRAGPipeline
from embedding import EmbeddingEngine

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ChromaVectorDB:
    def __init__(self, persist_dir: str = "./chroma_db"):
        self.persist_dir = persist_dir
        # 1. Initialize Client
        self.client = chromadb.PersistentClient(path=persist_dir)
        
        # 2. Setup Auth Registry
        self.auth_file = os.path.join(persist_dir, "user_registry.json")
        self._load_registry()

    def _load_registry(self):
        if os.path.exists(self.auth_file):
            with open(self.auth_file, "r") as f:
                self.registry = json.load(f)
        else:
            self.registry = {}

    def register_user(self, user_id: str, password: str) -> bool:
        if user_id in self.registry:
            logger.error(f"User {user_id} already exists!")
            return False
        hashed_pw = hashlib.sha256(password.encode()).hexdigest()
        self.registry[user_id] = hashed_pw
        with open(self.auth_file, "w") as f:
            json.dump(self.registry, f)
        logger.info(f"User {user_id} registered.")
        return True

    def authenticate(self, user_id: str, password: str):
        hashed_pw = hashlib.sha256(password.encode()).hexdigest()
        if self.registry.get(user_id) != hashed_pw:
            raise Exception("Authentication Failed: Invalid ID or Password")
        return True

    # --- YOUR ORIGINAL ROBUST INSERTION LOGIC (Updated for Multi-Tenancy) ---

    def _prepare_batch(self, records: List[Dict]):
        ids, embeddings, documents, metadatas = [], [], [], []
        for rec in records:
            rec_id = str(rec.get("id") or rec["metadata"].get("parent_ref") or uuid.uuid4().hex)
            ids.append(rec_id)
            embeddings.append(rec.get("embedding"))
            documents.append(rec.get("text", ""))
            metadatas.append(rec.get("metadata", {}))
        return ids, embeddings, documents, metadatas

    def insert_user_data(self, user_id: str, embedded_data: Dict):
        """Modified version of your insert functions to target user collections"""
        parent_col = self.client.get_or_create_collection(name=f"{user_id}_parents")
        child_col = self.client.get_or_create_collection(name=f"{user_id}_children")

        # Insert Parents
        p_ids, p_embs, p_docs, p_metas = self._prepare_batch(embedded_data["parents"])
        batch_size = 500
        for i in range(0, len(p_ids), batch_size):
            parent_col.add(ids=p_ids[i:i+batch_size], embeddings=p_embs[i:i+batch_size], 
                           documents=p_docs[i:i+batch_size], metadatas=p_metas[i:i+batch_size])
        
        # Insert Children
        c_ids, c_embs, c_docs, c_metas = self._prepare_batch(embedded_data["children"])
        # Generate unique child IDs like your original code
        final_child_ids = [f"c_{i}_{uuid.uuid4().hex[:4]}" for i in range(len(c_ids))]
        for i in range(0, len(final_child_ids), batch_size):
            child_col.add(ids=final_child_ids[i:i+batch_size], embeddings=c_embs[i:i+batch_size], 
                          documents=c_docs[i:i+batch_size], metadatas=c_metas[i:i+batch_size])
        
        logger.info(f"✅ Data successfully isolated for user: {user_id}")

# --- UPDATED MAIN EXECUTION ---
if __name__ == "__main__":
    DATA_PATH = r"D:\project-dataset\RAG-project\stripe-data_clean"
    DB_PATH = r"D:\project dataset\RAG project\chromadb"

    db = ChromaVectorDB(persist_dir=DB_PATH)

    # 1. Setup User Identity
    print("--- USER SETUP ---")
    u_id = input("Enter Unique User ID: ")
    u_pw = input("Set Password: ")
    
    if not db.register_user(u_id, u_pw):
        # If user exists, ask for password to verify before adding more data
        try:
            db.authenticate(u_id, u_pw)
            print("User verified. Adding new data to your existing workspace...")
        except Exception as e:
            print(e)
            exit()

    # 2. Run your original Pipeline
    pipeline = AscendedRAGPipeline(base_path=DATA_PATH)
    parents, children = pipeline.process()

    embedder = EmbeddingEngine(model_name="BAAI/bge-m3", batch_size=4) 
    embedded_data = embedder.split_parent_child(parents, children)

    # 3. Store in the USER-SPECIFIC collections
    db.insert_user_data(u_id, embedded_data)
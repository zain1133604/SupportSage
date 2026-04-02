import logging
import chromadb
import os
from typing import List, Dict
from dotenv import load_dotenv
import uuid

# Import your custom classes
from chunking import AscendedRAGPipeline
from embedding import EmbeddingEngine

load_dotenv()

# --- LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ChromaVectorDB:
    def __init__(self, persist_dir: str = "./chroma_db"):
        # ✅ Modern Chroma Persistent Client
        self.client = chromadb.PersistentClient(path=persist_dir)

        # Separate collections for the Parent-Child (Hierarchical) RAG
        self.parent_collection = self.client.get_or_create_collection(name="parent_chunks")
        self.child_collection = self.client.get_or_create_collection(name="child_chunks")

        logger.info(f"ChromaDB initialized at {persist_dir}")

    def _prepare_batch(self, records: List[Dict]):
        ids, embeddings, documents, metadatas = [], [], [], []

        for i, rec in enumerate(records):
            # 1. ID Safety: Chroma REQUIRES strings
            rec_id = str(rec.get("id") or rec["metadata"].get("parent_ref") or uuid.uuid4().hex)
            ids.append(rec_id)
            
            # 2. Embedding Safety: Already converted in embedding.py, but let's be sure
            emb = rec.get("embedding")
            embeddings.append(emb)
            
            # 3. Text and Metadata
            documents.append(rec.get("text", ""))
            metadatas.append(rec.get("metadata", {}))

        return ids, embeddings, documents, metadatas

    def insert_parents(self, parent_records: List[Dict]):
        if not parent_records: return
        logger.info(f"Storing {len(parent_records)} Parents...")
        ids, embeddings, docs, metas = self._prepare_batch(parent_records)
        
        # Chroma handles better in chunks of 500
        batch_size = 500
        for i in range(0, len(ids), batch_size):
            self.parent_collection.add(
                ids=ids[i:i+batch_size],
                embeddings=embeddings[i:i+batch_size],
                documents=docs[i:i+batch_size],
                metadatas=metas[i:i+batch_size]
            )

    def insert_children(self, child_records: List[Dict]):
        if not child_records: return
        logger.info(f"Storing {len(child_records)} Children...")
        ids, embeddings, docs, metas = self._prepare_batch(child_records)
        
        # Unique child IDs with parent reference
        child_ids = [f"c_{i}_{uuid.uuid4().hex[:4]}" for i in range(len(ids))]
        
        batch_size = 500
        for i in range(0, len(child_ids), batch_size):
            self.child_collection.add(
                ids=child_ids[i:i+batch_size],
                embeddings=embeddings[i:i+batch_size],
                documents=docs[i:i+batch_size],
                metadatas=metas[i:i+batch_size]
            )

    def load_stats(self):
        p_count = self.parent_collection.count()
        c_count = self.child_collection.count()
        logger.info(f"DATABASE STATS -> Parents: {p_count} | Children: {c_count}")
        return p_count, c_count

# --- MAIN PIPELINE EXECUTION ---
if __name__ == "__main__":
    # 1. Configuration (Paths)
    # Using raw strings to avoid Windows path errors
    DATA_PATH = r"D:\project-dataset\RAG-project\stripe-data_clean"
    DB_PATH = r"D:\project dataset\RAG project\chromadb"

    # 2. Stage 1: Semantic Chunking (Local NLTK + BGE-M3)
    logger.info("--- STAGE 1: CHUNKING ---")
    # REMOVED API_KEY because your new AscendedRAGPipeline doesn't take it!
    pipeline = AscendedRAGPipeline(base_path=DATA_PATH)
    parents, children = pipeline.process()

    # 3. Stage 2: Local Embedding (RTX 3060 Ti Power)
    logger.info("--- STAGE 2: EMBEDDING ---")
    # REMOVED API_KEY because EmbeddingEngine is now LOCAL!
    embedder = EmbeddingEngine(model_name="BAAI/bge-m3", batch_size=4) 
    embedded_data = embedder.split_parent_child(parents, children)

    # 4. Stage 3: Database Storage
    logger.info("--- STAGE 3: STORAGE ---")
    db = ChromaVectorDB(persist_dir=DB_PATH)
    
    db.insert_parents(embedded_data["parents"])
    db.insert_children(embedded_data["children"])

    db.load_stats()
    logger.info("PIPELINE FULLY COMPLETE. LOCAL RAG IS READY.")
import torch
import logging
import numpy as np
from typing import List, Dict, Any
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document
import uuid
import gc

# --- LOGGING CONFIGURATION ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EmbeddingEngine:
    """
    High-performance Local Embedding Engine utilizing BAAI/bge-m3.
    Optimized for NVIDIA RTX 3060 Ti (CUDA) for unlimited local inference.
    """
    def __init__(self, model_name: str = "BAAI/bge-m3", batch_size: int = 32):
        self.model_name = model_name
        self.batch_size = batch_size
        self.dimensions = 1024  # BGE-M3 default
        
        # 1. Device Detection (RTX 3060 Ti Check)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing EmbeddingEngine on device: {self.device.upper()}")
        if self.device == "cuda":
            logger.info(f"GPU Detected: {torch.cuda.get_device_name(0)}")

        # 2. Load Model locally
        try:
            self.model = SentenceTransformer(self.model_name, device=self.device)
            logger.info(f"Successfully loaded {self.model_name} (1024-dim)")
        except Exception as e:
            logger.error(f"Failed to load local model: {e}")
            raise
    def embed_documents(self, docs: List[Document], show_progress: bool = True) -> List[Dict[str, Any]]:
            if not docs:
                return []

            import time 
            logger.info(f"SAFE-MODE: Using 'Pulse-Embedding' to prevent PSU spikes.")
            records = []
            
            # Disable gradients to save VRAM atemps low
            with torch.no_grad():
                for doc in tqdm(docs, desc="Pulse-Encoding", disable=not show_progress):
                    try:
                        time.sleep(0.02) 

                        # SINGLE INFERENCE
                        emb = self.model.encode(
                            [doc.page_content], 
                            batch_size=1, 
                            show_progress_bar=False,
                            convert_to_numpy=True,
                            normalize_embeddings=True 
                        )

                        records.append({
                            "id": doc.metadata.get("id") or doc.metadata.get("parent_ref") or str(uuid.uuid4()),
                            "embedding": emb[0].tolist(),
                            "text": doc.page_content,
                            "metadata": doc.metadata
                        })

                        if self.device == "cuda":
                            torch.cuda.synchronize() 
                            torch.cuda.empty_cache()

                        time.sleep(0.01)

                    except Exception as e:
                        logger.error(f"Inference failed: {e}")
                        continue 

            logger.info(f"Pulse-Encoding complete. PC survived.")
            return records
        
    def split_parent_child(self, parents: List[Document], children: List[Document]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Wrapper to process both parents and children through the pulse-encoder.
        """
        logger.info("Starting Dual-Stage Embedding (Parents & Children)")
        
        # 1. Embed Parents
        logger.info("--- Processing Parent Documents ---")
        parent_records = self.embed_documents(parents, show_progress=True)
        
        # 2. Embed Children
        logger.info("--- Processing Child Documents ---")
        child_records = self.embed_documents(children, show_progress=True)
        
        return {
            "parents": parent_records,
            "children": child_records
        } # Fixed the syntax here
import os
import hashlib
import logging
import uuid
import re
import torch
import numpy as np
import nltk
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import gc
import time

# Essential for robust sentence splitting
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

from langchain_community.document_loaders import (
    PyPDFLoader, UnstructuredMarkdownLoader, PythonLoader, TextLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# --- PROFESSIONAL LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()] # Remove the FileHandler for now
)
logger = logging.getLogger(__name__)

class AscendedRAGPipeline:
    def __init__(self, base_path: str):
            self.base_path = base_path
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Hardware identified: {self.device.upper()}")

            self.model_name = "BAAI/bge-m3"
            logger.info(f"Loading Local Intelligence: {self.model_name}...")
            
            # Load once, with trust_remote_code for BGE-M3 specifics
            self.model = SentenceTransformer(self.model_name, device=self.device, trust_remote_code=True)
            
            self.seen_file_hashes = set()
            self.seen_chunk_hashes = set() 
            self.stats = {"files": 0, "parents": 0, "children": 0, "deduped": 0, "filtered": 0}

    def _is_high_quality(self, text: str) -> bool:
        """The Quality Gate to filter out noise."""
        t = text.strip()
        if len(t) < 150: return False
        if len(set(t)) < 15: return False 
        return True

    def _classify_content(self, text: str) -> str:
        """Regex-based classification for the metadata."""
        if re.search(r'(def\s+\w+\(|class\s+\w+:|import\s+\w+|{\s*".*":\s*".*")', text):
            return "technical_code"
        if text.strip().startswith(('#', '##', '###')):
            return "structural_header"
        if re.search(r'(\d+\.\s+|[•\-\*]\s+)', text):
            return "list_data"
        return "narrative_prose"

    def custom_semantic_split(self, text: str) -> List[str]:
            # 1. HARD LIMIT & CLEANUP
            if len(text) > 20000: 
                text = text[:20000]

            sentences = nltk.sent_tokenize(text)
            if len(sentences) < 5: 
                return [text]
                
            # Limit sentences for PSU safety
            if len(sentences) > 30: 
                sentences = sentences[:30]

            # 2. THE "PSU SAVER" EMBEDDING LOOP
            embeddings = []
            for sent in sentences:
                time.sleep(0.05) # Pulse breathing
                emb = self.model.encode(
                    [sent], 
                    batch_size=1, 
                    show_progress_bar=False, 
                    convert_to_numpy=True
                )
                embeddings.append(emb[0])
                
                if self.device == "cuda":
                    torch.cuda.empty_cache()

            embeddings = np.array(embeddings)
            
            # 3. THE GROUPING LOGIC (The missing piece!)
            distances = []
            for i in range(len(embeddings) - 1):
                # Measure similarity between sentence i and i+1
                similarity = cosine_similarity([embeddings[i]], [embeddings[i+1]])[0][0]
                distances.append(similarity)

            # We split where similarity is low (meaning a topic change)
            # Using 85th percentile as a threshold for "different enough"
            if not distances:
                return [text]
                
            threshold = np.percentile(distances, 15) # 15th percentile of similarity = big jump
            
            chunks = []
            current_chunk = [sentences[0]]
            
            for i, dist in enumerate(distances):
                if dist < threshold:
                    # Similarity is low -> Create new chunk
                    chunks.append(" ".join(current_chunk))
                    current_chunk = [sentences[i+1]]
                else:
                    current_chunk.append(sentences[i+1])
            
            chunks.append(" ".join(current_chunk))

            # 4. FINAL RETURN (Crucial to prevent 'NoneType' error)
            return [c for c in chunks if len(c.strip()) > 10]

        # ... (rest of your cosine similarity logic)
    def process(self):
        loaders = {
            ".md": TextLoader, 
            ".pdf": PyPDFLoader, 
            ".py": PythonLoader, 
            ".txt": TextLoader,
            ".csv": TextLoader  # Add this!
        }
        final_parents, final_children = [], []
        child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)

        for root, _, files in os.walk(self.base_path):
            company = os.path.basename(root)
            for file in tqdm(files, desc=f"Ingesting {company}"):
                ext = os.path.splitext(file)[1].lower()
                if ext not in loaders: continue
                path = os.path.join(root, file)
                
                try:
                    with open(path, 'rb') as f:
                        file_data = f.read()
                        file_hash = hashlib.md5(file_data).hexdigest()
                    
                    if file_hash in self.seen_file_hashes: 
                        continue
                    self.seen_file_hashes.add(file_hash)

                    content = file_data.decode('utf-8', errors='ignore')

                    # 2. SPECIAL LOGIC FOR CSV
                    if ext == ".csv":
                        # We add a header hint so the AI knows what the columns are
                        content = f"CSV Data from {file}:\n" + content

                    raw_docs = [Document(page_content=content, metadata={"source": company, "file": file})]

                    for doc in raw_docs:
                        semantic_parents = self.custom_semantic_split(doc.page_content)
                        
                        for group_text in semantic_parents:
                            # 3. LOOSEN QUALITY GATE FOR CODE AND CSV
                            # If it's a .py or .csv file, we allow shorter text (50 chars instead of 150)
                            min_len = 50 if ext in [".py", ".csv"] else 150
                            
                            if len(group_text.strip()) < min_len:
                                self.stats["filtered"] += 1
                                continue
                    # 3. MANUAL DOCUMENT CREATION
                    # This replaces: raw_docs = loaders[ext](path).load()
                    # We skip the heavy LangChain loaders which are crashing your PC
                    raw_docs = [Document(page_content=content, metadata={"source": company, "file": file})]

                    for doc in raw_docs:
                        # This calls your custom_semantic_split which handles the 50k crop
                        semantic_parents = self.custom_semantic_split(doc.page_content)
                        
                        for group_text in semantic_parents:
                            if not self._is_high_quality(group_text):
                                self.stats["filtered"] += 1
                                continue
                            
                            parent_id = f"p-{uuid.uuid4().hex[:8]}"
                            final_parents.append(Document(
                                page_content=group_text, 
                                metadata={"id": parent_id, "source": company, "file": file}
                            ))

                            children_texts = child_splitter.split_text(group_text)
                            for j, c_text in enumerate(children_texts):
                                final_children.append(Document(
                                    page_content=c_text, 
                                    metadata={"parent_ref": parent_id, "source": company}
                                ))
                    
                    self.stats["files"] += 1

                    # --- THE SAFETY BRAKES ---
                    del file_data
                    del content
                    del raw_docs 
                    gc.collect() 
                    time.sleep(0.1)
                    if self.device == "cuda":
                        torch.cuda.empty_cache()

                except Exception as e:
                    logger.error(f"Hard Failure on {file}: {str(e)}")
            
        return final_parents, final_children




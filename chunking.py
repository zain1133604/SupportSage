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
    # we need punkt so we can have  highend splitting of sentence.if we don't use it we have to rely on the these signs(.,?). it is not that good as punkt splitting.
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
    handlers=[logging.StreamHandler()] # we are using the streamhandler to show the logs in the console or terminal. we can also use the filehandler. this way our logs will be saved on the .txt or .log files.
)
logger = logging.getLogger(__name__)

class AscendedRAGPipeline:
    def __init__(self, base_path: str):
            self.base_path = base_path
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Hardware identified: {self.device.upper()}")

            self.model_name = "BAAI/bge-m3"
            logger.info(f"Loading Local Intelligence: {self.model_name}...")
            
            # Load once, with trust_remote_code for BGE-M3 specifics. here we are loading the model in our memory.
            self.model = SentenceTransformer(self.model_name, device=self.device, trust_remote_code=True)# the last variable means here that we trust BAAI/bge-m3 creators. this way our python will not show any error.
            
            self.seen_file_hashes = set() # we are doing this cause we want every data/file to be unique. we don't want any thing duplicate in our data/file.
            self.seen_chunk_hashes = set() # In two different files we can have the same data.so we are setting this so even after the chunked if we have the same data coming again from the different file we should be able to filter it. so at the end we don't have duplicate data.
            self.stats = {"files": 0, "parents": 0, "children": 0, "deduped": 0, "filtered": 0} # Metrics to track processing progress and data quality results

    def _is_high_quality(self, text: str) -> bool:
        """The Quality Gate to filter out noise."""
        t = text.strip() # first we are striping all the grabage out like extra spaces at the end.
        if len(t) < 150: return False # Reject very short or incomplete text
        if len(set(t)) < 15: return False # Reject low-diversity text (e.g., repetitive or noisy content)
        return True

    def _classify_content(self, text: str) -> str:
        """ -Regex-based classification for the metadata.
            -Uses Regex patterns to tag text type (Code, Header, List, or Prose) for smarter retrieval metadata."""
        if re.search(r'(def\s+\w+\(|class\s+\w+:|import\s+\w+|{\s*".*":\s*".*")', text): # checks for code.
            return "technical_code" 
        if text.strip().startswith(('#', '##', '###')): # check for headers
            return "structural_header"
        if re.search(r'(\d+\.\s+|[•\-\*]\s+)', text): # check for list of lines
            return "list_data"
        return "narrative_prose" # if not anthing then its a normal data.

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
                    batch_size=1, # to avoid power cut.
                    show_progress_bar=False, 
                    convert_to_numpy=True # converting tensors to numpy array to clear the gpu memory.
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

        # ... (rest of our cosine similarity logic)
    def process(self):
        loaders = {
            ".md": TextLoader, 
            ".pdf": PyPDFLoader, 
            ".py": PythonLoader, 
            ".txt": TextLoader,
            ".csv": TextLoader  
        }
        final_parents, final_children = [], []
        child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)

        for root, _, files in os.walk(self.base_path): # walk through every folder and file.
            company = os.path.basename(root) # saving name of comany.
            for file in tqdm(files, desc=f"Ingesting {company}"):
                ext = os.path.splitext(file)[1].lower()
                if ext not in loaders: continue
                path = os.path.join(root, file)
                
                try:
                    with open(path, 'rb') as f:
                        file_data = f.read() # reading file
                        file_hash = hashlib.md5(file_data).hexdigest() # putting unique key to each file.
                    
                    if file_hash in self.seen_file_hashes: # if duplicate files come skip it.
                        continue
                    self.seen_file_hashes.add(file_hash)

                    content = file_data.decode('utf-8', errors='ignore')

                    # 2. SPECIAL LOGIC FOR CSV
                    if ext == ".csv":
                        # We add a header hint so the AI knows what the columns are
                        content = f"CSV Data from {file}:\n" + content

                    # --- CLEANED LOGIC ---
                    # 1. Create the doc once
                    raw_docs = [Document(page_content=content, metadata={"source": company, "file": file})]

                    for doc in raw_docs:
                        # 2. Run the HEAVY semantic split ONLY ONCE
                        semantic_parents = self.custom_semantic_split(doc.page_content)
                        
                        for group_text in semantic_parents:
                            # 3. Combined Quality Gate
                            # Check specific length for code/csv
                            min_len = 50 if ext in [".py", ".csv"] else 150
                            
                            # Run both checks: Basic Length AND the "Variety Gate" (_is_high_quality)
                            if len(group_text.strip()) < min_len or not self._is_high_quality(group_text):
                                self.stats["filtered"] += 1
                                continue
                            
                            # 4. If it passes, create the Parent and Children
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




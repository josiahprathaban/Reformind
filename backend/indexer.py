# indexer.py
"""
This module handles parsing the KJV Bible text and creating a vector index
using llama_index and HuggingFaceEmbedding.
"""

import os
import re
from typing import List, Dict, Any
import logging

from llama_index.core import (
    VectorStoreIndex, 
    Document,
    StorageContext,
    load_index_from_storage,
    Settings
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Constants
INDEX_STORAGE_PATH = os.environ.get("INDEX_STORAGE_PATH", "./index")
KJV_TEXT_PATH = os.environ.get("KJV_TEXT_PATH", "./data/kjv.txt")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def parse_bible_verses(file_path: str) -> List[Document]:
    """
    Parse the KJV Bible text file into Document objects.
    
    Args:
        file_path: Path to the KJV text file
        
    Returns:
        List of Document objects, each representing a Bible verse
    """
    logger.info(f"Parsing Bible verses from: {file_path}")
    documents = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                # Extract reference and text using regex
                match = re.match(r'([^0-9]+) (\d+):(\d+)\t(.*)', line.strip())
                if match:
                    book, chapter, verse, text = match.groups()
                    
                    # Create metadata for the verse
                    metadata = {
                        "book": book.strip(),
                        "chapter": int(chapter),
                        "verse": int(verse),
                        "reference": f"{book.strip()} {chapter}:{verse}"
                    }
                    
                    # Create a Document object
                    doc = Document(text=text.strip(), metadata=metadata)
                    documents.append(doc)
                else:
                    logger.warning(f"Could not parse line: {line}")
        
        logger.info(f"Successfully parsed {len(documents)} Bible verses")
        return documents
    
    except Exception as e:
        logger.error(f"Error parsing Bible verses: {str(e)}")
        raise

def create_and_persist_index(documents: List[Document]) -> VectorStoreIndex:
    """
    Create a vector index from the documents and persist it to disk.
    
    Args:
        documents: List of Document objects
        
    Returns:
        VectorStoreIndex object
    """
    logger.info("Creating vector index...")
    
    try:
        # Create embedding model
        embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL_NAME)
        
        # Configure settings with the embedding model
        Settings.embed_model = embed_model
        
        # Create and persist the index
        index = VectorStoreIndex.from_documents(documents)
        
        # Persist index to disk
        if not os.path.exists(INDEX_STORAGE_PATH):
            os.makedirs(INDEX_STORAGE_PATH)
        
        index.storage_context.persist(persist_dir=INDEX_STORAGE_PATH)
        logger.info(f"Index created and persisted to: {INDEX_STORAGE_PATH}")
        
        return index
    
    except Exception as e:
        logger.error(f"Error creating index: {str(e)}")
        raise

def load_or_create_index() -> VectorStoreIndex:
    """
    Load the index from disk if it exists, otherwise create a new one.
    
    Returns:
        VectorStoreIndex object
    """
    try:
        # Check if index exists
        if os.path.exists(INDEX_STORAGE_PATH):
            logger.info(f"Loading existing index from: {INDEX_STORAGE_PATH}")
            
            # Create embedding model
            embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL_NAME)
            
            # Configure settings with the embedding model
            Settings.embed_model = embed_model
            
            # Load index from disk
            storage_context = StorageContext.from_defaults(persist_dir=INDEX_STORAGE_PATH)
            index = load_index_from_storage(
                storage_context=storage_context
            )
            
            logger.info("Index loaded successfully")
            return index
        else:
            logger.info("No existing index found, creating new index...")
            
            # Parse Bible verses
            documents = parse_bible_verses(KJV_TEXT_PATH)
            
            # Create and persist index
            return create_and_persist_index(documents)
    
    except Exception as e:
        logger.error(f"Error loading or creating index: {str(e)}")
        raise

if __name__ == "__main__":
    # When run directly, create/update the index
    load_or_create_index()
    logger.info("Indexing complete")

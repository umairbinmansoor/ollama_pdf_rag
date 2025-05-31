"""Vector embeddings and database functionality."""
import logging
from typing import List
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS

logger = logging.getLogger(__name__)

class VectorStore:
    """Manages vector embeddings and database operations."""
    
    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": "cpu"},  # Use CPU for Streamlit Cloud compatibility
            encode_kwargs={"normalize_embeddings": True}
        )
        self.vector_db = None
    
    # def create_vector_db(self, documents: List, collection_name: str = "local-rag") -> Chroma:
    def create_vector_db(self, documents: List, collection_name: str = "local-rag") -> FAISS:
        """Create vector database from documents."""
        try:
            logger.info("Creating vector database")
            self.vector_db = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings,
                # collection_name=collection_name,
                # persist_directory="/tmp/chroma"  # Persist to disk for Streamlit Cloud
            )
            return self.vector_db
        except Exception as e:
            logger.error(f"Error creating vector database: {e}")
            raise
    
    def delete_collection(self) -> None:
        """Delete vector database collection."""
        if self.vector_db:
            try:
                logger.info("Deleting vector database collection")
                self.vector_db.delete_collection()
                self.vector_db = None
            except Exception as e:
                logger.error(f"Error deleting collection: {e}")
                raise
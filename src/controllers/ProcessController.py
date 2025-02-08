from langchain.embeddings import CacheBackedEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.storage import LocalFileStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.controllers import BaseController
from src.models.enums import ExtensionEnum
from src.helpers.config import get_settings
import hashlib
import os

class ProcessController(BaseController):

    def __init__(self):
        super().__init__()
        self.settings = get_settings()

    def chunk(self, docs, chunk_size=1000, chunk_overlap=50):
        
        # Initialize the text splitter with the specified chunk size and overlap
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        
        chunks = text_splitter.split_documents(docs)        
        return chunks


    def create_vector_index_and_embedding_model(self, chunks):
        """
        Creates an embedding model and vector index using Langchain and embeddings model.
        
        This function takes a list of text chunks (documents), creates an embedding model using 
        embeddings model, and then uses FAISS (a vector store) to create a vector index that allows for 
        efficient similarity search. The function includes caching.
        
        Args:
            chunks (list): List of Document objects that need to be indexed. 
        
        Returns:
            tuple: Returns a tuple containing:
                - embeddings_model: The embedding model used for encoding the text.
                - vector_index: The FAISS index that stores the vectors of the documents for fast retrieval.
        """
        
        # Step 1: Set up local storage for cached embeddings
        cache_dir = self.settings.CACHE_DIR
        store = LocalFileStore(cache_dir)
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize the the embedding model
        embeddings_model = OllamaEmbeddings(model=self.settings.EMBEDDING_MODEL_ID)

        # # Step 3: Create cache-backed embeddings with safe key encoding
        # namespace = hashlib.sha256(self.settings.EMBEDDING_MODEL_ID.encode()).hexdigest()
        # embedder = CacheBackedEmbeddings.from_bytes_store(
        #     embeddings_model,
        #     store,
        #     namespace=namespace
        # )
        
        # # Step 4: Check for existing vector store
        # vector_store_path = self.settings.VECTOR_DB_PATH
        # if os.path.exists(vector_store_path):
        #     try:
        #         vector_index = FAISS.load_local(
        #             vector_store_path,
        #             embedder,
        #             allow_dangerous_deserialization=True
        #         )
        #         return embeddings_model, vector_index
        #     except Exception as e:
        #         print(f"Error loading vector store: {e}")

        vector_index = FAISS.from_documents(chunks, embeddings_model)
        
        # # Step 7: Save vector store for future use
        # os.makedirs(vector_store_path, exist_ok=True)
        # vector_index.save_local(vector_store_path)
        
        return embeddings_model, vector_index
    
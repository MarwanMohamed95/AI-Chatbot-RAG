from langchain.embeddings import CacheBackedEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.storage import LocalFileStore
from langchain.text_splitter import NLTKTextSplitter
from src.controllers import BaseController
from src.models.enums import ExtensionEnum
from src.helpers.config import get_settings
import hashlib
import os
from tqdm import tqdm

import nltk
# Download necessary NLTK components for text tokenization
nltk.download('punkt_tab')
nltk.download('punkt')

class ProcessController(BaseController):

    def __init__(self):
        super().__init__()
        self.settings = get_settings()

    def chunk(self, file_path, docs, chunk_size=1000, chunk_overlap=50):
        
        # Extract text content from Document objects
        file_extension = BaseController().get_file_extension(file_path=file_path)
        # if file_extension == ExtensionEnum.CSV.value:
        #     texts = [",".join(doc.page_content) for doc in docs if hasattr(doc, "page_content")]
        
        # Initialize the text splitter with the specified chunk size and overlap
        text_splitter = NLTKTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        
        chunks = text_splitter.split_documents(docs)
        
        # Log the number of generated chunks for debugging
        return chunks


    def create_vector_index_and_embedding_model(self, chunks):
        """
        Creates an embedding model and vector index using Langchain and Ollama embeddings.
        
        This function takes a list of text chunks (documents), creates an embedding model using 
        Ollama, and then uses FAISS (a vector store) to create a vector index that allows for 
        efficient similarity search. The function includes caching and batch processing for 
        better performance.
        
        Args:
            chunks (list): List of Document objects that need to be indexed. 
                        Each chunk should have a page_content attribute containing the text.
        
        Returns:
            tuple: Returns a tuple containing:
                - embeddings_model: The Ollama embedding model used for encoding the text.
                - vector_index: The FAISS index that stores the vectors of the documents for fast retrieval.
        
        Notes:
            - Uses LocalFileStore for caching embeddings to prevent redundant computations
            - Processes documents in batches for better memory management
            - Includes progress tracking for longer operations
            - Safely handles index creation and storage
        """
        
        # Step 1: Set up local storage for cached embeddings
        cache_dir = get_settings().CACHE_DIR
        store = LocalFileStore(cache_dir)
        os.makedirs(cache_dir, exist_ok=True)
        
        # Step 2: Initialize the Ollama embedding model
        embeddings_model = OllamaEmbeddings(model=self.settings.EMBEDDING_MODEL_ID)
        
        # Step 3: Create cache-backed embeddings with safe key encoding
        namespace = hashlib.sha256(self.settings.EMBEDDING_MODEL_ID.encode()).hexdigest()
        embedder = CacheBackedEmbeddings.from_bytes_store(
            embeddings_model,
            store,
            namespace=namespace
        )
        
        # Step 4: Check for existing vector store
        vector_store_path = get_settings().VECTOR_DB_PATH
        if os.path.exists(vector_store_path):
            try:
                vector_index = FAISS.load_local(
                    vector_store_path,
                    embedder,
                    allow_dangerous_deserialization=True
                )
                return embeddings_model, vector_index
            except Exception as e:
                print(f"Error loading vector store: {e}")

        
        # Step 5: Process documents in batches to create vector store
        batch_size = 10
        all_embeddings = []
        all_texts = []
        
        for i in tqdm(range(0, len(chunks), batch_size), desc="Processing documents"):
            batch = chunks[i:i + batch_size]
            texts = [doc.page_content for doc in batch]
            all_texts.extend(texts)
            embeddings = embedder.embed_documents(texts)
            all_embeddings.extend(embeddings)
        
        # Step 6: Create FAISS index from processed embeddings
        vector_index = FAISS.from_embeddings(
            text_embeddings=list(zip(all_texts, all_embeddings)),
            embedding=embedder
        )
        
        # Step 7: Save vector store for future use
        os.makedirs(vector_store_path, exist_ok=True)
        vector_index.save_local(vector_store_path)
        
        return embeddings_model, vector_index
    
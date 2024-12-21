from langchain.embeddings import CacheBackedEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.storage import LocalFileStore
from langchain.text_splitter import NLTKTextSplitter
from src.controllers import BaseController
from src.helpers.config import get_settings

import nltk
# Download necessary NLTK components for text tokenization
nltk.download('punkt_tab')
nltk.download('punkt')

class ProcessController(BaseController):

    def __init__(self):
        super().__init__()
        self.settings = get_settings()

    def chunk(self, docs, chunk_size=1000, chunk_overlap=50):
        """
        Splits the documents into smaller chunks using NLTK-based text splitting.
        
        This method processes the text content from the provided Document objects and splits them into smaller
        chunks with specified chunk size and overlap using the `NLTKTextSplitter`.
        
        Args:
            docs (list): A list of Document objects containing the text to be split.
            chunk_size (int): The maximum size of each text chunk (default is 1000).
            chunk_overlap (int): The number of overlapping characters between consecutive chunks (default is 50).
        
        Returns:
            list: A list of Document objects representing the text chunks.
        """
        # Extract text content from Document objects
        texts = [doc.page_content for doc in docs if hasattr(doc, "page_content")]
        
        # Initialize the text splitter with the specified chunk size and overlap
        text_splitter = NLTKTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        
        # Split the text into chunks and collect them in a list
        chunks = []
        for text in texts:
            chunks.extend(text_splitter.create_documents([text]))
        
        # Log the number of generated chunks for debugging
        return chunks

    def create_vector_index_and_embedding_model(self, chunks):
        """
        Creates an embedding model and vector index using Langchain and Hugging Face embeddings.
        
        This function takes a list of text chunks (documents), creates an embedding model using 
        Hugging Face's pre-trained `e5-small-v2` model, and then uses FAISS (a vector store) to 
        create a vector index that allows for efficient similarity search.
        
        Args:
            chunks (list of str): List of text chunks (documents) that need to be indexed. 
                                Each chunk is a string representing a document or text segment.
        
        Returns:
            tuple: Returns a tuple containing:
                - embeddings_model: The Hugging Face embedding model used for encoding the text.
                - vector_index: The FAISS index that stores the vectors of the documents for fast retrieval.
        
        Notes:
            - The `LocalFileStore` is used to store embeddings and cache them for future use.
            - The embeddings are cached using the `CacheBackedEmbeddings` wrapper to prevent redundant computations.
            - This function uses a CPU-based model, which can be adjusted for GPU usage if needed.
        """
        
        # Step 1: Set up local storage for cached embeddings
        store = LocalFileStore("./assets/cache/")  # Directory where embeddings will be cached
        
        # Step 2: Define the embedding model ID and model arguments (device, remote code trust)
        embed_model_id = self.settings.EMBEDDING_MODEL_ID  # Hugging Face model ID for embeddings
        model_kwargs = {"device": self.settings.DEVICE, "trust_remote_code": True}  # Parameters for the model (use CPU)
        
        # Step 3: Create the Hugging Face embeddings model using the specified model ID and arguments
        embeddings_model = HuggingFaceEmbeddings(model_name=embed_model_id, model_kwargs=model_kwargs)
        
        # Step 4: Wrap the Hugging Face model with caching to avoid recalculating embeddings
        embedder = CacheBackedEmbeddings.from_bytes_store(embeddings_model, store, namespace=embed_model_id)
        
        # Step 5: Use FAISS to create a vector index from the documents (chunks)
        vector_index = FAISS.from_documents(chunks, embedder)  # Generate the index based on the document embeddings
        
        # Return the embeddings model and the FAISS vector index for later use
        return embeddings_model, vector_index

        
from langchain.embeddings import CacheBackedEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.storage import LocalFileStore

def create_vector_index_and_embedding_model(chunks):
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
    store = LocalFileStore("./cache/")  # Directory where embeddings will be cached
    
    # Step 2: Define the embedding model ID and model arguments (device, remote code trust)
    embed_model_id = 'intfloat/e5-small-v2'  # Hugging Face model ID for embeddings
    model_kwargs = {"device": "cpu", "trust_remote_code": True}  # Parameters for the model (use CPU)
    
    # Step 3: Create the Hugging Face embeddings model using the specified model ID and arguments
    embeddings_model = HuggingFaceEmbeddings(model_name=embed_model_id, model_kwargs=model_kwargs)
    
    # Step 4: Wrap the Hugging Face model with caching to avoid recalculating embeddings
    embedder = CacheBackedEmbeddings.from_bytes_store(embeddings_model, store, namespace=embed_model_id)
    
    # Step 5: Use FAISS to create a vector index from the documents (chunks)
    vector_index = FAISS.from_documents(chunks, embedder)  # Generate the index based on the document embeddings
    
    # Return the embeddings model and the FAISS vector index for later use
    return embeddings_model, vector_index

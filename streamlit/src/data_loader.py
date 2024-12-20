from langchain_community.document_loaders import TextLoader, PyPDFLoader, WebBaseLoader, CSVLoader, UnstructuredHTMLLoader
from langchain.text_splitter import NLTKTextSplitter
import nltk

# Download necessary NLTK components for text tokenization
nltk.download('punkt_tab')
nltk.download('punkt')

def read_txt(file_path):
    """
    Load and parse the content of a plain text (.txt) file.
    
    Args:
        file_path (str): The path to the text file to be loaded.
    
    Returns:
        list: A list of Document objects containing the file's content.
    """
    loader = TextLoader(file_path)
    docs = loader.load()
    return docs

def read_pdf(file_path):
    """
    Load and parse the content of a PDF file.
    
    Args:
        file_path (str): The path to the PDF file to be loaded.
    
    Returns:
        list: A list of Document objects containing the file's content.
    """
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    return docs

def read_csv(file_path):
    """
    Load and parse the content of a CSV file.
    
    Args:
        file_path (str): The path to the CSV file to be loaded.
    
    Returns:
        list: A list of Document objects containing the file's content.
    """
    loader = CSVLoader(file_path)
    docs = loader.load()
    return docs

def read_html(file_path):
    """
    Load and parse the content of an HTML file.
    
    Args:
        file_path (str): The path to the HTML file to be loaded.
    
    Returns:
        list: A list of Document objects containing the file's content.
    """
    loader = UnstructuredHTMLLoader(file_path)
    docs = loader.load()
    return docs

def read_webpage(page_url):
    """
    Load and parse the content of a webpage.
    
    Args:
        page_url (str): The URL of the webpage to be loaded.
    
    Returns:
        list: A list of Document objects containing the webpage's content.
    """
    loader = WebBaseLoader(web_paths=[page_url])
    docs = []
    
    # Asynchronously load and append webpage content
    for doc in loader.alazy_load():
        docs.append(doc)
    return docs

def chunk(docs, chunk_size=1000, chunk_overlap=50):
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
    print(f"Split into {len(chunks)} chunks")
    return chunks

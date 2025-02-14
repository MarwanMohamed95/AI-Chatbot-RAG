from .BaseController import BaseController
from fastapi import UploadFile
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from src.models.enums import ExtensionEnum, ResponseEnums
from src.controllers import DataController
from src.helpers.config import get_settings
import re
import ftfy
import nltk
import unicodedata

class DataController(BaseController):
    
    def __init__(self):
        super().__init__()
        self.size_scale = 1048576 # convert MB to bytes
        self.app_settings = get_settings()

    def validate_uploaded_file(self, file: UploadFile):

        if file.content_type not in self.app_settings.FILE_ALLOWED_TYPES:
            return False, ResponseEnums.FILE_TYPE_NOT_SUPPORTED.value

        if file.size > self.app_settings.FILE_MAX_SIZE * self.size_scale:
            return False, ResponseEnums.FILE_SIZE_EXCEEDED.value

        return True, ResponseEnums.FILE_VALIDATED_SUCCESS.value

    def read_file(self, file_path):
        
        # validate the file properties
        file_extension = BaseController().get_file_extension(file_path=file_path)

        if file_extension == ExtensionEnum.TXT.value:
            loader = TextLoader(file_path)
            docs = loader.load()
        
        elif file_extension == ExtensionEnum.PDF.value:
            loader = PyPDFLoader(file_path)
            docs = loader.load()
        
        return docs

    def merge_documents(self, docs):
        return " ".join([doc.page_content for doc in docs])

    def clean_text(self, text: str) -> str:
        """
        Cleans raw extracted text before chunking for RAG applications.
        """
        # Fix encoding issues (e.g., replace broken characters)
        text = ftfy.fix_text(text)

        # Normalize Unicode (e.g., convert accented chars to base form)
        text = unicodedata.normalize("NFKC", text)

        # Remove page numbers, headers, and footers
        text = re.sub(r"\bPage \d+\b", "", text)  # Remove "Page 1, Page 2..."
        text = re.sub(r"(Copyright|All rights reserved|Terms of use).*", "", text, flags=re.IGNORECASE)

        # Remove URLs & emails
        text = re.sub(r"http[s]?://\S+", "", text)  # URLs
        text = re.sub(r"\S+@\S+", "", text)  # Emails

        # Remove Roman numerals (I, II, III, IV, etc.)
        text = re.sub(r"\bM{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})\b", "", text, flags=re.IGNORECASE)

        # Normalize whitespace & punctuation
        text = re.sub(r"\s+", " ", text)  # Replace multiple spaces with a single space
        text = re.sub(r"([?.!,;:])\1+", r"\1", text)  # Reduce repeated punctuation (e.g., "!!!" -> "!")

        # Remove special characters (except important ones)
        text = re.sub(r"[^\w\s.,!?\"'()\-]", "", text)  

        # Convert to lowercase (optional)
        text = text.lower()

        return text.strip()
    
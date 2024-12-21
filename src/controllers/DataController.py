from .BaseController import BaseController
from fastapi import UploadFile
from langchain_community.document_loaders import TextLoader, PyPDFLoader, WebBaseLoader, CSVLoader, UnstructuredHTMLLoader
from src.models.enums import ExtensionEnum, ResponseEnums
from src.controllers import DataController
from src.helpers.config import get_settings

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

        elif file_extension == ExtensionEnum.CSV.value:
            loader = CSVLoader(file_path)
            docs = loader.load()

        elif file_extension == ExtensionEnum.HTML.value:
            loader = UnstructuredHTMLLoader(file_path)
            docs = loader.load()
        
        return docs

    def read_webpage(self, page_url):
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

from fastapi import APIRouter, UploadFile, File, Depends, status
from fastapi.responses import JSONResponse
import os
import aiofiles
from langchain.schema import Document
from src.models.enums import ResponseEnums, DataEnum
from src.helpers.config import get_settings, Settings
from src.controllers import DataController, ProcessController
from src.helpers.session_manager import get_session

# Initialize router
upload_process_router = APIRouter(prefix="/api/v1")

# Initialize controllers
data_controller = DataController()
data_processor = ProcessController()

@upload_process_router.post("/upload-and-process/")
async def upload_and_process(file: UploadFile = File(...),
                             app_settings: Settings = Depends(get_settings)):
    """
    Uploads a file, validates it, extracts text, cleans it, chunks it, and 
    stores it in a vector index in one go.
    """

    # Create a temporary directory for file uploads
    TEMP_DIR = app_settings.TEMP_DIR
    os.makedirs(TEMP_DIR, exist_ok=True)

    # Validate uploaded file
    is_valid, result_signal = data_controller.validate_uploaded_file(file=file)
    if not is_valid:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"signal": result_signal}
        )

    # Define file path
    file_path = os.path.join(TEMP_DIR, file.filename)

    # Save file asynchronously
    try:
        async with aiofiles.open(file_path, "wb") as f:
            while chunk := await file.read(app_settings.FILE_DEFAULT_CHUNK_SIZE):
                await f.write(chunk)
    except Exception:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"signal": ResponseEnums.FILE_UPLOAD_FAILED.value}
        )

    # Read and process file
    try:
        docs = data_controller.read_file(file_path=file_path)
        if not docs:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"signal": ResponseEnums.LOADING_FILE_FAILED.value}
            )

        # Clean text
        text = data_controller.merge_documents(docs)
        text = data_controller.clean_text(text)
        document = Document(page_content=text)

        # Chunk text
        chunks = data_processor.chunk([document], chunk_size=DataEnum.CHUNK_SIZE.value, 
                                      chunk_overlap=DataEnum.OVERLAP_SIZE.value)

        if not chunks:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"signal": ResponseEnums.PROCESSING_FAILED.value}
            )

        # Generate embeddings and vector index
        embedding_model, vector_index = data_processor.create_vector_index_and_embedding_model(chunks)

        # Create retriever
        retriever = vector_index.as_retriever(search_type="similarity", search_kwargs={"k": 3})

        # Store retriever in session
        session_id = app_settings.SESSION_ID
        session = get_session(session_id)
        session['retriever'] = retriever

        return JSONResponse(
            content={
                "signal": ResponseEnums.PROCESSING_SUCCESS.value,
                "file_name": file.filename,
                "inserted_chunks": len(chunks),
                "session_id": session_id
            }
        )

    except Exception:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"signal": ResponseEnums.PROCESSING_FAILED.value}
        )

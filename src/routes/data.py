from fastapi import APIRouter, UploadFile, File, Form, Depends, status
from fastapi.responses import JSONResponse
import os
import aiofiles
from src.models.enums import ResponseEnums, DataEnum
from src.helpers.config import get_settings, Settings
from src.controllers import DataController, ProcessController
from src.helpers.session_manager import get_session
from langchain.schema import Document

uploader_router = APIRouter(
    prefix="/api/v1",
)

# validate the file properties
data_controller = DataController()

@uploader_router.post("/upload-file/")
async def upload_file(file: UploadFile = File(...),
                      app_settings: Settings = Depends(get_settings)):
    
    # Create a temporary directory for file uploads
    TEMP_DIR = app_settings.TEMP_DIR
    os.makedirs(TEMP_DIR, exist_ok=True)

    is_valid, result_signal = data_controller.validate_uploaded_file(file=file)

    if not is_valid:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "signal": result_signal
            }
        )

    file_path = os.path.join(TEMP_DIR, file.filename)

    try:
        async with aiofiles.open(file_path, "wb") as f:
            while chunk := await file.read(app_settings.FILE_DEFAULT_CHUNK_SIZE):
                await f.write(chunk)
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "signal": ResponseEnums.FILE_UPLOAD_FAILED.value
            }
        )

    return JSONResponse(
            content={
                "signal": ResponseEnums.FILE_UPLOAD_SUCCESS.value,
                "file_name": file_path,
            }
        )

processor_router = APIRouter(
    prefix="/api/v1",
)

@processor_router.post("/process-file/")
async def process_file(file_path: str,
                       app_settings: Settings = Depends(get_settings)):
    
    session_id = app_settings.SESSION_ID
    data_processor = ProcessController()
    try:
        
        docs = data_controller.read_file(file_path=file_path)

        if docs is None or len(docs) == 0:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "signal": ResponseEnums.LOADING_FILE_FAILED.value
                }
            )
        
        text = data_controller.merge_documents(docs)
        text = data_controller.clean_text(text)
        document = Document(page_content=text)

        chunks = data_processor.chunk([document], chunk_size=DataEnum.CHUNK_SIZE.value, 
                                      chunk_overlap=DataEnum.OVERLAP_SIZE.value)

        if chunks is None or len(chunks) == 0:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "signal": ResponseEnums.PROCESSING_FAILED.value
                }
            )

        embedding_model, vector_index = data_processor.create_vector_index_and_embedding_model(chunks)

        retriever = vector_index.as_retriever(search_type="similarity", search_kwargs={"k": 5})

        session = get_session(session_id)
        session['retriever'] = retriever

        return JSONResponse(
            content={
                "signal": ResponseEnums.PROCESSING_SUCCESS.value,
                "inserted_chunks": len(chunks),
                "session_id": session_id
            }
        )
    except Exception as e:
        return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "signal": ResponseEnums.PROCESSING_FAILED.value
                }
            )

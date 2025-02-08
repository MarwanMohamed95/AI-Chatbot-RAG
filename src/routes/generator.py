from fastapi import APIRouter, Form, Depends, status
from fastapi.responses import JSONResponse, StreamingResponse
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from src.controllers import LLMController
from src.helpers.session_manager import get_session
from src.helpers.config import get_settings, Settings
from src.models.enums import ResponseEnums
from typing import AsyncGenerator

llm_controller = LLMController()

generator_router = APIRouter(
    prefix="/api/v1",
)

@generator_router.post("/generate-answer/")
async def generate_answer(
    question: str = Form(...),
    app_settings: Settings = Depends(get_settings)
) -> StreamingResponse:
    try:
        session_id = app_settings.SESSION_ID
        session = get_session(session_id)

        if 'retriever' not in session:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "signal": ResponseEnums.VECTORDB_RETRIEVED_FAILED.value
                }
            )
        
        retriever = session['retriever']
        model_name = app_settings.GENERATION_MODEL_ID

        if 'chat_history' not in session:
            session['chat_history'] = ChatMessageHistory()

        chat_history = session['chat_history']

        summerizer_chain = llm_controller.create_context_aware_chain(
            retriever, 
            model_name,
            app_settings.SUMMERIZATION_TEMPERATURE
        )
        
        retriever_answer_chain = llm_controller.create_answering_chain(
            model_name,
            summerizer_chain,
            app_settings.GENERATION_TEMPERATURE
        )

        final_chain = RunnableWithMessageHistory(
            retriever_answer_chain,
            lambda x: ChatMessageHistory(messages=chat_history.messages[-4:]),
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

        query = f"Answer the following question in summary: {question}"

        async def stream_response() -> AsyncGenerator[str, None]:
            accumulated_answer = ""
            try:
                async for chunk in final_chain.astream(
                    {"input": query},
                    config={"configurable": {"session_id": session_id}}
                ):
                    if isinstance(chunk, dict) and "answer" in chunk:
                        token = chunk["answer"]
                    else:
                        token = str(chunk)

                    accumulated_answer += token
                    # Format as proper SSE
                    yield f"{token}\n\n"
                
                # Send a completion signal
                yield "[DONE]\n\n"
                
            except Exception as e:
                yield f"Error: {str(e)}\n\n"
                yield "[DONE]\n\n"

            # Update chat history after streaming
            chat_history.add_user_message(query)
            chat_history.add_ai_message(accumulated_answer)

        return StreamingResponse(
            stream_response(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream",
            }
        )

    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "signal": ResponseEnums.RAG_ANSWER_ERROR.value,
                "error": str(e)
            }
        )
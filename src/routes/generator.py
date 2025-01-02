# generate_answer.py
from fastapi import APIRouter, Form, Depends, status
from fastapi.responses import JSONResponse
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from src.controllers import LLMController
from src.helpers.session_manager import get_session
from src.helpers.config import get_settings, Settings
from src.models.enums import ResponseEnums

llm_controller = LLMController()

generator_router = APIRouter(
    prefix="/api/v1",
)

@generator_router.post("/generate-answer/")
async def generate_answer(session_id: str = Form(...), user_query: str = Form(...),
                          app_settings: Settings = Depends(get_settings)):
    try:
        session = get_session(session_id)

        if 'retriever' not in session:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "signal": ResponseEnums.VECTORDB_RETRIEVED_FAILED.value
                }
            )
        
        retriever = session['retriever']
        ollama_model_name = app_settings.GENERATION_MODEL_ID

        if 'chat_history' not in session:
            session['chat_history'] = ChatMessageHistory()

        chat_history = session['chat_history']

        retriever_chain = llm_controller.create_context_aware_chain(retriever, ollama_model_name,
                                                                    app_settings.SUMMERIZATION_TEMPERATURE)
        
        retriever_answer_chain = llm_controller.create_answering_chain(
            ollama_model_name,
            retriever_chain,
            app_settings.GENERATION_TEMPERATURE
        )

        final_chain = RunnableWithMessageHistory(
            retriever_answer_chain,
            lambda x: chat_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

        response = final_chain.invoke(
            {"input": user_query},
            config={"configurable": {"session_id": session_id}}
        )

        if isinstance(response, dict) and "answer" in response:
            answer = response["answer"]
        else:
            answer = response

        chat_history.add_user_message(user_query)
        chat_history.add_ai_message(answer)

        return JSONResponse(
            content={
                "answer": answer,
                "signal": ResponseEnums.RAG_ANSWER_SUCCESS.value
            }
        )
    
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "signal": ResponseEnums.RAG_ANSWER_ERROR.value
            }
        )

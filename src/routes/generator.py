# generate_answer.py
from fastapi import APIRouter, Form
from fastapi.responses import JSONResponse
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from src.controllers import LLMController
from src.helpers.session_manager import get_session

llm_controller = LLMController()

generator_router = APIRouter(
    prefix="/api/v1",
)

@generator_router.post("/generate-answer/")
async def generate_answer(session_id: str = Form(...), user_query: str = Form(...)):
    try:
        session = get_session(session_id)

        if 'retriever' not in session:
            return JSONResponse(status_code=400, content={"detail": "Retriever not found in session"})

        retriever = session['retriever']
        ollama_model_name = "llama3.2:3b"

        if 'chat_history' not in session:
            session['chat_history'] = ChatMessageHistory()

        chat_history = session['chat_history']

        retriever_chain = llm_controller.create_context_aware_chain(retriever, ollama_model_name)
        retriever_answer_chain = llm_controller.create_answering_chain(
            ollama_model_name,
            retriever_chain,
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

        return {"answer": answer}
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})

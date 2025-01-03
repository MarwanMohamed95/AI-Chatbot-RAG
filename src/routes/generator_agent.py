from fastapi import APIRouter, Form, Depends, status
from fastapi.responses import JSONResponse
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools.base import ToolException
from langchain.tools import tool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from src.controllers import LLMController
from src.helpers.session_manager import get_session
from src.helpers.config import get_settings, Settings
from src.models.enums import ResponseEnums
from typing import Dict, Union

llm_controller = LLMController()

generator_router = APIRouter(
    prefix="/api/v1",
)

@generator_router.post("/generate-answer/")
async def generate_answer(user_query: str = Form(...),
                        app_settings: Settings = Depends(get_settings)):
    # try:
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
        ollama_model_name = app_settings.GENERATION_MODEL_ID

        if 'chat_history' not in session:
            session['chat_history'] = ChatMessageHistory()

        chat_history = session['chat_history']

        # Create retriever tools instance
        @tool
        def search_documents(query: Union[str, Dict[str, str]]) -> str:
            """Use this tool to search through documents for relevant information."""
            try:
                # Create a context-aware query using chat history and current query
                if chat_history.messages:
                    # Convert chat history to a format suitable for the history-aware retriever
                    history_messages = []
                    for msg in chat_history.messages[-4:]:  # Use last 4 messages for context
                        if msg.type == "human":
                            history_messages.append(HumanMessage(content=msg.content))
                        else:
                            history_messages.append(AIMessage(content=msg.content))
                    
                    # Use the history-aware retriever
                    context = summarizer_chain.invoke({
                        "chat_history": history_messages,
                        "input": query if isinstance(query, str) else query['query']
                    })
                    
                    # Extract text content from the context if it's a Document object or list of Documents
                    if isinstance(context, list):
                        if all(hasattr(doc, 'page_content') for doc in context):
                            search_query = ' '.join(doc.page_content for doc in context)
                        else:
                            search_query = ' '.join(str(doc) for doc in context)
                    elif hasattr(context, 'page_content'):
                        search_query = context.page_content
                    else:
                        search_query = str(context)
                else:
                    search_query = query if isinstance(query, str) else query['query']
                
                response = retriever.invoke(search_query)
                return response
            except Exception as e:
                raise ToolException(f"Error searching documents: {str(e)}")

        # Create chat model
        llm = llm_controller.create_chat_model(
            ollama_model_name,
            temperature=app_settings.GENERATION_TEMPERATURE
        )

        # Create the summarizer chain
        summarizer_chain = llm_controller.create_context_aware_chain(
            retriever,
            ollama_model_name,
            app_settings.SUMMERIZATION_TEMPERATURE
        )

        # Create prompt with enhanced context handling
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful AI assistant. Use the search_documents tool to find relevant information from the documents to answer questions accurately.
             Consider the chat history when formulating your responses to maintain context and coherence.
             Don't include the query in the final answer.
             Don't mention that the results based on searching in the history"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        # Create the agent
        agent = create_tool_calling_agent(llm, tools=[search_documents], prompt=prompt)

        # Create agent executor
        agent_executor = AgentExecutor(
            agent=agent,
            tools=[search_documents],
            handle_parsing_errors=True,
            max_iterations=3
        )

        # Wrap with message history
        final_chain = RunnableWithMessageHistory(
            agent_executor,
            lambda x: chat_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="output"
        )

        # Format chat history for the chain
        formatted_history = [
            HumanMessage(content=msg.content) if msg.type == "human" 
            else AIMessage(content=msg.content)
            for msg in chat_history.messages[-4:]  # Use last 4 messages for context
        ]

        # Generate response
        response = final_chain.invoke(
            {
                "input": user_query,
                "chat_history": formatted_history,
            },
            config={"configurable": {"session_id": session_id}}
        )

        if isinstance(response, dict) and "output" in response:
            answer = response["output"]
        else:
            answer = response

        # Update chat history
        chat_history.add_user_message(user_query)
        chat_history.add_ai_message(answer)

        return JSONResponse(
            content={
                "answer": answer,
                "signal": ResponseEnums.RAG_ANSWER_SUCCESS.value
            }
        )
    
    # except Exception as e:
    #     print(f"Error in generate_answer: {str(e)}")
    #     return JSONResponse(
    #         status_code=status.HTTP_400_BAD_REQUEST,
    #         content={
    #             "signal": ResponseEnums.RAG_ANSWER_ERROR.value
    #         }
    #     )
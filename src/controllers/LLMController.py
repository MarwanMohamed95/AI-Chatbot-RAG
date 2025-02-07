from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_ollama import ChatOllama 
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools.base import ToolException
from langchain.tools import tool
from src.helpers.config import get_settings
from src.helpers.session_manager import get_session
from src.controllers import BaseController
from typing import Dict, Union

class LLMController(BaseController):

    def __init__(self):
        super().__init__()
        self.settings = get_settings()

    def create_chat_model(self, model_name, temperature):
        return ChatOllama(model=model_name, temperature=temperature, base_url='http://localhost:11434')

    def create_context_aware_chain(self, retriever, model_name, temperature):
        """
        Creates a context-aware chain that uses history to help the user to get the answer, considering both the latest 
        query and the previous chat history. This chain ensures that the latest user question..
        
        This function initializes an Ollama model for query rephrasing and combines it with a history-aware 
        retriever to answer the query based on the chat history. The goal is to make the query more 
        concise, clear, and independent of the ongoing conversation while retaining its meaning.

        Args:
            retriever (object): The retriever that will fetch context from a knowledge base or past chat history.
            model_name (str): The name of the Ollama model used for rephrasing the query and handling the context.

        Returns:
            history_aware_retriever (object): The context-aware retriever that reformulates the user's query 
                                            considering the chat history.

        """
        
        # Step 1: Initialize the Ollama model with the provided model name and configuration
        llm_summarise = self.create_chat_model(model_name=model_name, temperature=temperature)
        # Step 2: Define the system prompt to guide the model in answering the user question
        prompt = hub.pull("langchain-ai/chat-langchain-rephrase")

        
        # Step 3: Create a prompt template using the system prompt and placeholders for chat history and user input
        template = ChatPromptTemplate.from_messages([("system", prompt.template), 
                                                    MessagesPlaceholder("chat_history"), 
                                                    ("human", "{input}")])

        # Step 4: Create a history-aware retriever using the LLM (Ollama) and the prompt defined above
        # The retriever will handle querying the knowledge base or past chat history to fetch relevant context.
        history_aware_retriever = create_history_aware_retriever(llm_summarise, retriever, template)
        
        # Return the context-aware retriever, which can be used for querying
        return history_aware_retriever

    def create_answering_chain(self, model_name, retriever_chain, temperature):
        """
        Creates a question-answering chain using Ollama as the language model.
        
        Args:
            model_name (str): The name of the Ollama model to be used for generating answers.
            retriever_chain: The retriever chain that provides context-aware retrieval.
        
        Returns:
            chain: The question-answering chain that incorporates context-aware retrieval.
        """

        llm_answer = self.create_chat_model(model_name=model_name, temperature=temperature)

        # Define the system prompt
        system_prompt = """You are an AI assistant that answer user queries accurately.
            Give the answer to the user query based on the context provided.
            Give the answer directly without mentioning the context.
            Give the answer in summary.
            If you don't know the answer just say I don't know the answer.
            Context information is below:
            {context}"""

        # Create prompt template for the question-answering task
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])

        # Create the full chain
        chain = (
            {
                "context": retriever_chain,
                "input": RunnablePassthrough()
            }
            | qa_prompt
            | llm_answer
            | StrOutputParser()
        )

        return chain

    def create_agent_executer(self, model_name, temperature):
        
        SESSION_ID = "1234" 
        session = get_session(SESSION_ID)
        retriever = session["retriever"]

        @tool
        def search_documents(query: Union[str, Dict[str, str]]) -> str:
            """Use this tool to search through documents for relevant information."""
            try:
                response = retriever.invoke(query)
                return response
            except Exception as e:
                raise ToolException(f"Error searching documents: {str(e)}")
            
        tools = [search_documents]

        llm = self.create_chat_model(model_name=model_name, temperature=temperature)

        # Define the system prompt
        system_prompt = """You are an AI assistant that answer user queries accurately.
            Don't mention that you are using previous context but give the answer directly.
            If you don't know the answer just say I don't know the answer.
            """

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        agent = create_tool_calling_agent(llm, tools, prompt)

        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            handle_parsing_errors=True,
            max_iterations=3
        )

        return agent_executor



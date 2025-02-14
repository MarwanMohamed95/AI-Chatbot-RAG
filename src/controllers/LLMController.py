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
        
        This function initializes a model for query rephrasing and combines it with a history-aware 
        retriever to answer the query based on the chat history. The goal is to make the query more 
        concise, clear, and independent of the ongoing conversation while retaining its meaning.

        Args:
            retriever (object): The retriever that will fetch context from a knowledge base or past chat history.
            model_name (str): The name of the model used for rephrasing the query and handling the context.

        Returns:
            history_aware_retriever (object): The context-aware retriever that reformulates the user's query 
                                            considering the chat history.

        """
        # Initialize the answer model with the provided model name and configuration
        llm_summarise = self.create_chat_model(model_name=model_name, temperature=temperature)
        
        # Define the rephrasing prompt to guide the model in rephrasing the history-aware query
        rephrasing_prompt = """Rephrase the user's question to be self-contained, considering the chat history but focusing on the current query. 
        Maintain the original intent while making it standalone.
        Do not include previous answers or explanations in your rephrasing."""

        template = ChatPromptTemplate.from_messages([
            ("system", rephrasing_prompt), 
            MessagesPlaceholder("chat_history"), 
            ("human", "{input}")
        ])

        history_aware_retriever = create_history_aware_retriever(
            llm_summarise, 
            retriever, 
            template
        )
        
        return history_aware_retriever

    def create_answering_chain(self, model_name, retriever_chain, temperature):
        """
        Creates a question-answering chain.
        
        Args:
            model_name (str): The name of the model to be used for generating answers.
            retriever_chain: The retriever chain that provides context-aware retrieval.
        
        Returns:
            chain: The question-answering chain that incorporates context-aware retrieval.
        """
        # Initialize the answer model with the provided model name and configuration
        llm_answer = self.create_chat_model(model_name=model_name, temperature=temperature)

        # Define the system prompt to guide the model in answering the user question
        system_prompt = """You are an intelligent assistant providing responses based on a retrieval-augmented generation system. 
            Always prioritize retrieved content, ensuring responses are accurate, well-structured, and relevant. 
            Give the answer in summary without too much details.
            If the retrieved documents do not contain an answer just say "I don't know the answer". 
            Don't mention that the answer based on provided or previous information.
            """

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

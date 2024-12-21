from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_community.llms.ollama import Ollama
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from src.helpers.config import get_settings
from src.controllers import BaseController

class LLMController(BaseController):

    def __init__(self):
        super().__init__()
        self.settings = get_settings()

    def create_context_aware_chain(self, retriever, model_name, temperature):
        """
        Creates a context-aware chain that reformulates the user’s query, considering both the latest 
        query and the previous chat history. This chain ensures that the latest user question, even if it 
        references prior context in the chat history, is rephrased into a standalone question that can be 
        understood without requiring the full conversation history.
        
        This function initializes an Ollama model for query rephrasing and combines it with a history-aware 
        retriever to reformulate the query based on the chat history. The goal is to make the query more 
        concise, clear, and independent of the ongoing conversation while retaining its meaning.

        Args:
            retriever (object): The retriever that will fetch context from a knowledge base or past chat history.
            model_name (str): The name of the Ollama model used for rephrasing the query and handling the context.

        Returns:
            history_aware_retriever (object): The context-aware retriever that reformulates the user's query 
                                            considering the chat history.

        """
        
        # Step 1: Initialize the Ollama model with the provided model name and configuration
        llm_summarise = Ollama(model=model_name, temperature=temperature, num_predict=256)
        
        # Step 2: Define the system prompt to guide the model in rephrasing the user’s query
        # This prompt explains the task of formulating a query that can stand alone and be understood
        # without requiring the full chat history.
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is.")
        
        # Step 3: Create a prompt template using the system prompt and placeholders for chat history and user input
        contextualize_q_prompt = ChatPromptTemplate.from_messages([("system", contextualize_q_system_prompt),
                                                                MessagesPlaceholder("chat_history"),
                                                                ("human", "{input}")])

        # Step 4: Create a history-aware retriever using the LLM (Ollama) and the prompt defined above
        # The retriever will handle querying the knowledge base or past chat history to fetch relevant context,
        # then the query is reformulated based on the available history.
        history_aware_retriever = create_history_aware_retriever(llm_summarise, retriever, contextualize_q_prompt)
        
        # Return the context-aware retriever, which can be used for querying
        return history_aware_retriever

    def create_qa_RAG_chain_history(self, llm_pipeline, retriever, system_prompt):
        """
        Creates a Retrieval-Augmented Generation (RAG) chain with chat history for conversational AI.
        
        This function integrates a language model pipeline with a history-aware retriever to create a 
        question-answering system that can use the past chat history to provide more contextually relevant answers 
        to user queries. The system prompt contains a placeholder `{context}` that will be filled with the 
        relevant retrieved context during the retrieval process.
        
        Args:
            llm_pipeline (object): The language model pipeline (e.g., Ollama model) used for answering queries.
            retriever (object): A history-aware retriever that fetches relevant context from the past chat history.
            system_prompt (str): A system prompt guiding the model's behavior and tone of responses. 
                                It should contain a `{context}` placeholder to inject the retrieved context.
        
        Returns:
            rag_chain (object): The fully constructed RAG chain that can answer queries using retrieved context 
                                and chat history.
        
        Notes:
            - The system prompt should be designed to instruct the LLM on how to behave when responding, 
            incorporating context from the retrieved chunks.
            - The chain uses a retrieval process combined with the `stuff_documents_chain` for processing the response.
        """
        # Step 1: Create a prompt template with placeholders for system and user inputs, and chat history
        qa_prompt = ChatPromptTemplate.from_messages([("system", system_prompt),
                                                    MessagesPlaceholder("chat_history"),
                                                    ("human", "{input}")])

        # Step 2: Create a chain that combines the LLM with the custom prompt for answering queries
        question_answer_chain = create_stuff_documents_chain(llm_pipeline, qa_prompt)
        
        # Step 3: Create a full RAG chain combining the retriever and question-answering chain
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        return rag_chain

    def create_answering_chain(self, model_name, retriever_chain, temperature):
        """
        Creates a question-answering chain using Ollama as the language model.
        
        Args:
            model_name (str): The name of the Ollama model to be used for generating answers.
            retriever_chain: The retriever chain that provides context-aware retrieval.
        
        Returns:
            chain: The question-answering chain that incorporates context-aware retrieval.
        """
        
        # Initialize the Ollama language model
        llm_answer = Ollama(
            model=model_name,
            temperature=temperature,
        )
        
        # Define the system prompt
        system_prompt = """You are an intelligent and versatile AI assistant designed to assist with a wide variety of tasks. 
            You excel in providing clear, concise, and accurate information, creative ideas, and thoughtful responses 
            tailored to the user's needs. You can engage in casual conversation, help solve problems, explain complex 
            concepts in simple terms, and support the user in learning new skills or making decisions.
            When answering:
            Do not provide standalone questions or the chat history in the response.
            Do not include the context in your response.
            Be polite, empathetic, and professional and answer in short sentences.
            Strive to be as concise as possible while ensuring clarity and completeness.
            If you don't know something or lack enough context, admit it and suggest a way forward.
            Avoid making assumptions unless instructed by the user.
            Always aim to create a positive, helpful, and engaging user experience.
            Do not reformulated standalone questions in the response.
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

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_community.llms.ollama import Ollama

def create_context_aware_chain(retriever, model_name):
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
    llm_summarise = Ollama(model=model_name, temperature=0.0, num_predict=256)
    
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

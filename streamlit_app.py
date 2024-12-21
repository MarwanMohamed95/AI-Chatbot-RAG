import os
import streamlit as st
from src.controllers import DataController, ProcessController, LLMController
import re
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.callbacks.base import BaseCallbackHandler
import ollama
from src.helpers.config import get_settings

SESSION_ID = "1234"
data_controller = DataController()
process_controller = ProcessController()
llm_controller = LLMController()
app_settings = get_settings()

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container: st.delta_generator.DeltaGenerator, initial_text: str = ""):
        self.container = container
        self.text = initial_text
        self.run_id_sequence = set()

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text + "▌")

    def on_llm_end(self, *args, **kwargs) -> None:
        self.container.markdown(self.text)

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state.chat_store:
        st.session_state.chat_store[session_id] = ChatMessageHistory()
    return st.session_state.chat_store[session_id]

def file_added():
    if st.session_state.file_uploaded is None:
        st.error("No file uploaded.")
        return

    uploaded_file = st.session_state.file_uploaded
    temp_dir = "assets/temp"
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    try:
        with st.spinner("Processing file, please wait..."):
            st.session_state.disabled_sum_model = False
            
            docs = data_controller.read_file(file_path=file_path)

            chunks = process_controller.chunk(docs)
            embedding_model, vector_index = process_controller.create_vector_index_and_embedding_model(chunks)
            st.session_state.retriever = vector_index.as_retriever(search_type="similarity", search_kwargs={"k": 3})
            
            st.success(f"File processed successfully, and retriever created.")
    
    except ValueError as e:
        st.error(f"Failed to process the file: {e}")

def update_summarization_model():
    st.session_state.disabled_ans_model = False
    model_name = st.session_state.sum_model
    ollama_model_name = re.search("(.*)  Size:", model_name).group(1)
    st.session_state.retriever_chain = llm_controller.create_context_aware_chain(st.session_state.retriever, ollama_model_name)

def update_answer_model():
    model_name = st.session_state.ans_model
    ollama_model_name = re.search("(.*)  Size:", model_name).group(1)
    
    # Create the base answering chain with streaming enabled
    st.session_state.retriever_answer_chain = llm_controller.create_answering_chain(
        ollama_model_name, 
        st.session_state.retriever_chain,
    )

    st.session_state.final_chain = RunnableWithMessageHistory(
        st.session_state.retriever_answer_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

# Streamlit page configuration
st.set_page_config(page_title="RAG Assistant")
st.title("RAG Assistant")

# Initialize session state variables
if "disabled_ans_model" not in st.session_state:
    st.session_state.disabled_ans_model = True

if "disabled_sum_model" not in st.session_state:
    st.session_state.disabled_sum_model = True

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_store" not in st.session_state:
    st.session_state.chat_store = {}

if "disabled" not in st.session_state:
    st.session_state.disabled = False

# Sidebar setup
with st.sidebar:
    models_ollama = ollama.list()["models"]
    model_name = [m['name'] for m in models_ollama]
    model_size = [float(m["size"]) for m in models_ollama]
    name_detail = zip(model_name, model_size)
    name_detail = sorted(name_detail, key=lambda x: x[1])
    model_info = [f"{name}  Size: {size/(10**9):.2f}GB" for name, size in name_detail]

    uploaded_file = st.file_uploader(
        "Choose a file to upload",
        type=["txt", "csv", "json", "html", "pdf"],
        on_change=file_added,
        key="file_uploaded"
    )

    st.selectbox("Choose a model for context summarization", model_info, index=None, on_change=update_summarization_model, placeholder="Select model", key="sum_model", disabled=st.session_state.disabled_sum_model)
    st.selectbox("Choose a model for answering", model_info, index=None, on_change=update_answer_model, placeholder="Select model", key="ans_model", disabled=st.session_state.disabled_ans_model)

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if user_query := st.chat_input("Message Assistant", disabled=st.session_state.disabled):
    if "retriever" not in st.session_state:
        st.error("Retriever, summarization model and answering model were not set.")
    elif "retriever_chain" not in st.session_state:
        st.error("Summarization model and answering model were not set.")
    elif "retriever_answer_chain" not in st.session_state:
        st.error("Answering model was not set.")
    elif user_query:
        # Add user message to chat
        st.session_state.messages.append({"role": "Human", "content": user_query})
        with st.chat_message("Human"):
            st.markdown(user_query)

        # Generate and stream AI response
        with st.chat_message("AI"):
            message_placeholder = st.empty()
            stream_handler = StreamHandler(message_placeholder)
            
            try:
                response = st.session_state.final_chain.invoke(
                    {"input": user_query},
                    config={
                        "configurable": {"session_id": SESSION_ID},
                        "callbacks": [stream_handler]
                    }
                )
                
                # Add AI response to chat history
                st.session_state.messages.append({"role": "AI", "content": stream_handler.text})
                
                # Update chat store for LangChain
                if SESSION_ID not in st.session_state.chat_store:
                    st.session_state.chat_store[SESSION_ID] = ChatMessageHistory()
                st.session_state.chat_store[SESSION_ID].add_user_message(user_query)
                st.session_state.chat_store[SESSION_ID].add_ai_message(stream_handler.text)
                
            except Exception as e:
                st.error(f"Error during streaming: {str(e)}")
                
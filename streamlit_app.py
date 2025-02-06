import os
import streamlit as st
from src.controllers import DataController, ProcessController, LLMController
import re
from src.models.enums import DataEnum
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
import ollama
from src.helpers.config import get_settings

SESSION_ID = "1234"
data_controller = DataController()
process_controller = ProcessController()
llm_controller = LLMController()
app_settings = get_settings()

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state.chat_store:
        st.session_state.chat_store[session_id] = ChatMessageHistory()
    return st.session_state.chat_store[session_id]

def file_added():
    if st.session_state.file_uploaded is None:
        st.error("No file uploaded.")
        return

    uploaded_file = st.session_state.file_uploaded
    temp_dir = app_settings.TEMP_DIR
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    try:
        with st.spinner("Processing file, please wait..."):
            
            docs = data_controller.read_file(file_path=file_path)

            chunks = process_controller.chunk(docs, chunk_size=DataEnum.CHUNK_SIZE.value, 
                                              chunk_overlap=DataEnum.OVERLAP_SIZE.value)
            embedding_model, vector_index = process_controller.create_vector_index_and_embedding_model(chunks)
            st.session_state.retriever = vector_index.as_retriever(search_type="similarity", search_kwargs={"k": 3})
            
            st.success(f"File processed successfully, and retriever created.")
    
    except ValueError as e:
        st.error(f"Failed to process the file: {e}")

def update_summarization_model():
    model_name = st.session_state.sum_model
    ollama_model_name = re.search("(.*)  Size:", model_name).group(1)
    st.session_state.summerizer_chain = llm_controller.create_context_aware_chain(st.session_state.retriever, ollama_model_name, app_settings.SUMMERIZATION_TEMPERATURE)

def update_answer_model():
    model_name = st.session_state.ans_model
    ollama_model_name = re.search("(.*)  Size:", model_name).group(1)
    
    # Create the base answering chain
    st.session_state.answer_chain = llm_controller.create_answering_chain(
        ollama_model_name, 
        st.session_state.summerizer_chain,
        app_settings.GENERATION_TEMPERATURE
    )

    st.session_state.final_chain = RunnableWithMessageHistory(
        st.session_state.answer_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

def chat_with_llm(session_id, input):
    for output in st.session_state.final_chain.stream({'input': input}, config={'configurable': {'session_id': session_id}}):
        yield output

# Streamlit page configuration
st.set_page_config(page_title="AI Assistant")
st.title("AI Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_store" not in st.session_state:
    st.session_state.chat_store = {}

# Sidebar setup
with st.sidebar:
    models_ollama = ollama.list()["models"]
    model_name = [m['model'] for m in models_ollama]
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

    st.selectbox("Choose a model for context summarization", model_info, index=None, on_change=update_summarization_model, placeholder="Select model", key="sum_model")
    st.selectbox("Choose a model for answering", model_info, index=None, on_change=update_answer_model, placeholder="Select model", key="ans_model")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if user_query := st.chat_input("What is in your mind ?"):
    if "retriever" not in st.session_state:
        st.error("Retriever, summarization model and answering model were not set.")
    elif "summerizer_chain" not in st.session_state:
        st.error("Summarization model and answering model were not set.")
    elif "answer_chain" not in st.session_state:
        st.error("Answering model was not set.")
    elif user_query:
        with st.chat_message("user"):
            st.markdown(user_query)

        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": user_query})

        with st.chat_message("assistant"):
            response = st.write_stream(chat_with_llm(SESSION_ID, user_query))
        
        # Add AI response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Update chat store for LangChain
        if SESSION_ID not in st.session_state.chat_store:
            st.session_state.chat_store[SESSION_ID] = ChatMessageHistory()
        st.session_state.chat_store[SESSION_ID].add_user_message(user_query)
        st.session_state.chat_store[SESSION_ID].add_ai_message(response)
        
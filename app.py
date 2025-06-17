import os
import streamlit as st
import logging
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# --- Basic Setup ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
CHROMA_PERSIST_DIR = "chroma_db_paperpulse_st"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TEMP_UPLOAD_DIR = "temp_uploads"
PROMPT_TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "config", "prompt_template.md")

# --- Core Functions (adapted for Streamlit) ---

@st.cache_resource
def get_embeddings():
    """Load embedding model (cached for performance)."""
    return SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)

def load_and_chunk_documents(uploaded_files, chunk_size=1000, chunk_overlap=200):
    """Loads uploaded PDFs and splits them into chunks."""
    documents = []
    if not os.path.exists(TEMP_UPLOAD_DIR):
        os.makedirs(TEMP_UPLOAD_DIR)

    for uploaded_file in uploaded_files:
        temp_path = os.path.join(TEMP_UPLOAD_DIR, uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        try:
            loader = PyPDFLoader(temp_path)
            docs = loader.load()
            for doc in docs:
                doc.metadata['source'] = uploaded_file.name
            documents.extend(docs)
        except Exception as e:
            st.error(f"Error loading {uploaded_file.name}: {e}")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunked_documents = text_splitter.split_documents(documents)
    return chunked_documents

@st.cache_resource
def get_vectorstore(_documents=None, persist=True):
    """Creates a new Chroma vector store or loads an existing one."""
    embeddings = get_embeddings()
    
    # If documents are provided, create a new vector store
    if _documents:
        with st.spinner("Creating and persisting vector store... This may take a moment."):
            vectorstore = Chroma.from_documents(
                documents=_documents,
                embedding=embeddings,
                persist_directory=CHROMA_PERSIST_DIR if persist else None
            )
        st.success(f"Vector store created and persisted!")
        return vectorstore
    
    # Otherwise, load the existing vector store
    if not os.path.exists(CHROMA_PERSIST_DIR):
        st.warning("No vector store found. Please ingest documents first.")
        return None
    return Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=embeddings)

def get_rag_chain(retriever, groq_api_key):
    """Creates the RAG chain."""
    try:
        llm = ChatGroq(temperature=0, groq_api_key=groq_api_key, model_name="llama3-70b-8192")
    except Exception as e:
        st.error(f"Failed to initialize Groq LLM: {e}")
        return None

    # Load the prompt template
    try:
        with open(PROMPT_TEMPLATE_PATH, "r") as f:
            template_content = f.read()
    except FileNotFoundError:
        st.error(f"Prompt template file not found at {PROMPT_TEMPLATE_PATH}. Please ensure 'config/prompt_template.md' exists.")
        return None # Cannot proceed without template

    prompt = ChatPromptTemplate.from_template(template_content)

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

# --- Streamlit App ---

st.set_page_config(page_title="PaperPulse", page_icon="📄", layout="wide")
st.title("📄 PaperPulse: Your RAG-based Research Assistant")

# --- Sidebar ---
with st.sidebar:
    st.header("Configuration")
    groq_api_key = st.text_input("Groq API Key", type="password", value=os.getenv("GROQ_API_KEY", ""))
    
    st.markdown("---")
    app_mode = st.radio("Choose a Page", ["Ingest Documents", "Chat with Papers"])
    st.markdown("---")
    if st.button("Clear Vector Store"):
        if os.path.exists(CHROMA_PERSIST_DIR):
            import shutil
            shutil.rmtree(CHROMA_PERSIST_DIR)
            st.cache_resource.clear()
            st.success("Vector store cleared.")
        else:
            st.info("No vector store to clear.")

    st.info("PaperPulse helps you chat with your research papers using a powerful RAG pipeline.")

# --- Ingestion Page ---
if app_mode == "Ingest Documents":
    st.header("Ingest Your Research Papers")
    
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type="pdf",
        accept_multiple_files=True,
        help="Upload one or more PDF documents to build your knowledge base."
    )

    if st.button("Process and Ingest Documents") and uploaded_files:
        if not groq_api_key:
            st.error("Please enter your Groq API key in the sidebar.")
        else:
            chunked_docs = load_and_chunk_documents(uploaded_files)
            if chunked_docs:
                # Clear cache before creating a new vector store to ensure it rebuilds
                st.cache_resource.clear()
                get_vectorstore(_documents=chunked_docs, persist=True)
                st.success(f"Successfully ingested {len(uploaded_files)} documents.")
                st.info("You can now go to the 'Chat with Papers' page.")

# --- Chat Page ---
elif app_mode == "Chat with Papers":
    st.header("Chat with Your Papers")

    if not groq_api_key:
        st.error("Please enter your Groq API key in the sidebar to start chatting.")
    else:
        vectorstore = get_vectorstore()
        if vectorstore is None:
            st.warning("The knowledge base is empty. Please go to the 'Ingest Documents' page to add papers.")
        else:
            # Initialize chat history
            if "messages" not in st.session_state:
                st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help you with your research papers?"}]

            # Display chat messages
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # Get user input
            if prompt := st.chat_input("Ask a question about your papers..."):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
                        rag_chain = get_rag_chain(retriever, groq_api_key)
                        if rag_chain:
                            response = rag_chain.invoke(prompt)
                            st.markdown(response)
                            st.session_state.messages.append({"role": "assistant", "content": response})

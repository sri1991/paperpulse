import os
import argparse
import logging
from tqdm import tqdm
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

# --- Configuration ---
CHROMA_PERSIST_DIR = "chroma_db_paperpulse"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
PROMPT_TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "config", "prompt_template.md")

# --- Core Functions ---

def load_and_chunk_documents(data_dir: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> list[Document]:
    """Loads PDFs from a directory, and splits them into chunks."""
    logging.info(f"Loading and chunking documents from {data_dir}")
    documents = []
    for filename in tqdm(os.listdir(data_dir), desc="Processing files"):
        if filename.endswith(".pdf"):
            file_path = os.path.join(data_dir, filename)
            try:
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                for doc in docs:
                    doc.metadata['source'] = filename # Add filename to metadata
                documents.extend(docs)
            except Exception as e:
                logging.error(f"Error loading {filename}: {e}")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunked_documents = text_splitter.split_documents(documents)
    logging.info(f"Loaded and chunked {len(documents)} documents into {len(chunked_documents)} chunks.")
    return chunked_documents

def create_or_get_vectorstore(documents: list[Document] = None, persist: bool = True) -> Chroma:
    """Creates a new Chroma vector store or loads an existing one."""
    embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
    
    if documents:
        logging.info("Creating new vector store...")
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=CHROMA_PERSIST_DIR if persist else None
        )
        if persist:
            logging.info(f"Vector store created and persisted at {CHROMA_PERSIST_DIR}")
    else:
        logging.info(f"Loading existing vector store from {CHROMA_PERSIST_DIR}")
        if not os.path.exists(CHROMA_PERSIST_DIR):
            raise FileNotFoundError(f"Persistence directory not found: {CHROMA_PERSIST_DIR}. Please run the 'ingest' command first.")
        vectorstore = Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=embeddings)
    return vectorstore

def create_rag_chain(retriever):
    """Creates the RAG chain with a prompt template and the Groq LLM."""
    logging.info("Creating RAG chain...")

    # Check for Groq API key
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY environment variable not set!")

    # Create the Groq LLM
    llm = ChatGroq(temperature=0, groq_api_key=groq_api_key, model_name="llama3-70b-8192")

    # Load the prompt template
    try:
        with open(PROMPT_TEMPLATE_PATH, "r") as f:
            template_content = f.read()
    except FileNotFoundError:
        logging.error(f"Prompt template file not found at {PROMPT_TEMPLATE_PATH}")
        raise  # Re-raise the exception as the template is critical

    prompt = ChatPromptTemplate.from_template(template_content)

    # Create the RAG chain
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

# --- Main Execution ---

def main():
    """Main function to handle command-line arguments for ingestion and querying."""
    parser = argparse.ArgumentParser(description="PaperPulse: A RAG-based research assistant.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents into the vector store.")
    ingest_parser.add_argument("datadir", type=str, help="Directory containing PDF documents to ingest.")

    # Query command
    query_parser = subparsers.add_parser("query", help="Start an interactive query session.")

    args = parser.parse_args()

    if args.command == "ingest":
        if not os.path.isdir(args.datadir):
            logging.error(f"Data directory not found: {args.datadir}")
            return
        documents = load_and_chunk_documents(args.datadir)
        create_or_get_vectorstore(documents, persist=True)
        logging.info("Ingestion complete.")

    elif args.command == "query":
        vectorstore = create_or_get_vectorstore()
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        rag_chain = create_rag_chain(retriever)

        logging.info("Starting interactive query session... (Type 'exit' to quit)")
        while True:
            try:
                query = input("\nAsk a question: ")
                if query.lower() == 'exit':
                    break
                if not query.strip():
                    continue

                print("\nThinking...")
                response = rag_chain.invoke(query)
                print("\nAnswer:")
                print(response)

            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()

# PaperPulse: Your RAG-based Research Assistant

PaperPulse is a powerful and intuitive application designed to help you chat with your research papers. It leverages a Retrieval-Augmented Generation (RAG) pipeline to provide accurate, context-aware answers from your own document library. This tool is available as both a user-friendly Streamlit web application and a command-line interface (CLI).

## 🚀 Features

- **Interactive Chat Interface**: Ask questions about your research papers in natural language and get concise, relevant answers.
- **PDF Document Support**: Upload and process your research papers in PDF format.
- **Two Convenient Modes**:
  - **Streamlit Web App**: A rich, interactive user interface for easy document management and chat.
  - **Command-Line Interface (CLI)**: For users who prefer working in the terminal.
- **Powered by State-of-the-Art AI**:
  - **`all-MiniLM-L6-v2`**: For efficient and accurate document embeddings.
  - **Groq & Llama 3**: For lightning-fast and intelligent response generation.
  - **ChromaDB**: For persistent and scalable vector storage.

## 🛠️ Setup and Installation

Follow these steps to set up and run PaperPulse on your local machine.

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/paperpulse.git
cd paperpulse
```

### 2. Create a Virtual Environment

It's recommended to use a virtual environment to manage project dependencies.

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### 3. Install Dependencies

Install the required Python packages using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

PaperPulse requires a Groq API key to function. Create a `.env` file in the root of the project directory and add your key:

```
GROQ_API_KEY="your-groq-api-key"
```

## 🏃‍♀️ Running the Application

You can run PaperPulse in two ways:

### 1. Streamlit Web App

To launch the web application, run the following command in your terminal:

```bash
streamlit run app.py
```

This will open the PaperPulse interface in your web browser, where you can:

- **Ingest Documents**: Upload your PDF files to build the knowledge base.
- **Chat with Papers**: Ask questions and get answers from your documents.

### 2. Command-Line Interface (CLI)

The CLI is ideal for programmatic access or for users who prefer the terminal.

#### Ingest Documents

To process and ingest documents from a directory, use the `ingest` command:

```bash
python paperpulse.py ingest path/to/your/papers
```

#### Query Documents

To start an interactive chat session, use the `query` command:

```bash
python paperpulse.py query
```

## 📂 Project Structure

```
.env
app.py
chroma_db_paperpulse_st/
paperpulse.py
requirements.txt
temp_uploads/
```

- **`app.py`**: The main file for the Streamlit web application.
- **`paperpulse.py`**: The main file for the command-line interface.
- **`requirements.txt`**: A list of all the Python packages required for the project.
- **`.env`**: Stores environment variables, such as your Groq API key.
- **`chroma_db_paperpulse_st/`**: The directory where the Chroma vector store for the Streamlit app is persisted.
- **`temp_uploads/`**: A temporary directory for storing uploaded files during processing.

## 🤝 Contributing

Contributions are welcome! If you have any ideas, suggestions, or bug reports, please open an issue or submit a pull request.

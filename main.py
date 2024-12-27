from flask import Flask, render_template, request
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
import sys
import glob

class SuppressStdout:
    """
    Context manager to suppress standard output and errors.
    Useful for clean operations where output is unnecessary.
    """
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr

def load_and_split_files(directory_path, chunk_size=500, chunk_overlap=0):
    """
    Loads multiple PDF and TXT files from a directory and splits them into manageable text chunks.
    """
    all_splits = []
    # Load PDFs
    pdf_files = glob.glob(os.path.join(directory_path, "*.pdf"))
    for pdf_file in pdf_files:
        print(f"Processing {pdf_file}...")
        loader = PyPDFLoader(pdf_file)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        all_splits.extend(text_splitter.split_documents(documents))

    # Load TXT files
    txt_files = glob.glob(os.path.join(directory_path, "*.txt"))
    for txt_file in txt_files:
        print(f"Processing {txt_file}...")
        with open(txt_file, 'r', encoding='utf-8') as f:
            text = f.read()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        splits = text_splitter.split_text(text)
        all_splits.extend(splits)
    if not all_splits:
            print("Warning: No PDF or TXT files found in the specified directory.")
    return all_splits

def create_vectorstore(splits):
    """
    Creates a Chroma vectorstore from the provided text splits using GPT4All embeddings.
    If no splits are provided, an empty vectorstore is created.
    """
    if not splits:
        print("No documents to index. Creating an empty vectorstore.")
        splits = ["No data available"]  # Placeholder document to avoid failure
    with SuppressStdout():
        return Chroma.from_documents(documents=splits, embedding=GPT4AllEmbeddings())

def get_prompt_template():
    """
    Returns a concise, context-driven prompt template for the QA system.
    """
    template = (
        """Use the following pieces of context to answer the question at the end.\n"
        "If you don't know the answer, just say that you don't know.\n"
        "Use five sentences maximum and keep the answer concise.\n"
        "You are NiNa, a cheeky IT Forensic Analyst, created by Maximilian Gutowski.\n"
        "{context}\n"
        "Question: {question}\n"
        "Helpful Answer:"""
    )
    return PromptTemplate(input_variables=["context", "question"], template=template)

def initialize_llm():
    """
    Initializes the OllamaLLM with the specified model (provided via environment variable) and streaming callback handler.
    """
    model_name = os.getenv("MODEL_NAME", "hf.co/LockeLamora2077/NiNaa")  # Default model if not set
    return OllamaLLM(
        model=model_name,
        callbacks=[StreamingStdOutCallbackHandler()],
        base_url="http://host.docker.internal:11434",
        temperature=0.0
    )

# Flask App Setup
app = Flask(__name__)

# Chat history for both templates
chat_histories = {
    "index.html": [],
    "clean.html": []
}

# Default template
current_template = "index.html"

def create_app(qa_chain):
    """
    Creates a Flask app to serve the QA system with an HTTP GUI and chat history.
    """
    global app

    @app.route("/", methods=["GET", "POST"])
    def index():
        global current_template

        if request.method == "POST":
            # Handle template switching
            selected_template = request.form.get("template")
            if selected_template in chat_histories:
                current_template = selected_template

        return render_template(
            current_template,
            chat_history=chat_histories[current_template],
            current_template=current_template
        )

    @app.route("/query", methods=["POST"])
    def query():
        global current_template

        query_text = request.form.get("query")
        if not query_text:
            return render_template(
                current_template, 
                chat_history=chat_histories[current_template],
                current_template=current_template
            )

        # Mock result for demonstration
        result = qa_chain({"query": query_text})
        answer = result.get("result", "No response. Are you trying to stump me? 🤔")

        # Append to the current template's chat history
        chat_histories[current_template].append({
            "question": query_text,
            "answer": answer
        })

        # Limit chat history to last 5 entries
        if len(chat_histories[current_template]) > 5:
            chat_histories[current_template].pop(0)

        return render_template(
            current_template,
            chat_history=chat_histories[current_template],
            current_template=current_template
        )
    return app

def main():
    """
    Main function to run the PDF-based QA retrieval system as a Flask server with an HTTP GUI.
    """
    # Load and process PDFs
    pdf_directory = "/app/pdf"  # Directory containing multiple PDFs
    splits = load_and_split_files(pdf_directory)
    vectorstore = create_vectorstore(splits)

    # Initialize components
    qa_prompt = get_prompt_template()
    llm = initialize_llm()
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": qa_prompt},
    )

    # Run Flask server
    app = create_app(qa_chain)
    print("Starting server at http://localhost:8000")
    app.run(host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()

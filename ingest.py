# Import os to read environment variables
import os

# Load variables from .env file
from dotenv import load_dotenv

# Load PDF documents
from langchain_community.document_loaders import PyPDFLoader

# Split large text into smaller chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker

# OpenAI embedding model
from langchain_huggingface import HuggingFaceEmbeddings


# PostgreSQL pgvector integration
from langchain_postgres import PGVector


# Load .env variables into Python
load_dotenv()


# Read database connection string from .env
DATABASE_URL = os.getenv("DATABASE_URL")


# Read collection/table name from .env
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag_documents")



embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)


# Function to load PDF and save embeddings
def ingest_pdf(file_path: str):
    # Create PDF loader from file path
    loader = PyPDFLoader(file_path)

    # Load PDF pages as LangChain documents
    documents = loader.load()

    # Create text splitter
    # text_splitter = RecursiveCharacterTextSplitter(
    #     chunk_size=1000,        # Maximum characters per chunk
    #     chunk_overlap=200,      # Overlap keeps context between chunks
    # )

    splitter = SemanticChunker(embeddings)


    # Split PDF pages into smaller chunks
    chunks = splitter.split_documents(documents)

    # Store chunks and embeddings in PostgreSQL pgvector
    vector_store = PGVector.from_documents(
        documents=chunks,                  # Text chunks to store
        embedding=embeddings,              # Embedding model
        connection=DATABASE_URL,           # PostgreSQL connection
        collection_name=COLLECTION_NAME,   # Collection name
        use_jsonb=True,                    # Store metadata as JSONB
    )

    # Return number of stored chunks
    return len(chunks)


# Run this file directly from terminal
if __name__ == "__main__":
    # Set your PDF file path
    pdf_path = "docs/sample.pdf"

    # Ingest PDF into vector database
    total_chunks = ingest_pdf(pdf_path)

    # Print result
    print(f"✅ Ingested {total_chunks} chunks into pgvector.")
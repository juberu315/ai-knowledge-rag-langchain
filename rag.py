# Import os to read environment variables
import os

# Load .env variables
from dotenv import load_dotenv

# OpenAI chat model
from langchain_openai import ChatOpenAI

# OpenAI embedding model
from langchain_openai import OpenAIEmbeddings

# PostgreSQL pgvector integration
from langchain_postgres import PGVector

# Prompt template
from langchain_core.prompts import ChatPromptTemplate

# Output parser converts model response to string
from langchain_core.output_parsers import StrOutputParser


# Load variables from .env
load_dotenv()


# Read database URL from .env
DATABASE_URL = os.getenv("DATABASE_URL")


# Read collection name from .env
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag_documents")


# Create embedding model
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)


# Connect to existing PGVector store
vector_store = PGVector(
    embeddings=embeddings,
    connection=DATABASE_URL,
    collection_name=COLLECTION_NAME,
    use_jsonb=True,
)


# Convert vector store into retriever
retriever = vector_store.as_retriever(
    search_kwargs={
        "k": 4  # Return top 4 most similar chunks
    }
)


# Create LLM model
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)


# Create RAG prompt
prompt = ChatPromptTemplate.from_template("""
You are a helpful AI assistant.

Answer the user question using only the context below.

If the answer is not found in the context, say:
"I don't know based on the provided document."

Context:
{context}

Question:
{question}
""")


# Format retrieved documents into one text block
def format_docs(docs):
    # Join page contents from retrieved documents
    return "\n\n".join(doc.page_content for doc in docs)


# Main RAG function
def ask_rag(question: str):
    # Retrieve relevant chunks from vector database
    docs = retriever.invoke(question)

    # Convert retrieved documents into plain text
    context = format_docs(docs)

    # Create chain: prompt -> LLM -> string output
    chain = prompt | llm | StrOutputParser()

    # Run chain with context and question
    answer = chain.invoke({
        "context": context,
        "question": question
    })

    # Return final answer and source chunks
    return {
        "answer": answer,
        "sources": [
            {
                "page": doc.metadata.get("page"),
                "source": doc.metadata.get("source"),
                "content": doc.page_content[:300]
            }
            for doc in docs
        ]
    }
# Import FastAPI
from fastapi import FastAPI

# Import Pydantic for request body validation
from pydantic import BaseModel

# Import RAG function
from rag import ask_rag


# Create FastAPI app
app = FastAPI(
    title="LangChain RAG API",
    description="Simple RAG system using LangChain, OpenAI, and PostgreSQL pgvector",
    version="1.0.0"
)


# Request body schema
class QuestionRequest(BaseModel):
    # User question
    question: str


# Health check endpoint
@app.get("/")
def root():
    # Return simple API status
    return {
        "message": "LangChain RAG API is running"
    }


# RAG question endpoint
@app.post("/ask")
def ask_question(request: QuestionRequest):
    # Send user question to RAG function
    result = ask_rag(request.question)

    # Return RAG result
    return result
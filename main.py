from fastapi import FastAPI, HTTPException, Body, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Dict, Any
import os
import json
from langchain_community.document_loaders.csv_loader import CSVLoader
from pathlib import Path
from sentence_transformers import SentenceTransformer
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

app = FastAPI(title="Hotel Booking RAG API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the static directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Input model for queries
class QueryInput(BaseModel):
    query: str

# Model for chat history
class ChatMessage(BaseModel):
    role: str
    content: str

# Initialize global variables
vector_store = None
rag_chain = None
vector_store2 = None
rag_chain2 = None
chat_history = []

# Setup function to initialize the RAG system
@app.on_event("startup")
async def startup_event():
    global vector_store, rag_chain
    
    # Set up Groq LLM
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        print("Warning: GROQ_API_KEY not set, using default placeholder")
        api_key = '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'
    
    os.environ["GROQ_API_KEY"] = api_key
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.0,
        max_retries=2,
    )

    # Define embedding model
    embeddings = SentenceTransformer('all-MiniLM-L6-v2')

    # Load existing FAISS index
    try:
        vector_store = FAISS.load_local(
            "General_FAISS", 
            embeddings.encode,
            allow_dangerous_deserialization=True
        )
        vector_store2 = FAISS.load_local(
            "Analysis_FAISS", 
            embeddings.encode,
            allow_dangerous_deserialization=True
        )
        print("Existing index loaded successfully!")
    except Exception as e:
        raise Exception(f"Failed to load FAISS index: {e}")

    # Set up the retriever with search parameters
    retriever = vector_store.as_retriever(
        search_type="similarity",
    )
    retriever2 = vector_store2.as_retriever(
        search_type="similarity",
    )

    # Set up system prompt
    system_prompt = """You are a helpful hotel booking assistant. Answer questions based on the provided context.
    If you don't know something or can't find it in the context, say so honestly.
    Always be concise and to the point.

    Context: {context}
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # Create the question-answer chain
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    rag_chain2 = create_retrieval_chain(retriever2, question_answer_chain)


# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# As ask endpoint
@app.post("/query")
async def process_query(query_input: QueryInput):
    global rag_chain, chat_history
    
    if not rag_chain:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        # Add user message to history
        chat_history.append({"role": "user", "content": query_input.query})
        
        # Get answer from RAG chain
        response = rag_chain.invoke({"input": query_input.query})
        
        # Ensure we have a valid response
        if not response or 'answer' not in response:
            raise HTTPException(status_code=500, detail="Failed to get response from RAG chain")
            
        answer = response.get('answer', "I'm sorry, I couldn't process that query.")
        
        # Add assistant response to history
        chat_history.append({"role": "assistant", "content": answer})
        
        return {
            "answer": answer
        }
    except Exception as e:
        print(f"Error processing query: {str(e)}")  # For debugging
        raise HTTPException(status_code=500, detail=str(e))
    
# Ask endpoint
@app.post("/ask")
async def process_query(query_input: QueryInput):
    global rag_chain, chat_history
    
    if not rag_chain:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        # Add user message to history
        chat_history.append({"role": "user", "content": query_input.query})
        
        # Get answer from RAG chain
        response = rag_chain.invoke({"input": query_input.query})
        
        # Ensure we have a valid response
        if not response or 'answer' not in response:
            raise HTTPException(status_code=500, detail="Failed to get response from RAG chain")
            
        answer = response.get('answer', "I'm sorry, I couldn't process that query.")
        
        # Add assistant response to history
        chat_history.append({"role": "assistant", "content": answer})
        
        return {
            "answer": answer
        }
    except Exception as e:
        print(f"Error processing query: {str(e)}")  # For debugging
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyse")
async def process_query(query_input: QueryInput):
    global rag_chain2, chat_history
    
    if not rag_chain2:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        # Add user message to history
        chat_history.append({"role": "user", "content": query_input.query})
        
        # Get answer from RAG chain
        response = rag_chain2.invoke({"input": query_input.query})
        
        # Ensure we have a valid response
        if not response or 'answer' not in response:
            raise HTTPException(status_code=500, detail="Failed to get response from RAG chain")
            
        answer = response.get('answer', "I'm sorry, I couldn't process that query.")

        return {
            "answer": answer
        }
    except Exception as e:
        print(f"Error processing query: {str(e)}")  # For debugging
        raise HTTPException(status_code=500, detail=str(e))
    
# Get chat history endpoint
@app.get("/history")
async def get_chat_history():
    return {"history": chat_history}

# Clear chat history endpoint
@app.post("/clear")
async def clear_chat_history():
    global chat_history
    chat_history = []
    return {"status": "Chat history cleared"}

class Message(BaseModel):
    message: str

@app.get("/", response_class=HTMLResponse)
async def root():
    with open("static/index.html") as f:
        return f.read()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
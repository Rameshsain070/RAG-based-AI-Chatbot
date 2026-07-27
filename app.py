from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag import rag_graph  # This imports your working LangGraph from rag.py

# Initialize FastAPI app
app = FastAPI(
    title="Agentic AI RAG Chatbot API",
    description="API for answering questions grounded strictly in the Agentic AI eBook",
    version="1.0.0"
)

# Define the Request Schema
class QueryRequest(BaseModel):
    question: str

# Define the Response Schema (Matches assignment requirements)
class QueryResponse(BaseModel):
    answer: str
    confidence_score: float
    context: list[str]

@app.get("/")
def root():
    return {"status": "online", "message": "Agentic AI RAG API is running. Send POST requests to /chat"}

@app.post("/chat", response_model=QueryResponse)
def chat_endpoint(request: QueryRequest):
    try:
        # Run the question through your LangGraph pipeline
        result = rag_graph.invoke({"question": request.question})
        
        return {
            "answer": result["answer"],
            "confidence_score": result["confidence_score"],
            "context": result["context"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
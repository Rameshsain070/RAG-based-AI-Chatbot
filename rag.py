import os
from typing import List, TypedDict
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore
from langgraph.graph import StateGraph, START, END

load_dotenv()

# Configuration
INDEX_NAME = "agentic-ai-ebook-final"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# 1. Define the LangGraph State
class GraphState(TypedDict):
    question: str
    context: List[str]
    answer: str
    confidence_score: float

# 2. Structured Output Schema for LLM
class GroundedResponse(BaseModel):
    answer: str = Field(
        description="The answer to the user's question. MUST be derived strictly from the provided context."
    )
    confidence_score: float = Field(
        description="A float between 0.0 and 1.0 representing how strongly the provided context supports the answer."
    )

# Initialize Embeddings & Vector Store
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)

# 3. Retrieve Node
def retrieve_node(state: GraphState) -> dict:
    question = state["question"]
    # Retrieve top 4 most relevant text chunks from Pinecone
    docs = vectorstore.similarity_search(question, k=4)
    context_chunks = [doc.page_content for doc in docs]
    return {"context": context_chunks}

# 4. Generate Node
def generate_node(state: GraphState) -> dict:
    question = state["question"]
    context = state["context"]
    formatted_context = "\n\n---\n\n".join(context)

    system_prompt = (
        "You are an AI assistant evaluating an eBook on Agentic AI. "
        "Your task is to answer the user's question STRICTLY using only the provided context below.\n"
        "Rules:\n"
        "1. If the answer cannot be completely deduced from the context, state: "
        "'I cannot answer this question based on the provided document.'\n"
        "2. Do not use outside knowledge.\n"
        "3. Set the confidence_score to 1.0 if the answer is fully present in context, "
        "lower (or 0.0) if partially or completely missing.\n\n"
        f"Context:\n{formatted_context}"
    )

    # Use Groq's Llama 3 model (Free, Fast, and no region blocks)
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY")
    )
    structured_llm = llm.with_structured_output(GroundedResponse)

    response: GroundedResponse = structured_llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ])

    return {
        "answer": response.answer,
        "confidence_score": response.confidence_score
    }

# 5. Assemble and Compile Graph
workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)

workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

rag_graph = workflow.compile()

# Local Test Execution
if __name__ == "__main__":
    test_query = "What is Agentic AI?"
    print(f"Querying: {test_query}\n")
    result = rag_graph.invoke({"question": test_query})

    print("--- ANSWER ---")
    print(result["answer"])
    print("\n--- CONFIDENCE SCORE ---")
    print(result["confidence_score"])
    print("\n--- RETRIEVED CHUNKS ---")
    for i, chunk in enumerate(result["context"], 1):
        print(f"[{i}] {chunk[:120]}...\n")
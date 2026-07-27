import os
import time
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

# Configuration
PDF_URL = "https://konverge.ai/pdf/Ebook-Agentic-AI.pdf"
INDEX_NAME = "agentic-ai-ebook-final"  # New index name for the new embeddings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384  # Dimension for this HuggingFace model

def main():
    print("1. Initializing Pinecone client...")
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
    if INDEX_NAME not in existing_indexes:
        print(f"Index '{INDEX_NAME}' not found. Creating it now...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        while not pc.describe_index(INDEX_NAME).status["ready"]:
            time.sleep(1)
        print("Index created successfully!")
    else:
        print(f"Index '{INDEX_NAME}' already exists.")

    print("\n2. Downloading and parsing the PDF...")
    loader = PyPDFLoader(PDF_URL)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages from the PDF.")

    print("\n3. Chunking the text...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split the document into {len(chunks)} chunks.")

    print("\n4. Generating embeddings and uploading to Pinecone...")
    # Using free HuggingFace embeddings instead of OpenAI
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=INDEX_NAME
    )
    
    print("\n Ingestion complete! The Vector DB is ready for queries.")

if __name__ == "__main__":
    main()
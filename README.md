# Agentic AI Chatbot

I built a custom chatbot that answers questions strictly based on the provided *Agentic AI* eBook. I avoided using low-code, drag-and-drop platforms and built the entire pipeline from scratch using Python.

##  Try it live!
You don't even need to download the code to test it. I deployed a live UI that you can interact with right now: https://rag-based-ai-chatbot-aiqcvfhjnxc6uzdjmvjtaj.streamlit.app/

---

## How it works (Behind the Scenes)
This app uses a standard RAG (Retrieval-Augmented Generation) setup to make sure the AI doesn't hallucinate or make things up:

1. **Reading the Book:** First, a Python script reads the PDF and chops it into small, searchable paragraphs.
2. **The Database:** These paragraphs are stored inside a Pinecone vector database.
3. **The Brain:** When you ask a question, the app finds the most relevant paragraphs from the book and sends them to Groq (using their smart Llama 3.3 70B model). 
4. **Strict Rules:** The AI is instructed to *only* use the text from the book to write its answer. It also generates a "confidence score" so you know exactly how reliable the answer is.


## How to run it locally

If you want to look at the code and run it on your own machine, here is the simple setup:

**1. Download the code**
git clone [https://github.com/Rameshsain070/RAG-based-AI-Chatbot.git](https://github.com/Rameshsain070/RAG-based-AI-Chatbot.git)
cd RAG-based-AI-Chatbot

**2. Install the requirements**
pip install -r requirements.txt

**3. Add your API keys**
Create a file named .env in the main folder and paste your free API keys inside:
PINECONE_API_KEY="your-pinecone-key".
GROQ_API_KEY="your-groq-key".

**4. Start the Chatbot**
To launch the chat interface, simply run:
python ingest.py
uvicorn app:app --reload

**5 streamlit run ui.py**
(Note: If you want to test the raw backend API instead, you can run uvicorn app:app --reload and go to http://localhost:8000/docs).

import streamlit as st
from rag import rag_graph  # Directly runs the LangGraph logic

# Page Setup
st.set_page_config(page_title="Agentic AI Chatbot", page_icon="🤖")
st.title("🤖 Agentic AI RAG Chatbot")
st.caption("Ask questions strictly grounded in the Agentic AI eBook.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! Ask me anything about Agentic AI."}]

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("E.g., What is Agentic AI?"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Searching eBook and thinking..."):
        try:
            # Invoke LangGraph directly
            result = rag_graph.invoke({"question": prompt})
            
            answer = result["answer"]
            confidence = result["confidence_score"]
            context_chunks = result["context"]

            with st.chat_message("assistant"):
                st.markdown(answer)
                
                with st.expander(f"📊 Confidence Score: {confidence} | View Source Context"):
                    for i, chunk in enumerate(context_chunks, 1):
                        st.info(f"**Chunk {i}:**\n{chunk}")
            
            st.session_state.messages.append({"role": "assistant", "content": answer})

        except Exception as e:
            st.error(f"Error processing query: {str(e)}")
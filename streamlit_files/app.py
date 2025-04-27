import streamlit as st
import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import database
import llm_connection

sys.modules['torch._classes'].__path__ = []

st.set_page_config(page_title="Legal Chatbot", page_icon="🧠")
st.title("🧠 Local Legal Chatbot")


if "messages" not in st.session_state:
    st.session_state.messages = []


if prompt := st.chat_input("Ask a legal question..."):
  
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            context = "\n\n".join(database.get_context(prompt))

            full_prompt = f"""You are a helpful legal assistant. Use the context below to answer the question.

Context:
{context}

Question: {prompt}
Answer:"""

            full_response = ""
            response_container = st.empty()

            for token in llm_connection.query_model(full_prompt):
                full_response += token
                response_container.markdown(full_response + "▌")

            response_container.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})

for msg in st.session_state.messages[:-2 if prompt else None]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

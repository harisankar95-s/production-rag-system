import streamlit as st
import requests
import uuid
import time

st.title("AI Research Assistant")

try:
    health = requests.get("http://127.0.0.1:8000/health", timeout=5)
    st.sidebar.success(f"FastAPI: {health.json()}")
except Exception as e:
    st.sidebar.error(f"FastAPI unreachable: {e}")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me anything about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = "Error: Could not connect to backend."
            for attempt in range(3):
                try:
                    response = requests.post(
                        "http://127.0.0.1:8000/ask",
                        json={"question": prompt, "thread_id": st.session_state.thread_id},
                        timeout=120
                    )
                    answer = response.json()["answer"]
                    break
                except Exception as e:
                    if attempt < 2:
                        time.sleep(2)
                    else:
                        answer = f"Error: {e}"

        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
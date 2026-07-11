import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/ask")

st.set_page_config(page_title="R&D Assistant", layout="wide")
st.title("Горно-металлургическая база знаний")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Задайте вопрос..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Анализ графа знаний..."):
            try:
                response = requests.post(API_URL, json={"query": prompt})
                response.raise_for_status()
                answer = response.json().get("answer", "Ошибка получения ответа.")
            except Exception as e:
                answer = f"Ошибка подключения к API: {e}"

            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

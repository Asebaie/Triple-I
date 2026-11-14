import html
import os
import re
from pathlib import Path

import httpx
import streamlit as st

import rag

SERVICE_ERROR = "Сервис языковой модели сейчас недоступен. Попробуйте ещё раз через минуту."
MAX_QUESTION_CHARS = 4000
CLAUSE = re.compile(r"(?<!\d)(?<!\d\.)\d{1,2}\.\d{1,2}(?!\.?\d)")

st.set_page_config(page_title="Юридический ассистент", page_icon="⚖️")


@st.cache_resource(show_spinner="Индексирую базу знаний…")
def load_index() -> rag.Index:
    return rag.build_index(Path(os.environ.get("DATA_DIR", "data")))


def as_dialog(messages: list[dict]) -> str:
    if len(messages) == 1:
        return messages[0]["content"]
    roles = {"user": "Пользователь", "assistant": "Ассистент"}
    return "\n".join(f"{roles[m['role']]}: {m['content']}" for m in messages)


def highlighted(text: str, fragments: list[str]) -> str:
    marked = html.escape(text)
    for fragment in fragments:
        escaped = html.escape(fragment)
        marked = marked.replace(escaped, f"<mark>{escaped}</mark>")
    return f"<div style='white-space: pre-wrap; font-size: 0.9rem'>{marked}</div>"


def cited_lines(document: rag.Document, answer: str) -> list[str]:
    clauses = set(CLAUSE.findall(answer))
    return [line for line in document.text.split("\n") if line.split(" ", 1)[0].rstrip(".") in clauses]


def show_sources(answer: str, sources: list[tuple[rag.Document, list[rag.Chunk]]]) -> None:
    if not sources:
        st.caption("Ответ не опирается на документы базы знаний.")
        return
    with st.expander(f"Подтверждающие документы: {len(sources)}"):
        for document, chunks in sources:
            st.markdown(f"**{document.title}** · {document.category}")
            if document.source:
                st.caption(document.source)
            if len(document.text) <= rag.WHOLE_DOC_CHARS:
                st.html(highlighted(document.text, cited_lines(document, answer)))
            else:
                for chunk in chunks:
                    st.html(highlighted(chunk.text, []))
            st.divider()


try:
    index = load_index()
except (httpx.HTTPError, RuntimeError):
    st.error(SERVICE_ERROR)
    st.stop()

messages = st.session_state.setdefault("messages", [])

with st.sidebar:
    st.header("База знаний")
    if st.button("Новый диалог", width="stretch"):
        messages.clear()
        st.rerun()
    titles = {f"{d.title} ({d.category})": d for d in index.documents}
    document = titles[st.selectbox("Документ", list(titles))]
    if document.source:
        st.caption(document.source)
    with st.container(height=600):
        st.html(highlighted(document.text, []))

st.title("Юридический ассистент")
st.caption("Отвечает только по документам базы знаний и ссылается на положения, на которые опирается.")

for message in messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message["role"] == "assistant":
            show_sources(message["content"], message["sources"])

question = st.chat_input("Задайте вопрос по договорам, регламентам или законам", max_chars=MAX_QUESTION_CHARS)
if question and question.strip():
    messages.append({"role": "user", "content": question.strip()})
    with st.chat_message("user"):
        st.write(messages[-1]["content"])
    try:
        with st.spinner("Ищу ответ в документах…"):
            result = rag.answer(index, as_dialog(messages))
    except (httpx.HTTPError, RuntimeError):
        messages.pop()
        st.error(SERVICE_ERROR)
    else:
        messages.append({"role": "assistant", "content": result.text, "sources": result.sources})
        st.rerun()

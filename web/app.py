import html

import streamlit as st

from assistant import LegalAssistant

MAX_QUESTION_CHARS = 4000
ANSWER_TIMEOUT = 300

st.set_page_config(page_title="Юридический ассистент", page_icon="⚖️")


@st.cache_resource(show_spinner="Загружаю базу знаний…")
def load_assistant() -> LegalAssistant:
    assistant = LegalAssistant(answer_timeout=ANSWER_TIMEOUT)
    assistant.warmup()
    return assistant


def highlighted(text: str, spans: list[tuple[int, int]]) -> str:
    parts, last = [], 0
    for start, end in sorted(spans):
        if start < last:
            continue
        parts += [html.escape(text[last:start]), f"<mark>{html.escape(text[start:end])}</mark>"]
        last = end
    parts.append(html.escape(text[last:]))
    return f"<div style='white-space: pre-wrap; font-size: 0.9rem'>{''.join(parts)}</div>"


def show_citations(citations: list[dict]) -> None:
    if not citations:
        st.caption("Подтверждающие документы для этого ответа не указаны.")
        return
    documents: dict[str, list[dict]] = {}
    for citation in citations:
        documents.setdefault(citation["doc_id"], []).append(citation)
    with st.expander(f"Подтверждающие документы: {len(documents)}"):
        for doc_id, cited in documents.items():
            document = assistant.get_document(doc_id)
            clauses = ", ".join(f"п. {c['clause']}" for c in cited if c["clause"])
            st.markdown(f"**{document['title']}**" + (f" · {clauses}" if clauses else ""))
            if document["status"]:
                st.caption(document["status"])
            spans = [(c["start"], c["end"]) for c in cited if c["found"]]
            with st.container(height=360):
                st.html(highlighted(document["text"], spans))
            st.divider()


assistant = load_assistant()
health = assistant.health()
if not health["kb_loaded"]:
    st.error(f"База знаний не загружена: {health['kb_error']}")
    st.stop()

messages = st.session_state.setdefault("messages", [])

with st.sidebar:
    st.header("База знаний")
    if st.button("Новый диалог", width="stretch"):
        messages.clear()
        st.rerun()
    titles = {d["title"]: d["doc_id"] for d in assistant.list_documents()}
    document = assistant.get_document(titles[st.selectbox("Документ", list(titles))])
    if document["status"]:
        st.caption(document["status"])
    with st.container(height=600):
        st.html(highlighted(document["text"], []))

st.title("Юридический ассистент")
st.caption("Отвечает только по документам базы знаний и ссылается на положения, на которые опирается.")

for message in messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message["role"] == "assistant":
            show_citations(message["citations"])

question = st.chat_input("Задайте вопрос по договорам, регламентам или законам", max_chars=MAX_QUESTION_CHARS)
if question and question.strip():
    messages.append({"role": "user", "content": question.strip()})
    with st.chat_message("user"):
        st.write(messages[-1]["content"])
    with st.spinner("Ищу ответ в документах…"):
        result = assistant.answer([{"role": m["role"], "content": m["content"]} for m in messages])
    if result.status == "error":
        messages.pop()
        st.error(result.text)
    else:
        messages.append({"role": "assistant", "content": result.text, "citations": result.citations})
        st.rerun()

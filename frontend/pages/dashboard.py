import os
import sqlite3

import pandas as pd
import streamlit as st

st.set_page_config(page_title="R&D Dashboard", layout="wide")
st.title("Аналитическая панель R&D")


@st.cache_data
def get_cached_dashboard_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../.."))
    db_path = os.path.join(project_root, "knowledge_base.db")

    if not os.path.exists(db_path):
        return None, None

    conn = sqlite3.connect(db_path)
    facts = pd.read_sql_query("SELECT * FROM facts", conn)
    params = pd.read_sql_query("SELECT * FROM parameters", conn)
    conn.close()
    return facts, params


df_facts, df_params = get_cached_dashboard_data()

if df_facts is None or df_facts.empty:
    st.warning("База данных пуста или еще не создана. Запустите сначала скрипт импорта данных.")
else:
    col1, col2, col3 = st.columns(3)
    col1.metric("Всего фактов", len(df_facts))
    col2.metric("Всего параметров", len(df_params))
    col3.metric("Источников данных", df_facts["source_file"].nunique())

    st.write("---")
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Популярные материалы")
        mat_counts = df_facts["material"].value_counts().head(15)
        st.bar_chart(mat_counts)

    with col_right:
        st.subheader("Популярные процессы")
        proc_counts = df_facts["process"].value_counts().head(15)
        st.bar_chart(proc_counts)

    st.write("---")
    st.subheader("География исследований")
    geo_counts = df_facts["geography"].value_counts()
    st.write(geo_counts)

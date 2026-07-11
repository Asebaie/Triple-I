import os
import sqlite3

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Data Gaps Analysis", layout="wide")
st.title("Анализ пробелов в знаниях R&D")


@st.cache_data
def get_cached_gaps_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../.."))
    db_path = os.path.join(project_root, "knowledge_base.db")

    if not os.path.exists(db_path):
        return None, None, None

    conn = sqlite3.connect(db_path)

    query_gaps_proc = """
        SELECT DISTINCT f.process, f.source_file 
        FROM facts f 
        LEFT JOIN parameters p ON f.id = p.fact_id 
        WHERE p.id IS NULL;
    """
    df_gaps_proc = pd.read_sql_query(query_gaps_proc, conn)

    query_geo = """
        SELECT material, COUNT(DISTINCT geography) as geo_count 
        FROM facts 
        GROUP BY material 
        HAVING geo_count = 1;
    """
    df_geo = pd.read_sql_query(query_geo, conn)

    query_low_conf = """
        SELECT material, process, outcome, source_file 
        FROM facts 
        WHERE LOWER(confidence_level) LIKE '%низк%' OR LOWER(confidence_level) LIKE '%теор%';
    """
    df_low_conf = pd.read_sql_query(query_low_conf, conn)

    conn.close()
    return df_gaps_proc, df_geo, df_low_conf


df_gaps_proc, df_geo, df_low_conf = get_cached_gaps_data()

if df_gaps_proc is None:
    st.warning("База данных еще не заполнена или отсутствует. Запустите импорт данных.")
else:
    st.subheader("1. Процессы без зарегистрированных числовых параметров")
    if not df_gaps_proc.empty:
        st.write(df_gaps_proc)
    else:
        st.success("Все процессы имеют числовые параметры.")

    st.write("---")
    st.subheader("2. Неизученные географические зоны по материалам")
    st.write("Материалы, исследовавшиеся только в одной географической зоне (дефицит сравнительного анализа):")
    st.write(df_geo)

    st.write("---")
    st.subheader("3. Факты со слабой достоверностью (требующие верификации)")
    if not df_low_conf.empty:
        st.write(df_low_conf)
    else:
        st.info("Фактов с низким уровнем достоверности не найдено.")

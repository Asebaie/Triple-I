import json

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from core.llm_client import ask_llm_json
from core.logger import logger
from core.prompts import DECOMPOSE_PROMPT, SQL_GENERATION_PROMPT, SYNTHESIS_PROMPT


def extract_json_substring(text_data: str) -> str:
    start = text_data.find("{")
    end = text_data.rfind("}")
    if start != -1 and end != -1:
        return text_data[start : end + 1]
    return text_data


async def decompose_query(user_query: str) -> list:
    response = await ask_llm_json(system_prompt=DECOMPOSE_PROMPT, user_text=user_query)
    try:
        cleaned = extract_json_substring(response)
        queries = json.loads(cleaned).get("sub_queries", [user_query])
        logger.info(f"Decompose: {queries}")
        return queries
    except Exception:
        return [user_query]


async def get_sql_from_query(user_query: str) -> str:
    response = await ask_llm_json(system_prompt=SQL_GENERATION_PROMPT, user_text=user_query)
    try:
        cleaned = extract_json_substring(response)
        sql = json.loads(cleaned).get("sql_query", "")
        sql = sql.replace("LOWER(material)", "material_lower")
        sql = sql.replace("LOWER(process)", "process_lower")
        sql = sql.replace("LOWER(outcome)", "outcome_lower")
        sql = sql.replace("LOWER(name)", "name_lower")
        logger.info(f"SQL: {sql}")
        return sql
    except Exception:
        return ""


async def execute_query(db: AsyncSession, sql_query: str) -> list:
    if not sql_query:
        return []

    query_upper = sql_query.upper()
    for forbidden in ["DELETE", "DROP", "UPDATE", "INSERT", "ALTER", "TRUNCATE"]:
        if forbidden in query_upper:
            logger.warning(f"Blocked SQL Injection attempt: {sql_query}")
            return []

    try:
        result = await db.execute(text(sql_query))
        rows = result.fetchall()
        results = []
        for row in rows:
            row_dict = dict(row._mapping)
            if "source_file" in row_dict:
                row_dict["источник"] = row_dict.pop("source_file")
            results.append(row_dict)
        return results
    except Exception as e:
        logger.error(f"SQL Error: {e}")
        return []


async def fallback_keyword_search(db: AsyncSession, query: str) -> list:
    logger.info("Fallback search...")
    words = [w.strip() for w in query.split() if len(w.strip()) > 3]
    if not words:
        return []

    conditions = []
    for word in words:
        base_word = word[:-2] if len(word) > 4 else word
        base_word = base_word.lower()
        conditions.append(f"LOWER(material) LIKE '%{base_word}%'")
        conditions.append(f"LOWER(process) LIKE '%{base_word}%'")
        conditions.append(f"LOWER(outcome) LIKE '%{base_word}%'")

    sql = f"SELECT * FROM facts WHERE {' OR '.join(conditions)} LIMIT 5"
    return await execute_query(db, sql)


def format_local_results(grouped_results: list) -> str:
    text = "Вот что найдено в базе знаний:\n\n"
    for group in grouped_results:
        sub_query = group["sub_query"]
        results = group["results"]

        text += f"### 🔍 По запросу: *{sub_query}*\n"
        if not results:
            text += "└ *Информации в базе данных пока нет.*\n\n"
            continue

        for i, row in enumerate(results, 1):
            material = row.get("material", "Неизвестно")
            process = row.get("process", "Неизвестно")
            outcome = row.get("outcome", "Неизвестно")
            source = row.get("источник", "Неизвестно")
            text += f"{i}. **Материал**: {material} | **Процесс**: {process} | **Результат**: {outcome} *(Источник: {source})*\n"
        text += "\n"
    return text


async def synthesize_answer(user_query: str, db_results: list) -> str:
    total_results = sum(len(g["results"]) for g in db_results)
    if total_results == 0:
        return "Не найдено информации по вашему запросу."

    flat_results = []
    for g in db_results:
        flat_results.extend(g["results"])

    context = str(flat_results)[:8000]
    response = await ask_llm_json(
        system_prompt=SYNTHESIS_PROMPT, user_text=f"Вопрос: {user_query}\nДанные из базы: {context}"
    )

    if not response or response.strip() == "{}" or response.strip() == "":
        return format_local_results(db_results)

    return response.strip()


async def ask_question(db: AsyncSession, query: str) -> str:
    try:
        sub_queries = await decompose_query(query)
        grouped_results = []
        seen_ids = set()

        for sub_query in sub_queries:
            sql = await get_sql_from_query(sub_query)
            results = await execute_query(db, sql)

            if not results:
                results = await fallback_keyword_search(db, sub_query)

            unique_results = []
            for row in results:
                row_id = row.get("id")
                if row_id not in seen_ids:
                    seen_ids.add(row_id)
                    unique_results.append(row)

            grouped_results.append({"sub_query": sub_query, "results": unique_results})

        return await synthesize_answer(query, grouped_results)
    except Exception as e:
        logger.error(f"Error in ask_question: {e}")
        return "Произошла ошибка при обработке запроса."

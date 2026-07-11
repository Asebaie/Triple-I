import pytest
from sqlalchemy import text

from core.retriever import (
    decompose_query,
    execute_query,
    fallback_keyword_search,
    format_local_results,
    get_sql_from_query,
)


@pytest.mark.asyncio
async def test_execute_query_empty_sql(db_session):
    result = await execute_query(db_session, "")

    assert result == []


@pytest.mark.asyncio
async def test_execute_query_valid_sql(db_session):
    await db_session.execute(
        text(
            "INSERT INTO facts (material, material_lower, process, process_lower, "
            "geography, outcome, outcome_lower, confidence_level, source_file) "
            "VALUES ('Медь', 'медь', 'Плавка', 'плавка', 'РФ', 'Успех', 'успех', 'Высокий', 'test.pdf')"
        )
    )
    await db_session.commit()

    result = await execute_query(db_session, "SELECT * FROM facts WHERE material = 'Медь'")

    assert len(result) == 1
    assert result[0]["material"] == "Медь"
    assert result[0]["источник"] == "test.pdf"


@pytest.mark.asyncio
async def test_execute_query_invalid_sql(db_session):
    result = await execute_query(db_session, "SELECT * FROM nonexistent_table")

    assert result == []


@pytest.mark.asyncio
async def test_execute_query_sql_injection_attempt(db_session):
    result = await execute_query(db_session, "DROP TABLE facts;")

    assert result == []


@pytest.mark.asyncio
async def test_decompose_query_valid(mock_llm):
    mock_llm.return_value = '{"sub_queries": ["температура плавки", "география"]}'

    result = await decompose_query("Какая температура плавки меди в России?")

    assert result == ["температура плавки", "география"]


@pytest.mark.asyncio
async def test_decompose_query_fallback(mock_llm):
    mock_llm.return_value = "not a json"

    result = await decompose_query("Простой запрос")

    assert result == ["Простой запрос"]


@pytest.mark.asyncio
async def test_get_sql_from_query_replaces_lower(mock_llm):
    mock_llm.return_value = '{"sql_query": "SELECT * FROM facts WHERE LOWER(material) = \'медь\'"}'

    sql = await get_sql_from_query("медь")

    assert "material_lower" in sql
    assert "LOWER(material)" not in sql


@pytest.mark.asyncio
async def test_get_sql_from_query_empty_on_error(mock_llm):
    mock_llm.return_value = "это не json"

    sql = await get_sql_from_query("запрос")

    assert sql == ""


@pytest.mark.asyncio
async def test_fallback_keyword_search_short_words(db_session):
    result = await fallback_keyword_search(db_session, "а б в")

    assert result == []


@pytest.mark.asyncio
async def test_fallback_keyword_search_builds_sql(db_session):
    await db_session.execute(
        text(
            "INSERT INTO facts (material, material_lower, process, process_lower, "
            "geography, outcome, outcome_lower, confidence_level, source_file) "
            "VALUES ('Медный сплав', 'медный сплав', 'Плавка', 'плавка', 'РФ', 'Успех', 'успех', 'Высокий', 't.pdf')"
        )
    )
    await db_session.commit()

    result = await fallback_keyword_search(db_session, "медный сплав")

    assert len(result) >= 1


def test_format_local_results_empty():
    grouped = [{"sub_query": "тест", "results": []}]
    text = format_local_results(grouped)

    assert "Информации в базе данных пока нет" in text


def test_format_local_results_with_data():
    grouped = [
        {
            "sub_query": "медь",
            "results": [
                {
                    "material": "Медь",
                    "process": "Плавка",
                    "outcome": "Успех",
                    "источник": "test.pdf",
                }
            ],
        }
    ]
    text = format_local_results(grouped)

    assert "Медь" in text
    assert "Плавка" in text
    assert "test.pdf" in text

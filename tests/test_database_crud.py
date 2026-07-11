import pytest
from sqlalchemy import text

from core.schemas import DocumentExtraction, KnowledgeFact, Parameter
from db.crud import save_extraction_to_db


@pytest.mark.asyncio
async def test_save_extraction_creates_fact(db_session):
    extraction = DocumentExtraction(
        facts=[
            KnowledgeFact(
                material="Медь",
                process="Плавка",
                geography="РФ",
                year=2023,
                parameters=[],
                outcome="Успешно",
                confidence_level="Высокий",
            )
        ]
    )

    await save_extraction_to_db(db_session, extraction, source_filename="test.pdf")

    result = await db_session.execute(text("SELECT * FROM facts WHERE material = 'Медь'"))
    rows = result.fetchall()

    assert len(rows) == 1
    assert rows[0].source_file == "test.pdf"


@pytest.mark.asyncio
async def test_save_extraction_creates_lower_columns(db_session):
    extraction = DocumentExtraction(
        facts=[
            KnowledgeFact(
                material="МЕДЬ",
                process="ПЛАВКА",
                geography="РФ",
                parameters=[],
                outcome="УСПЕШНО",
                confidence_level="Высокий",
            )
        ]
    )

    await save_extraction_to_db(db_session, extraction, "test.pdf")

    result = await db_session.execute(text("SELECT material_lower, process_lower, outcome_lower FROM facts"))
    row = result.fetchone()

    assert row.material_lower == "медь"
    assert row.process_lower == "плавка"
    assert row.outcome_lower == "успешно"


@pytest.mark.asyncio
async def test_save_extraction_creates_parameters_with_fact_id(db_session):
    extraction = DocumentExtraction(
        facts=[
            KnowledgeFact(
                material="Титан",
                process="Ковка",
                geography="США",
                year=2024,
                parameters=[
                    Parameter(name="ТЕМПЕРАТУРА", value_max=1000.0, unit="°C"),
                    Parameter(name="Давление", value_min=100.0, value_max=500.0, unit="МПа"),
                ],
                outcome="Успех",
                confidence_level="Средний",
            )
        ]
    )

    await save_extraction_to_db(db_session, extraction, "titan.pdf")

    fact_result = await db_session.execute(text("SELECT id FROM facts WHERE material = 'Титан'"))
    fact_id = fact_result.scalar()

    param_result = await db_session.execute(text("SELECT * FROM parameters WHERE fact_id = :fid"), {"fid": fact_id})
    params = param_result.fetchall()

    assert len(params) == 2
    assert params[0].name_lower == "температура"
    assert params[1].value_max == 500.0


@pytest.mark.asyncio
async def test_save_multiple_facts(db_session):
    extraction = DocumentExtraction(
        facts=[
            KnowledgeFact(
                material="Медь",
                process="Плавка",
                geography="РФ",
                parameters=[],
                outcome="Успех",
                confidence_level="Высокий",
            ),
            KnowledgeFact(
                material="Алюминий",
                process="Литье",
                geography="КНР",
                parameters=[],
                outcome="Брак",
                confidence_level="Низкий",
            ),
        ]
    )

    await save_extraction_to_db(db_session, extraction, "multi.pdf")

    from sqlalchemy import text

    result = await db_session.execute(text("SELECT COUNT(*) FROM facts"))
    count = result.scalar()

    assert count == 2

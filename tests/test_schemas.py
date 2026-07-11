import pytest
from pydantic import ValidationError

from core.schemas import DocumentExtraction, KnowledgeFact, Parameter


def test_valid_knowledge_fact():
    fact = KnowledgeFact(
        material="Медь",
        process="Плавка",
        geography="РФ",
        year=2023,
        parameters=[Parameter(name="Температура", value_max=1200.0, unit="°C")],
        outcome="Успешно",
        confidence_level="Высокий",
    )
    extraction = DocumentExtraction(facts=[fact])

    assert len(extraction.facts) == 1
    assert extraction.facts[0].material == "Медь"
    assert extraction.facts[0].parameters[0].value_max == 1200.0


def test_fact_without_year():
    fact = KnowledgeFact(
        material="Алюминий",
        process="Литье",
        geography="КНР",
        parameters=[],
        outcome="Брак",
        confidence_level="Низкий",
    )

    assert fact.year is None


def test_fact_with_multiple_parameters():
    fact = KnowledgeFact(
        material="Титан",
        process="Ковка",
        geography="США",
        year=2024,
        parameters=[
            Parameter(name="Температура", value_min=800.0, value_max=1000.0, unit="°C"),
            Parameter(name="Давление", value_max=500.0, unit="МПа"),
        ],
        outcome="Успешно",
        confidence_level="Высокий",
    )

    assert len(fact.parameters) == 2


def test_fact_rejects_missing_required_fields():
    with pytest.raises(ValidationError):
        KnowledgeFact(
            process="Плавка",
            geography="РФ",
            year=2023,
            parameters=[],
            outcome="Успешно",
            confidence_level="Высокий",
        )


def test_fact_rejects_invalid_year_type():
    with pytest.raises(ValidationError):
        KnowledgeFact(
            material="Медь",
            process="Плавка",
            geography="РФ",
            year="две тысячи двадцать три",
            parameters=[],
            outcome="Успешно",
            confidence_level="Высокий",
        )


def test_empty_extraction():
    extraction = DocumentExtraction(facts=[])

    assert extraction.facts == []

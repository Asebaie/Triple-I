import pytest

from core.extractor import extract_facts_from_text
from core.schemas import DocumentExtraction


@pytest.mark.asyncio
async def test_extractor_valid_json(mock_llm):
    mock_llm.return_value = """
    {
      "facts": [
        {
          "material": "Алюминий",
          "process": "Литье",
          "geography": "КНР",
          "year": 2022,
          "parameters": [],
          "outcome": "Брак",
          "confidence_level": "Низкий"
        }
      ]
    }
    """

    result = await extract_facts_from_text("Текст про алюминий")

    assert isinstance(result, DocumentExtraction)
    assert len(result.facts) == 1
    assert result.facts[0].material == "Алюминий"
    assert result.facts[0].outcome == "Брак"


@pytest.mark.asyncio
async def test_extractor_multiple_facts(mock_llm):
    mock_llm.return_value = """
    {
      "facts": [
        {"material": "Медь", "process": "Плавка", "geography": "РФ",
         "parameters": [], "outcome": "Успех", "confidence_level": "Высокий"},
        {"material": "Титан", "process": "Ковка", "geography": "США",
         "parameters": [], "outcome": "Брак", "confidence_level": "Средний"}
      ]
    }
    """

    result = await extract_facts_from_text("Текст")

    assert len(result.facts) == 2


@pytest.mark.asyncio
async def test_extractor_handles_broken_json(mock_llm):
    mock_llm.return_value = '{"facts": [{"material": "Медь", "process": "Плавка"'

    result = await extract_facts_from_text("Текст")

    assert result.facts == []


@pytest.mark.asyncio
async def test_extractor_handles_empty_response(mock_llm):
    mock_llm.return_value = ""

    result = await extract_facts_from_text("Текст")

    assert result.facts == []


@pytest.mark.asyncio
async def test_extractor_handles_invalid_schema(mock_llm):
    mock_llm.return_value = '{"wrong_key": "value"}'

    result = await extract_facts_from_text("Текст")

    assert result.facts == []


@pytest.mark.asyncio
async def test_extractor_passes_schema_to_llm(mock_llm):
    mock_llm.return_value = '{"facts": []}'

    await extract_facts_from_text("Текст")

    call_args = mock_llm.call_args
    system_prompt = call_args.kwargs.get("system_prompt") or call_args.args[0]

    assert "DocumentExtraction" in system_prompt or "facts" in system_prompt

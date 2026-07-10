from core.llm_client import ask_llm_json
from core.prompts import EXTRACTION_SYSTEM_PROMPT
from core.schemas import DocumentExtraction


async def extract_facts_from_text(raw_text: str) -> DocumentExtraction:
    json_schema = DocumentExtraction.model_json_schema()
    full_system_prompt = f"{EXTRACTION_SYSTEM_PROMPT}\n\nОжидаемая структура JSON:\n{json_schema}"

    json_response = await ask_llm_json(system_prompt=full_system_prompt, user_text=raw_text)

    try:
        return DocumentExtraction.model_validate_json(json_response)
    except Exception:
        return DocumentExtraction(facts=[])

from typing import List, Optional

from pydantic import BaseModel, Field


class Parameter(BaseModel):
    name: str = Field(description="Название параметра")
    value_min: Optional[float] = Field(None, description="Минимальное значение")
    value_max: Optional[float] = Field(None, description="Максимальное значение")
    unit: Optional[str] = Field(None, description="Единица измерения")


class KnowledgeFact(BaseModel):
    material: str = Field(description="Сырье или материал")
    process: str = Field(description="Технология или процесс")
    geography: str = Field(description="Регион применения")
    year: Optional[int] = Field(None, description="Год исследования")
    parameters: List[Parameter] = Field(description="Параметры процесса")
    outcome: str = Field(description="Вывод или результат")
    confidence_level: str = Field(description="Уровень достоверности")


class DocumentExtraction(BaseModel):
    facts: List[KnowledgeFact] = Field(description="Список технологических фактов")

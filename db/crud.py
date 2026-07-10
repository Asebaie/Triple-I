from sqlalchemy.ext.asyncio import AsyncSession

from core.schemas import DocumentExtraction
from db.models import FactModel, ParameterModel


async def save_extraction_to_db(db: AsyncSession, extraction: DocumentExtraction, source_filename: str):
    for fact_data in extraction.facts:
        db_fact = FactModel(
            material=fact_data.material,
            material_lower=fact_data.material.lower() if fact_data.material else "",
            process=fact_data.process,
            process_lower=fact_data.process.lower() if fact_data.process else "",
            geography=fact_data.geography,
            year=fact_data.year,
            outcome=fact_data.outcome,
            outcome_lower=fact_data.outcome.lower() if fact_data.outcome else "",
            confidence_level=fact_data.confidence_level,
            source_file=source_filename,
        )
        db.add(db_fact)
        await db.flush()

        for param_data in fact_data.parameters:
            db_param = ParameterModel(
                fact_id=db_fact.id,
                name=param_data.name,
                name_lower=param_data.name.lower() if param_data.name else "",
                value_min=param_data.value_min,
                value_max=param_data.value_max,
                unit=param_data.unit,
            )
            db.add(db_param)

    await db.commit()

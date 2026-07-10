import asyncio
import os
import sys

from sqlalchemy import text

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.extractor import extract_facts_from_text
from core.logger import logger
from db.crud import save_extraction_to_db
from db.database import AsyncSessionLocal, init_db
from parsers.document_parser import extract_text_from_docx, extract_text_from_pdf
from parsers.text_chunker import chunk_text


async def get_already_processed_files(db_session) -> set:
    try:
        result = await db_session.execute(text("SELECT DISTINCT source_file FROM facts;"))
        rows = result.fetchall()
        return {row[0] for row in rows if row[0]}
    except Exception:
        return set()


async def process_file(file_path: str, filename: str, db_session):
    logger.info(f"Начало обработки файла: {filename}")

    if filename.lower().endswith(".docx"):
        text = extract_text_from_docx(file_path)
    else:
        text = extract_text_from_pdf(file_path)

    if not text.strip():
        logger.warning(f"Файл {filename} пуст или не удалось извлечь текст")
        return

    chunks = chunk_text(text)
    logger.info(f"Файл {filename} разбит на {len(chunks)} частей")

    for i, chunk in enumerate(chunks):
        logger.info(f"Парсинг части {i + 1}/{len(chunks)} для {filename}...")
        extraction = await extract_facts_from_text(chunk)
        if extraction and extraction.facts:
            logger.info(f"Извлечено фактов: {len(extraction.facts)} из части {i + 1}")
            await save_extraction_to_db(db_session, extraction, filename)
        else:
            logger.warning(f"Факты не найдены в части {i + 1} для {filename}")

        await asyncio.sleep(2.0)


async def main(data_dir: str):
    logger.info("Запуск процесса импорта данных...")
    await init_db()

    async with AsyncSessionLocal() as db_session:
        processed_files = await get_already_processed_files(db_session)
        logger.info(f"Найдено уже обработанных файлов в базе: {len(processed_files)}")

        for root, _, files in os.walk(data_dir):
            for file in files:
                ext = file.lower()
                if ext.endswith(".pdf") or ext.endswith(".docx"):
                    if file in processed_files:
                        logger.info(f"Файл {file} уже обработан ранее. Пропуск...")
                        continue

                    file_path = os.path.join(root, file)
                    await process_file(file_path, file, db_session)

    logger.info("Процесс импорта данных завершен")


if __name__ == "__main__":
    TARGET_DIR = "./Источники информации"
    asyncio.run(main(TARGET_DIR))

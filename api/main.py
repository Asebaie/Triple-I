import asyncio

from fastapi import FastAPI

from api.routes import router
from core.logger import logger
from db.database import init_db

app = FastAPI(title="R&D Knowledge Graph API")


@app.on_event("startup")
async def startup_event():
    logger.info("Запуск API и инициализация базы данных...")
    await init_db()


app.include_router(router)

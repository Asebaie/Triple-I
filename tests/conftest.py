from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from db.models import Base


@pytest.fixture(scope="function")
async def engine():
    eng = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield eng
    await eng.dispose()


@pytest.fixture
async def db_session(engine):
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with async_session() as session:
        yield session


@pytest.fixture
def mock_llm():
    with patch("core.retriever.ask_llm_json", new_callable=AsyncMock) as mock_retriever, patch(
        "core.extractor.ask_llm_json", new_callable=AsyncMock
    ) as mock_extractor:

        class UnifiedMock:
            def __init__(self):
                self._return_value = None
                self._retriever = mock_retriever
                self._extractor = mock_extractor

            @property
            def return_value(self):
                return self._return_value

            @return_value.setter
            def return_value(self, value):
                self._return_value = value
                self._retriever.return_value = value
                self._extractor.return_value = value

            def __getattr__(self, name):
                return getattr(self._extractor, name)

        yield UnifiedMock()


@pytest.fixture(autouse=True)
def mock_logger():
    with patch("core.logger.logger") as mock_log:
        yield mock_log

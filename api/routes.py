from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from core.retriever import ask_question
from db.database import get_db

router = APIRouter()


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str


@router.post("/ask", response_model=QueryResponse)
async def ask_question_endpoint(request: QueryRequest, db: AsyncSession = Depends(get_db)):
    answer = await ask_question(db, request.query)
    return QueryResponse(answer=answer)

from fastapi import FastAPI
from pydantic import BaseModel
from src.core import get_answer

app = FastAPI()

class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    answer: str

@app.post("/ask")
def ask(request: AskRequest):
    answer = get_answer(request.question)
    return AskResponse(answer=answer)
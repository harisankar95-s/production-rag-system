from fastapi import FastAPI
from pydantic import BaseModel
from src.core import get_answer

app = FastAPI()

class AskRequest(BaseModel):
    question: str
    thread_id :str

class AskResponse(BaseModel):
    answer: str

@app.post("/ask")
def ask(request: AskRequest):
    answer = get_answer(request.question, request.thread_id)
    return AskResponse(answer=answer)
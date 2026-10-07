from fastapi import FastAPI
from pydantic import BaseModel
from app.llm.tool_chain import generate_tool_response
from fastapi.responses import FileResponse


app = FastAPI()


class QuestionRequest(BaseModel):
    question: str


@app.get("/")
def home():
    return FileResponse("app/static/index.html")


@app.post("/ask")
def ask_question(request: QuestionRequest):

    answer = generate_tool_response(request.question)

    return {
        "question": request.question,
        "answer": answer
    }
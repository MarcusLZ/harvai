from webbrowser import get
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from harvai.qa_model import get_answer

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def index():
    return {"Greeting": "Welcome to : HavAI - API"}

@app.get("/answer")
def answer(question, articles=1):
    # from the user input (question) and articles, get the answer from hugging face

    answer = get_answer(question)['answer']

    return {"question": question, "answer": answer }

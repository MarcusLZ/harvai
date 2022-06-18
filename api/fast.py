from webbrowser import get
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from harvai.qa_model import get_answer
import os


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

    return {"Greeting": "Welcome to harvai API"}

@app.get("/answer")
def answer(question, retriever="BM25", article_number=10):
    # from the user input (question) and articles, get the answer from hugging face

    answer, parsed_context = get_answer(question, retriever, int(article_number))

    return {"question": question, "answer": answer['answer'], "parsed_context" : parsed_context }

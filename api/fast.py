from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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

@app.get("/classifier")
def classifier(question):
    # from the user input (question) get relative to one or mutliple articles

    articles = "Please insert here articles"

    return {"question": question,
            "articles": articles }

@app.get("/answer")
def answer(question, articles=1):
    # from the user input (question) and articles, get the answer from hugging face

    answer = "Please insert here answer"

    return {"question": question,
            "answer": answer }

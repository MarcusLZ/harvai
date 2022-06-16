from contextvars import Context
from webbrowser import get
from transformers import pipeline

from harvai.data import preprocessing_user_input
from harvai.nn_model import Nn_model
from harvai.bm25 import Bm25
from harvai.dpr import DPR


def get_answer(question,retriever,article_number=1):
    """ Instanciate and use the transformer model"""

    context, parsed_context = get_context(question, retriever,article_number)
    model = pipeline('question-answering', model='etalab-ia/camembert-base-squadFR-fquad-piaf', tokenizer='etalab-ia/camembert-base-squadFR-fquad-piaf')

    return model({ 'question': question, 'context': context }) , parsed_context

def get_context(question, retriever,article_number):
    """calling the research model/function"""

    retriever_dictonnary =  {"KNN" : Nn_model(article_number), "BM25":Bm25(article_number), "DPR":DPR(article_number)}
    retriever = retriever_dictonnary[retriever]
    retriever.clean_data()
    retriever.fit()
    question = preprocessing_user_input(question)
    retriever.predict(question)
    context = retriever.get_articles_text_only()
    parsed_context = retriever.get_articles_parsed() # Liste d'articles

    return context, parsed_context

if __name__ == "__main__":

    test_answer = get_answer("quelle est la vitesse normale autorisée sur l'autoroute ?", "DPR",10)
    print (test_answer)

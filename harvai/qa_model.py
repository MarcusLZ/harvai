from contextvars import Context
from re import A
from webbrowser import get
from transformers import pipeline

from harvai.data import preprocessing_user_input
from harvai.nn_model import Nn_model
from harvai.bm25 import Bm25
from harvai.dpr import DPR
from harvai.embedding import Embedding


def get_answer(question,retriever,article_number,digits=False):
    """ Instanciate and use the transformer model"""

    context, parsed_context , article_reference = get_context(question, retriever,article_number,digits)
    model = pipeline('question-answering', model='etalab-ia/camembert-base-squadFR-fquad-piaf', tokenizer='etalab-ia/camembert-base-squadFR-fquad-piaf')

    return model({ 'question': question, 'context': context }) , parsed_context , context , article_reference

def get_context(question, retriever,article_number,digits=False):
    """calling the research model/function"""

    retriever_dictonnary =  {"KNN" : Nn_model(article_number,digits), "BM25":Bm25(article_number,digits), "DPR":DPR(article_number,digits), "Embedding":Embedding(article_number,digits)}
    retriever = retriever_dictonnary[retriever]
    retriever.clean_data()
    retriever.fit()
    question = preprocessing_user_input(question)
    retriever.predict(question)
    context = retriever.get_articles_text_only()
    parsed_context = retriever.get_articles_parsed() # Liste d'articles
    article_reference = retriever.get_article_reference()

    return context, parsed_context , article_reference

if __name__ == "__main__":


    answer, parsed_context, context,article_reference = get_answer("quelle est la vitesse normale autorisée sur l'autoroute ?", "KNN",2,digits=False)
    print (answer, parsed_context, context, article_reference)

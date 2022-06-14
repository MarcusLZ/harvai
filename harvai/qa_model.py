from webbrowser import get
from transformers import pipeline

from harvai.data import preprocessing_user_input
from harvai.nn_model import Nn_model
from harvai.bm25 import Bm25
from harvai.dpr import DPR


def get_answer(question, retriever="KNN"):
    """ Instanciate and use the transformer model"""

    context = get_context(question, retriever)
    model = pipeline('question-answering', model='etalab-ia/camembert-base-squadFR-fquad-piaf', tokenizer='etalab-ia/camembert-base-squadFR-fquad-piaf')

    return model({ 'question': question, 'context': context })


def get_context(question, retriever):
    """calling the research model/function"""

    retriever_dictonnary =  {"KNN" : Nn_model(), "BM25":Bm25(), "DPR":DPR()}
    retriever = retriever_dictonnary[retriever]
    retriever.clean_data()
    retriever.fit()
    question = preprocessing_user_input(question)
    retriever.predict(question)
    text = retriever.get_articles_text_only(article_number=3)

    return text

if __name__ == "__main__":

    test_answer = get_answer("quelle est la vitesse normale autorisée sur l'autoroute ?", "KNN")
    print (test_answer)

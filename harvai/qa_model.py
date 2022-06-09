from webbrowser import get
from transformers import pipeline
from harvai.nn_model import Nn_model


def get_answer(question, model="KNN"):
    """ Instanciate and use the transformer model"""

    context = get_context(question, model)
    model = pipeline('question-answering', model='etalab-ia/camembert-base-squadFR-fquad-piaf', tokenizer='etalab-ia/camembert-base-squadFR-fquad-piaf')

    return model({ 'question': question, 'context': context })


def get_context(question, model):
    """calling the research model/function"""

    if model == "KNN":
        model = Nn_model()
        model.clean_data()
        model.fit()
        model.predict(question)
        text = model.get_articles_text_only(article_number=3)
    elif model == "WordToVec":
        pass

    return text

if __name__ == "__main__":

    test_answer = get_answer("Que se passe t'il si je grille un feux rouge ?", "KNN")
    print (test_answer)

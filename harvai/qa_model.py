from webbrowser import get
from transformers import pipeline
from harvai.data import get_clean_preproc_data
from harvai.nn_model import Nn_model


def get_answer(question):
    """ Instanciate and use the transformer model"""

    context = get_context(question)

    model = pipeline('question-answering', model='etalab-ia/camembert-base-squadFR-fquad-piaf', tokenizer='etalab-ia/camembert-base-squadFR-fquad-piaf')

    return model({ 'question': question, 'context': context })


def get_context(question):
    """calling the research model/function"""
    df = get_clean_preproc_data().article_wo_punctuation

    model = Nn_model(df).fit()

    articles = model.predict(question)[0][0]

    return df[articles]

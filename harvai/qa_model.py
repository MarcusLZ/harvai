from transformers import pipeline
from harvai.data import get_clean_preproc_data
from harvai.nn_model import Nn_model


def get_asnwer(question, context):
    """ Instanciate and use the transformer model"""

    model = pipeline('question-answering', model='etalab-ia/camembert-base-squadFR-fquad-piaf', tokenizer='etalab-ia/camembert-base-squadFR-fquad-piaf')

    return model({ 'question': question, 'context': context })


def get_context(question):
    """calling the research model/function"""
    df = get_clean_preproc_data()

    model = Nn_model().fit()

    return model.predict(question)

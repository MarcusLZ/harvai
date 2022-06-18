import pandas as pd
from harvai.params import get_path_generated_question_dataset
import os


def score(model):
    """Evaluate retriever model on a dataset of generated questions, return the percentage of correct articles found"""

    dataset = pd.read_csv(get_path_generated_question_dataset(os.getcwd()))
    dataset.drop(columns='Unnamed: 0')
    score = 0
    for index,row in  dataset.iterrows():
        model.predict(row['questions'])
        if row['id'] in model.articles:
            score += 1

    return score/len(dataset)

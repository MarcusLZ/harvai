import pandas as pd
from harvai.params import get_path_generated_question_dataset
import os
from tqdm import tqdm


def score(model,dataset_portion=50,verbose=True):
    """Evaluate retriever model on a dataset of generated questions, return the percentage of correct articles found"""

    dataset = pd.read_csv(get_path_generated_question_dataset(os.getcwd()))
    dataset.drop(columns='Unnamed: 0')


    # keeps only a percentage of the dataset to score the model
    if dataset_portion > 100:
        dataset_portion = 100
    elif dataset_portion < 0:
        dataset_portion = 1

    last_row_number = int(len(dataset)*float(dataset_portion/100))

    score = 0
    for index,row in  tqdm(dataset.loc[0:last_row_number].iterrows(), total=last_row_number):
        model.predict(row['questions'])
        if row['id'] in model.articles:
            score += 1


    return score/len(dataset.loc[0:last_row_number])

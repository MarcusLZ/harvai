import pandas as pd
from harvai.params import get_path_generated_question_dataset
import os
from tqdm import tqdm
import numpy as np


def score(model,dataset_portion=100):
    """Evaluate retriever model on a dataset of generated questions, return the percentage of correct articles found"""

    dataset = pd.read_csv(get_path_generated_question_dataset(os.getcwd()))
    dataset.drop(columns='Unnamed: 0',inplace=True)


    # keeps only a percentage of the dataset to score the model
    if dataset_portion > 100:
        dataset_portion = 100
    elif dataset_portion < 0:
        dataset_portion = 1

    last_row_number = int(len(dataset)*float(dataset_portion/100))

    recall = 0
    position = []
    for index,row in  tqdm(dataset.loc[0:last_row_number].iterrows(), total=last_row_number):
        model.predict((row['questions_preproc']))
        if row['id'] in model.articles:
            recall += 1
            position.append(model.articles.index(row['id'])+1)



    return {'recall':recall/len(dataset.loc[0:last_row_number]),'average rank' : np.average(position)}

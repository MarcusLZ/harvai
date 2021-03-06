import numpy as np
from harvai.data import get_clean_preproc_data
from sentence_transformers import SentenceTransformer, util
from harvai.params import get_path_embedding
import os

class Embedding():
    def __init__(self, article_number,digits=False):
        self.data = None
        self.model = None
        self.doc_emb = None
        self.articles = None
        self.article_number = article_number
        self.digits = digits

    def clean_data(self):
        self.data = get_clean_preproc_data(self.digits)

    def fit(self):
        self.model = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-cos-v1',device ='cpu')
        self.doc_emb = np.load(get_path_embedding(os.getcwd()))


    def predict(self,question):
        query_emb = self.model.encode(question)
        scores = util.dot_score(query_emb, self.doc_emb)[0].cpu().tolist()
        self.data['score']=scores
        self.articles = self.data.sort_values(by='score', ascending=False).index[0:self.article_number].tolist()


    def get_articles_parsed(self): # Liste d'articles
        return self.data.sort_values(by='score', ascending=False)['article_content'][0:self.article_number].tolist()

    def get_article_reference(self):
        articles_references = []
        articles = self.articles
        for i in articles:
            articles_references.append(self.data.article_reference[i])
        return articles_references


    def get_articles_text_only (self):
        article_list = self.data.sort_values(by='score', ascending=False)['article_content'][0:self.article_number].tolist()
        return ''.join(article_list)


if __name__ == "__main__":

    test = Embedding(10)
    test.clean_data()
    test.fit()
    test.predict("quelle est la vitesse maximale sur l autoroute ?")
    print(test.data)
    print(test.articles)
    print(test.get_articles_parsed())

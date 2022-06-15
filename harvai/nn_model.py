from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from harvai.data import get_clean_preproc_data


class Nn_model():
    def __init__(self):
        self.data = None
        self.model = None
        self.vectorizer = None
        self.articles = None

    def clean_data(self):
        self.data = get_clean_preproc_data()

    def fit(self):
        self.vectorizer = TfidfVectorizer(max_df=0.8)
        features = self.vectorizer.fit_transform(self.data.article_lemmatized)

        self.model = NearestNeighbors(n_neighbors=10)
        self.model.fit(features)

    def predict(self,question):
        input = self.vectorizer.transform([question])
        self.articles = self.model.kneighbors(input, return_distance=False)

    def get_articles_parsed(self, article_number=10):
        articles_parsed = []
        article = self.articles[0][0:article_number]
        for i in article:
            articles_parsed.append(self.data.article_content[i])
        return articles_parsed

    def get_articles_text_only (self, article_number=10):
        if len(self.articles[0])< article_number :
            article = self.articles[0]
        else:
            article = self.articles[0][0:article_number]
        return ''.join(self.data.article_content[article])



if __name__ == "__main__":

    test = Nn_model()
    test.clean_data()
    test.fit()
    test.predict("quelle est la vitesse maximum sur l autoroute ?")
    print(test.get_articles_parsed())

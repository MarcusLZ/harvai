from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


class Nn_model(object):
    def __init__(self, data):
        self.data = data
        self.model = None
        self.vectorizer = None

    def fit(self):
        vec = TfidfVectorizer(max_df=0.8)
        features = vec.fit_transform(self.data)
        knn = NearestNeighbors(n_neighbors=10)
        knn.fit(features)
        self.model= knn
        self.vectorizer = vec
        return self

    def predict(self,sentence):
        return self.model.kneighbors(self.vectorizer.transform([sentence]), return_distance=False)

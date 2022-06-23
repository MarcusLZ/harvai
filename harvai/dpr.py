from harvai.data import get_clean_preproc_data
from haystack.document_stores.faiss import FAISSDocumentStore
from haystack.nodes import DensePassageRetriever
from haystack.pipelines import DocumentSearchPipeline
import os
from harvai.params import get_path_faiss, get_path_retriever


class DPR():
    def __init__(self,article_number,digits=False):
        self.data = None
        self.document_store = None
        self.model = None
        self.vectorizer = None
        self.articles = None
        self.article_number = article_number
        self.digits = digits

    def clean_data(self):
        self.data = get_clean_preproc_data(self.digits)
        df = self.data
        df['id'] = df.index
        df = df[['article_lemmatized','id']]
        df = df.to_dict(orient='index')
        formatted_data = []
        for key,text in df.items():
            formatted_data.append({'id':key,'content':text['article_lemmatized']})

        self.document_store = FAISSDocumentStore.load(get_path_faiss(os.getcwd()))


    def fit(self):
        self.model = DensePassageRetriever.load(get_path_retriever(os.getcwd()), document_store=self.document_store, use_gpu = False)

    def predict(self,question):
        p_retrieval = DocumentSearchPipeline(self.model)
        candidate_documents = p_retrieval.run(query=question, params={"Retriever": {"top_k":self.article_number}})
        self.articles = [int(candidate_documents['documents'][id].id) for id in range(0,self.article_number)]

    def get_articles_parsed(self): # Liste d'articles
        articles_parsed = []
        article = self.articles
        for i in article:
            articles_parsed.append(self.data.article_content[i])
        return articles_parsed

    def get_article_reference(self):
        articles_references = []
        articles = self.articles
        for i in articles:
            articles_references.append(self.data.article_reference[i])
        return articles_references

    def get_articles_text_only (self):
        return ''.join(self.data.article_content[self.articles])


if __name__ == "__main__":

    test = DPR(5)
    test.clean_data()
    test.fit()
    test.predict("quelle est la vitesse normale sur l autoroute ?")
    print(test.articles)
    print(test.get_articles_parsed())

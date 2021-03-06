from harvai.data import get_clean_preproc_data
from haystack.document_stores import ElasticsearchDocumentStore
from haystack.nodes import BM25Retriever



class Bm25():
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
        df = df[[preprocessing,'id']]
        df = df.to_dict(orient='index')
        formatted_data = []
        for key,text in df.items():
            formatted_data.append({'id':key,'content':text[preprocessing]})

        self.document_store = ElasticsearchDocumentStore(host="localhost", port="9200", username="", password="", index="document")
        self.document_store.write_documents(formatted_data)



    def fit(self):
        self.model = BM25Retriever(self.document_store)

    def predict(self,question):
        candidate_documents = self.model.retrieve(query=question,top_k=self.article_number)
        self.articles = [int(document.id) for document in candidate_documents]

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
        return ''.join(self.data.article_lowered[self.articles])




if __name__ == "__main__":

    test = Bm25(5)
    test.clean_data()
    test.fit()
    test.predict("quelle est la vitesse normale sur l autoroute ?")
    print(test.articles)
    #print(test.get_articles_text_only())

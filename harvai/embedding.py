from harvai.data import get_clean_preproc_data
from haystack.document_stores import ElasticsearchDocumentStore, InMemoryDocumentStore
from haystack.nodes import EmbeddingRetriever


class Embedding():
    def __init__(self):
        self.data = None
        self.document_store = None
        self.model = None
        self.vectorizer = None
        self.articles = None

    def clean_data(self):
        self.data = get_clean_preproc_data()
        df = self.data
        df['id'] = df.index
        df = df[['article_lowered','id']]
        df = df.to_dict(orient='index')
        formatted_data = []
        for key,text in df.items():
            formatted_data.append({'id':key,'content':text['article_lowered']})

        self.document_store = InMemoryDocumentStore(similarity="dot_product",embedding_dim=768)


        #self.document_store = ElasticsearchDocumentStore(host="localhost", port="9200", username="", password="", index="document", similarity="cosine", embedding_dim=768 )
        self.document_store.write_documents(formatted_data)



    def fit(self):
        self.model = EmbeddingRetriever( document_store=self.document_store,
        embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1",
        model_format="sentence_transformers",
        use_gpu = False,
        )
        self.document_store.update_embeddings(self.model)

    def predict(self,question):
        candidate_documents = self.model.retrieve(query=question)
        self.articles = [int(candidate_documents[id].id) for id in range(0,10)]

    def get_articles_text_only (self, article_number=1):
        return ''.join(self.data.article_lowered[self.articles])


if __name__ == "__main__":

    test = Embedding()
    test.clean_data()
    test.fit()
    test.predict("quelle est la vitesse maximum sur l autoroute ?")
    print(test.articles)
    #print(test.get_articles_text_only())

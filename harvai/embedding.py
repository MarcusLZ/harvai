from harvai.data import get_clean_preproc_data

from haystack.nodes import EmbeddingRetriever
from haystack.document_stores.faiss import FAISSDocumentStore
from haystack.pipelines import DocumentSearchPipeline


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

        self.document_store = FAISSDocumentStore.load(index_path="haystack_data/embedding/testfile_path")



    def fit(self):
        self.model = EmbeddingRetriever(document_store=self.document_store,
        embedding_model="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        model_format="sentence_transformers",
        use_gpu = False,
        )

    def predict(self,question):
        p_retrieval = DocumentSearchPipeline(self.model)
        candidate_documents = p_retrieval.run(query=question, params={"Retriever": {"top_k": 10}})
        self.articles = [int(candidate_documents['documents'][id].id) for id in range(0,10)]


    def get_articles_text_only (self, article_number=1):
        return ''.join(self.data.article_lowered[self.articles])


if __name__ == "__main__":

    test = Embedding()
    test.clean_data()

    test.fit()
    test.predict("quelle est la vitesse maximum sur l autoroute ?")
    print(test.articles)
    print(test.get_articles_text_only())

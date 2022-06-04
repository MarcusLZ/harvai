import pandas as pd
import PyPDF2 # PDF reader
import re # Regex


from harvai.params import DATA_LOCAL_PATH
from harvai.preprocessing import article_number,article_content,article_lower,remove_numbers,remove_punctuation,remove_stopwords, tfidf_format

def get_clean_preproc_data():
    code_brut = get_data()
    data = clean_data(code_brut)
    preproc_data = preprocessing_data(data)
    return preproc_data

def get_data(online=False):
    code_brut = ''
    if online == False :
        PDF = open(DATA_LOCAL_PATH,'rb')
        Reader = PyPDF2.PdfFileReader(PDF)
        for i in range(Reader.numPages):
            Pages = Reader.getPage(i)
            code_brut += Pages.extractText()
    return code_brut

def clean_data(code_brut):
    code = code_brut.replace("\n", " ") # retrait passage à la ligne
    code = re.sub("\'", " ", code) # retrait des apostrophes : \'
    code = code.replace("Code de la route. - Dernière modification le 01 juin 2022 - Document généré le 31 mai 2022","") # retrait bas de page
    articles = re.findall(r"(Article \w\d*-?\d.*?(?=Article))",code) # split la string par article pour en faire une liste d'articles

    # depuis la liste d'articles vers un dataframe
    dict_articles = {}
    for i in range(len(articles)):
        dict_articles[i] = articles[i]
    data = pd.DataFrame.from_dict(dict_articles, orient='index',
                       columns=['article_base'])
    return data

def preprocessing_data(data):
    data['article_number'] = data['article_base'].apply(lambda x : article_number(x))
    data['article_content'] = data['article_base'].apply(lambda x : article_content(x))
    data['article_lowered'] = data['article_content'].apply(lambda x : article_lower(x))
    data['article_wo_numbers'] = data['article_lowered'].apply(lambda x : remove_numbers(x))
    data['article_wo_punctuation'] = data['article_wo_numbers'].apply(lambda x : remove_punctuation(x))
    data['article_wo_stopwords'] = data['article_wo_punctuation'].apply(lambda x : remove_stopwords(x))
    data['article_tfidf_format'] = data['article_wo_stopwords'].apply(lambda x : tfidf_format(x))
    return data


if __name__ == '__main__':
    code_brut = get_data()

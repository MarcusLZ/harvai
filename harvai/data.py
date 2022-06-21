import pandas as pd
import PyPDF2 # PDF reader
import re # Regex
import os
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import simplemma

from harvai.params import get_path_data, get_path_json
from harvai.preprocessing import article_number,article_content,article_lower,remove_numbers,remove_punctuation,remove_stopwords, tfidf_format,Lemmatize

def get_clean_preproc_data():
    code_brut = get_data()
    cleaned_data = clean_data(code_brut)
    preproc_data = preprocessing_data(cleaned_data)
    return preproc_data

def get_data(online=False):
    code_brut = ''
    if online == False :
        PDF = open(get_path_data(os.getcwd()),'rb')
        Reader = PyPDF2.PdfFileReader(PDF)
        for i in range(Reader.numPages):
            Pages = Reader.getPage(i)
            code_brut += Pages.extractText()
    return code_brut

def clean_data(code_brut):
    # Regex de recherche des items à retirer
    partie_names = re.findall(r"(Partie .*)",code_brut)
    section_names = re.findall(r"(Section \d*\w*? : .*)",code_brut)
    soussection_names = re.findall(r"(Sous-section \d*\w*? : .*)",code_brut)
    livre_names = re.findall(r"(Livre \d*\w*? : .*)",code_brut)
    titre_names = re.findall(r"(Titre \d*\w*? : .*)",code_brut)
    chapitre_names = re.findall(r"(Chapitre \d*\w*? : .*)",code_brut)

    # Remplacements
    for i in partie_names:
        code_brut = code_brut.replace(i,"")
    for i in section_names:
        code_brut = code_brut.replace(i,"")
    for i in soussection_names:
        code_brut = code_brut.replace(i,"")
    for i in livre_names:
        code_brut = code_brut.replace(i,"")
    for i in titre_names:
        code_brut = code_brut.replace(i,"")
    for i in chapitre_names:
        code_brut = code_brut.replace(i,"")

    code_brut = code_brut.replace("Code de la route. - Dernière modification le 01 juin 2022 - Document généré le 31 mai 2022","") # retrait bas de page
    code_brut = code_brut.replace("\n", " ") # retrait passage à la ligne
    code_brut = re.sub("\'", " ", code_brut) # retrait des apostrophes : \'

    articles = re.findall(r"(Article \w\d*-?\d.*?(?=Article))",code_brut) # split la string par article pour en faire une liste d'articles

    # depuis la liste d'articles vers un dataframe
    dict_articles = {}
    for i in range(len(articles)):
        dict_articles[i] = articles[i]
    data = pd.DataFrame.from_dict(dict_articles, orient='index',
                       columns=['article_base'])
    return data

def preprocessing_data(data):
    if os.path.exists(get_path_json(os.getcwd())):
        data = pd.read_json(path_or_buf=get_path_json(os.getcwd()))
        return data
    else :
        data['article_number'] = data['article_base'].apply(lambda x : article_number(x))
        data['article_content'] = data['article_base'].apply(lambda x : article_content(x))
        data['article_lowered'] = data['article_content'].apply(lambda x : article_lower(x))
        data['article_wo_numbers'] = data['article_lowered'].apply(lambda x : remove_numbers(x))
        data['article_wo_punctuation'] = data['article_wo_numbers'].apply(lambda x : remove_punctuation(x))
        data['article_wo_stopwords'] = data['article_wo_punctuation'].apply(lambda x : remove_stopwords(x))
        data['article_tfidf_format'] = data['article_wo_stopwords'].apply(lambda x : tfidf_format(x))
        data['article_lemmatized'] = data['article_tfidf_format'].apply(lambda x : Lemmatize(x))
        data.to_json('../raw_data/data_preproc.json')
        return data

def preprocessing_user_input(user_input):
    user_input = user_input.lower() # article_lower
    user_input = ''.join([i for i in user_input if not i.isdigit()]) # remove number

    for punctuation in string.punctuation : # remove ponctuation
        user_input = user_input.replace(punctuation," ")

    stop_words = set(stopwords.words('french')) # remove_stopwords
    word_tokens = word_tokenize(user_input)
    user_input = [word for word in word_tokens if not word in stop_words]
    user_input = [word for word in user_input if len(word)>1]

    user_input = " ".join([str(word) for word in user_input]) # tfidf_format

    langdata = simplemma.load_data('fr') # Lemmatize
    user_input = simplemma.text_lemmatizer(user_input,langdata)
    user_input = " ".join([str(word) for word in user_input])

    return user_input

if __name__ == '__main__':
    code_brut = get_data()

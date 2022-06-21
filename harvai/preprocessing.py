import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import FrenchStemmer

def article_number(x): # return le numero de l'article
    return x[:17]

def article_content(x): # return l'article
    return x[17:]

def article_lower(x): # retirer les espaces à gauche // lowercase
    x = x.lstrip()
    return x.lower()

def remove_numbers(x): # retirer les digits
    return ''.join([i for i in x if not i.isdigit()])

def remove_punctuation(x): # retirer la ponctuation
    for punctuation in string.punctuation :
        x = x.replace(punctuation," ")
    return x

def remove_stopwords(x): # Retirer les Stopwords
    stop_words = set(stopwords.words('french'))
    word_tokens = word_tokenize(x)
    x = [word for word in word_tokens if not word in stop_words]
    x = [word for word in x if len(word)>1] # retrait des caractères spéciaux restants
    return x


def tfidf_format(x): # Mise en forme pour TfIdf
    x = " ".join([str(word) for word in x])
    return x

def Lemmatize(x): # Mise en forme pour TfIdf + Lemmatized
    stemmer = FrenchStemmer()
    x = stemmer.stem(x)
    return x

import nltk

def start_nltk():
    recursos = [
        "punkt_tab",
        "averaged_perceptron_tagger_eng",
        "stopwords",
        "wordnet",
        "omw-1.4",
        "sentiwordnet"
    ]    
    for recurso in recursos:
        nltk.download(recurso)
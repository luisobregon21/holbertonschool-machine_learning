#!/usr/bin/env python3
''' TF_IDF '''
from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    '''
    creates a TF-IDF embedding
    :sentences: is a list of sentences to analyze
    :vocab: is a list of the vocabulary words to use for the analysis
    '''
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(sentences)
    if vocab is None:
        vocab = vectorizer.get_feature_names_out()
    embedding = X.toarray()
    return embedding, vocab

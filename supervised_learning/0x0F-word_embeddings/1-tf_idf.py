#!/usr/bin/env python3
''' TF_IDF '''
from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    '''
    creates a TF-IDF embedding
    :sentences: is a list of sentences to analyze
    :vocab: is a list of the vocabulary words to use for the analysis
    '''
    vectorizer = TfidfVectorizer(vocabulary=vocab)
    if vocab is None:
        vocab = []
    embedding = vectorizer.fit_transform(sentences).toarray()
    vocab = list(vectorizer.get_feature_names())
    return embedding, vocab

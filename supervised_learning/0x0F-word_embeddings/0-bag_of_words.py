#!/usr/bin/env python3
''' Bag of Words'''
from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(sentences, vocab=None):
    '''
    creates a bag of words embedding matrix
    :sentences: is a list of sentences to analyze
    :vocab: is a list of the vocabulary words to use for the analysis
    '''
    vectorizer = CountVectorizer(vocabulary=vocab)
    # Learn the vocabulary dictionary and return document-term matrix.
    X = vectorizer.fit_transform(sentences)
    if vocab is None:
        vocab = []
    embedding = vectorizer.fit_transform(sentences).toarray()
    vocab = list(vectorizer.get_feature_names())
    return embedding, vocab

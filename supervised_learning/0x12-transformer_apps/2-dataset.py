#!/usr/bin/env python3
''' Dataset class '''
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset():
    ''' loads and preps a dataset for machine translation '''

    def __init__(self):
        ''' constructor '''
        data_train = tfds.load("ted_hrlr_translate/pt_to_en",
                               split="train",
                               as_supervised=True)
        data_valid = tfds.load("ted_hrlr_translate/pt_to_en",
                               split="validation",
                               as_supervised=True)
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            data_train)
        self.data_train = data_train.map(self.tf_encode)
        self.data_valid = data_valid.map(self.tf_encode)

    def tokenize_dataset(self, data):
        '''
        creates sub-word tokenizers for dataset
        :data: tf.data.Dataset whose examples are formatted as a tuple (pt, en)
        pt: is the tf.Tensor containing the Portuguese sentence
        en: is the tf.Tensor containing the corresponding English sentence
        '''
        SubwordTextEncoder = tfds.deprecated.text.SubwordTextEncoder
        tokenizer_pt = SubwordTextEncoder.build_from_corpus(
            (pt.numpy() for pt, en in data),
            target_vocab_size=(2 ** 15))
        tokenizer_en = SubwordTextEncoder.build_from_corpus(
            (en.numpy() for pt, en in data),
            target_vocab_size=(2 ** 15))
        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        '''
        encodes a translation into tokens.
        :pt: is the tf.Tensor containing the Portuguese sentence
        :en: is the tf.Tensor containing the corresponding English
        Returns:
                pt_tokens: contains the Portuguese tokens
                en_tokens: contains the English tokens
        '''
        pt_start_index = self.tokenizer_pt.vocab_size
        pt_end_index = pt_start_index + 1
        en_start_index = self.tokenizer_en.vocab_size
        en_end_index = en_start_index + 1
        pt_tokens = [pt_start_index] + self.tokenizer_pt.encode(
            pt.numpy()) + [pt_end_index]
        en_tokens = [en_start_index] + self.tokenizer_en.encode(
            en.numpy()) + [en_end_index]
        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """
        Act as a wrapper for encode.
        :pt (Tensor): contains the Portuguese sentence
        """
        pt_encoded, en_encoded = tf.py_function(func=self.encode,
                                                inp=[pt, en],
                                                Tout=[tf.int64, tf.int64])
        pt_encoded.set_shape([None])
        en_encoded.set_shape([None])
        return pt_encoded, en_encoded

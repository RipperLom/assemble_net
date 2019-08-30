#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author: GZhao_zhaoguanzhi
# datetime: 2019/8/29 5:57 PM
# software: PyCharm

import math

import tensorflow as tf

import layers


class EmbeddingLayer(object):
    """
    a layer class: embedding layer
    """

    def __init__(self, vocab_size, emb_dim):
        """
        init function
        """
        super(EmbeddingLayer, self).__init__()

        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        init_scope = math.sqrt(6.0 / (vocab_size + emb_dim))
        emb_shape = [self.vocab_size, self.emb_dim]
        self.embedding = tf.Variable(tf.random_uniform(
            emb_shape, -init_scope, init_scope))

    def ops(self, input_x):
        """
        operation
        """
        return tf.nn.embedding_lookup(self.embedding, input_x)


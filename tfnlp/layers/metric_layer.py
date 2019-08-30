#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author: GZhao_zhaoguanzhi
# datetime: 2019/8/29 6:07 PM
# software: PyCharm

import tensorflow as tf


class CosineLayer(object):
    """
    a layer class: cosine layer
    """

    def __init__(self):
        """
        init function
        """
        pass

    def ops(self, input_a, input_b):
        """
        operation
        """
        norm_a = tf.nn.l2_normalize(input_a, dim=1)
        norm_b = tf.nn.l2_normalize(input_b, dim=1)
        cos_sim = tf.expand_dims(tf.reduce_sum(tf.multiply(norm_a, norm_b), 1), -1)
        return cos_sim


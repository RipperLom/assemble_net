#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author: GZhao_zhaoguanzhi
# datetime: 2019/8/29 6:10 PM
# software: PyCharm

import tensorflow as tf


class ReluLayer(object):
    """
    a layer class: relu Activation function
    """
    def __init__(self):
        """
        init function
        """
        pass

    def ops(self, input_x):
        """
        operation
        """
        return tf.nn.relu(input_x)


class ConcatLayer(object):
    """
    a layer class: concat layer
    """

    def __init__(self):
        """
        init function
        """
        pass

    def ops(self, blobs, concat_size):
        """
        operation
        """
        return tf.reshape(tf.concat(blobs, 1), [-1, concat_size])

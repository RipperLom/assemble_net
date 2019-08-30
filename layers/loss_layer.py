#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author: GZhao_zhaoguanzhi
# datetime: 2019/8/29 6:14 PM
# software: PyCharm

import tensorflow as tf


class PairwiseHingeLoss(object):
    """
    a layer class: pairwise hinge loss
    """

    def __init__(self, config):
        """
        init function
        """
        self.margin = float(config["margin"])

    def ops(self, score_pos, score_neg):
        """
        operation
        """
        return tf.reduce_mean(tf.maximum(0., score_neg +
                                         self.margin - score_pos))


class PairwiseLogLoss(object):
    """
    a layer class: pairwise log loss
    """

    def __init__(self, config=None):
        """
        init function
        """
        pass

    def ops(self, score_pos, score_neg):
        """
        operation
        """
        return tf.reduce_mean(tf.nn.sigmoid(score_neg - score_pos))


class SoftmaxWithLoss(object):
    """
    a layer class: softmax loss
    """

    def __init__(self, config=None):
        """
        init function
        """
        pass

    def ops(self, pred, label):
        """
        operation
        """
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred,
                                                                      labels=label))


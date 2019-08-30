#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author: GZhao_zhaoguanzhi
# datetime: 2019/8/30 10:23 AM
# software: PyCharm

import tensorflow as tf


class AdamOptimizer(object):
    """
    a optimizer class: AdamOptimizer
    """

    def __init__(self, config=None):
        """
        init function
        """
        self.lr = float(config["learning_rate"])

    def ops(self):
        """
        operation
        """
        return tf.train.AdamOptimizer(learning_rate=self.lr)



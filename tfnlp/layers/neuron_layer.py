#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: GZhao_zhaoguanzhi
# datetime: 2019/8/29 4:59 PM
# software: PyCharm

import tensorflow as tf

# from tensorflow.contrib.rnn import GRUCell
# from tensorflow.contrib.rnn import LSTMCell
# from tensorflow.python.ops import array_ops
# from tensorflow.python.ops.rnn import dynamic_rnn as rnn
# from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn


class FCLayer(object):
    """
    a layer class: a fc layer implementation in tensorflow
    """

    def __init__(self, num_in, num_out):
        """
        init function
        """
        super(FCLayer, self).__init__()
        self.num_in = num_in
        self.num_out = num_out
        self.weight = tf.Variable(tf.random_normal([num_in, num_out]))
        self.bias = tf.Variable(tf.random_normal([num_out]))

    def ops(self, input_x):
        """
        operation
        """
        out_without_bias = tf.matmul(input_x, self.weight)
        output = tf.nn.bias_add(out_without_bias, self.bias)
        return output


class CNNLayer(object):
    """
    a layer class: A CNN layer implementation in tensorflow
    """
    
    def __init__(self, seq_len,
                 emb_dim, win_size, kernel_size): 
        """
        init function
        """
        super(CNNLayer, self).__init__()
        self.max_seq_len = seq_len
        self.emb_dim = emb_dim
        self.win_size = win_size
        self.kernel_size = kernel_size
        filter_shape = [self.win_size, self.emb_dim, 1, self.kernel_size]
        self.conv_w = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1),
                                  name="conv")
        self.bias = tf.Variable(tf.constant(0.1, shape=[self.kernel_size]),
                                name="bias")

    def ops(self, emb):
        """
        operation
        """
        emb_expanded = tf.expand_dims(emb, -1)
        conv = tf.nn.conv2d(emb_expanded, self.conv_w,
                            strides=[1, 1, 1, 1],
                            padding="VALID",
                            name="conv_op")
        h = tf.nn.bias_add(conv, self.bias)
        pool = tf.nn.max_pool(h,
                              ksize=[1, self.max_seq_len - self.win_size + 1,
                                     1, 1],
                              strides=[1, 1, 1, 1],
                              padding="VALID",
                              name="pool")
        pool_flat = tf.reshape(pool, [-1, self.kernel_size])
        return pool_flat


class GRULayer(object):
    """
    a layer class: a GRU layer
    """

    def __init__(self, hidden_size):
        """
        init function
        """
        super(GRULayer, self).__init__()
        self.hidden_size = hidden_size
        self.cell = tf.nn.rnn_cell.GRUCell(self.hidden_size)

    def ops(self, input_emb, seq_length=None):
        """
        operation
        """
        rnn_outputs, _ = tf.nn.dynamic_rnn(self.cell, inputs=input_emb,
                                           dtype=tf.float32, sequence_length=seq_length)
        return rnn_outputs


class LSTMLayer(object):
    """
    a layer class: a LSTM layer
    """

    def __init__(self, hidden_size):
        """
        init function
        """
        super(LSTMLayer, self).__init__()
        self.hidden_size = hidden_size
        self.cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size)

    def ops(self, input_emb, seq_length=None):
        """
        operation
        """
        rnn_outputs, _ = tf.nn.dynamic_rnn(self.cell, inputs=input_emb,
                                           dtype=tf.float32, sequence_length=seq_length)
        return rnn_outputs


class BRNNLayer(object):
    """
    a layer class: Bi-directional LSTM/GRU
    """

    def __init__(self, hidden_size, rnn_type='lstm'):
        """
        init function
        """
        super(BRNNLayer, self).__init__()
        self.hidden_size = hidden_size
        type_cast = {'lstm': tf.nn.rnn_cell.LSTMCell, 'gru': tf.nn.rnn_cell.GRUCell}
        if rnn_type not in type_cast:
            cell_type = tf.nn.rnn_cell.LSTMCell
        else:
            cell_type = type_cast[rnn_type]
        self.fw_cell = cell_type(
            num_units=self.hidden_size, state_is_tuple=True)
        self.bw_cell = cell_type(
            num_units=self.hidden_size, state_is_tuple=True)

    def ops(self, input_emb, seq_length=None):
        """
        operation
        """
        bi_outputs, bi_left_state = tf.nn.bidirectional_dynamic_rnn(self.fw_cell, self.bw_cell,
                                                                    input_emb, sequence_length=seq_length,
                                                                    dtype=tf.float32)
        seq_encoder = tf.concat(bi_outputs, -1)
        return seq_encoder


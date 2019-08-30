#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author: GZhao_zhaoguanzhi
# datetime: 2019/8/29 8:25 PM
# software: PyCharm

import sys
import time
import tensorflow as tf


from utils.utility import clazz


class Trainer(object):

    def __init__(self, config, transform, net, loss, optimizer):
        self.thread_num = int(config["thread_num"])
        self.model_path = config["model_path"]
        self.model_file = config["model_prefix"]
        self.print_iter = int(config["print_iter"])
        self.data_size = int(config["data_size"])
        self.batch_size = int(config["batch_size"])
        self.epoch_iter = int(self.data_size / self.batch_size)
        self.mode = config["training_mode"]

        self.optimizer = optimizer
        self.transform = transform
        self.net = net
        self.loss = loss

    def train(self, config):
        """
        train
        """

        if self.mode == "pointwise":
            input_l, input_r, label_y = self.transform.ops()
            pred = self.net.predict(input_l, input_r)
            loss = self.loss.ops(pred, label_y)
        elif self.mode == "pairwise":
            input_l, pos_input, neg_input = self.transform.ops()
            pos_score = self.net.predict(input_l, pos_input)
            neg_score = self.net.predict(input_l, neg_input)
            loss = self.loss.ops(pos_score, neg_score)
        else:
            print(sys.stderr, "training mode not supported")
            sys.exit(1)

        # define optimizer

        optimizer = self.optimizer.ops().minimize(loss)

        saver = tf.train.Saver(max_to_keep=None)
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        avg_cost = 0.0
        with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=self.thread_num,
                                              inter_op_parallelism_threads=self.thread_num)) as sess:
            sess.run(init)
            coord = tf.train.Coordinator()
            read_thread = tf.train.start_queue_runners(sess=sess, coord=coord)
            step = 0
            epoch_num = 1
            start_time = time.time()
            while not coord.should_stop():
                try:
                    step += 1
                    c, _ = sess.run([loss, optimizer])
                    avg_cost += c

                    if step % self.print_iter == 0:
                        print("loss: %f" % ((avg_cost / self.print_iter)))
                        avg_cost = 0.0
                    if step % self.epoch_iter == 0:
                        end_time = time.time()
                        print("save model epoch%d, used time: %d" % (epoch_num, end_time - start_time))
                        save_path = saver.save(sess, "%s/%s.epoch%d" % (self.model_path, self.model_file, epoch_num))
                        epoch_num += 1
                        start_time = time.time()

                except tf.errors.OutOfRangeError:
                    save_path = saver.save(sess, "%s/%s.final" % (self.model_path, self.model_file))
                    coord.request_stop()
            coord.join(read_thread)
        sess.close()

    def predict(self, config):
        """
        predict
        """
        config.update({"num_epochs": "1", "batch_size": "1", "shuffle": "0", "train_file": config["test_file"]})

        test_l, test_r, label = self.transform.ops()
        # test network
        pred = self.net.predict(test_l, test_r)

        mean_acc = 0.0
        saver = tf.train.Saver()
        label_index = tf.argmax(label, 1)
        if self.mode == "pointwise":
            pred_prob = tf.nn.softmax(pred, -1)
            score = tf.reduce_max(pred_prob, -1)
            pred_index = tf.argmax(pred_prob, 1)
            correct_pred = tf.equal(pred_index, label_index)
            acc = tf.reduce_mean(tf.cast(correct_pred, "float"))
        elif self.mode == "pairwise":
            score = pred
            pred_index = tf.argmax(pred, 1)
            acc = tf.constant([0.0])
        modelfile = config["test_model_file"]
        result_file = open(config["test_result"], "w")
        step = 0
        init = tf.group(tf.global_variables_initializer(),
                        tf.local_variables_initializer())
        with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=1)) \
                as sess:
            sess.run(init)
            saver.restore(sess, modelfile)
            coord = tf.train.Coordinator()
            read_thread = tf.train.start_queue_runners(sess=sess, coord=coord)
            while not coord.should_stop():
                step += 1
                try:
                    ground, pi, a, prob = sess.run([label_index, pred_index, acc, score])
                    mean_acc += a
                    for i in range(len(prob)):
                        result_file.write("%d\t%d\t%f\n" % (ground[i], pi[i], prob[i]))
                except tf.errors.OutOfRangeError:
                    coord.request_stop()
            coord.join(read_thread)
        sess.close()
        result_file.close()
        if self.mode == "pointwise":
            mean_acc = mean_acc / step
            print(sys.stderr, "accuracy: %4.2f" % (mean_acc * 100))


#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author: GZhao_zhaoguanzhi
# datetime: 2019/8/29 5:42 PM
# software: PyCharm


import argparse
import sys

from utils.utility import clazz
from utils.utility import load_config

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='train',
                        help='task: train/predict, the default value is train.')
    parser.add_argument('--task_conf', default='./examples/cnn-pointwise.json',
                        help='task_conf: config file for this task')
    args = parser.parse_args()

    # 加载参数
    conf_dir = args.task_conf
    config = load_config(conf_dir)

    # 选择类进行组装
    transform = clazz(config, 'transform_py', 'transform_class')(config)
    net = clazz(config, 'net_py', 'net_class')(config)
    loss = clazz(config, "loss_py", "loss_class")(config)
    optimizer = clazz(config, 'optimizer_py', 'optimizer_class')(config)
    trainer = clazz(config, "trainer_py", "trainer_class")(config, transform, net, loss, optimizer)

    if args.task == 'train':
        trainer.train(config)
    elif args.task == 'predict':
        trainer.predict(config)
    else:
        print(sys.stderr, 'task type error.')


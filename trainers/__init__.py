#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author: GZhao_zhaoguanzhi
# datetime: 2019/8/29 4:53 PM
# software: PyCharm

# -*- coding=utf-8 -*-
from abc import abstractmethod

import tensorflow as tf
import os
import shutil
import logging
import json


class Trainer(object):
    def __init__(self):
        pass

    def train(self):
        pass

    @abstractmethod
    def train_step(self, batch_data):
        pass

    @abstractmethod
    def dev_step(self, batch_data):
        pass

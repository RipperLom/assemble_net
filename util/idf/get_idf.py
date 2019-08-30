#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
# 
# Copyright (c) 2019 aibot.me, Inc. All Rights Reserved
# 
########################################################################
 
"""
File: get_idf.py
Author: wangyan
Date: 2019/05/09 11:53:24
Brief: idf 词典
       输入分好词的句子
       https://radimrehurek.com/gensim/models/tfidfmodel.html
"""


import sys
import collections
from gensim import models


def load_stopword(file_name):
    stop_list = []
    for line in open(file_name):
        line = line.rstrip("\r\n")
        stop_list.append(line)
    return stop_list


# 计算idf
def get_idf(data_file):
    doc_freq_cnt = collections.Counter()
    doc_num = 0
    for line in open(file_name):
        doc_num += 1
        line = line.rstrip("\r\n")
        for tok in set(line.split("\t")):
            doc_freq_cnt[tok] += 1
    #dump
    for tok in doc_freq_cnt:
        cnt = doc_freq_cnt[tok]
        idf = models.tfidfmodel.df2idf(cnt, doc_num, log_base=10.0, add=0.001)
        print("%s\t%f" % (tok, idf))
    return True


if __name__ == "__main__":
    file_name = sys.argv[1]
    get_idf(file_name)


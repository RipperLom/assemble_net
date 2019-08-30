#! /bin/bash

#####################################################################
## Copyright (c) 2018 aibot.me, Inc. All Rights Reserved 
## @file run_w2v.sh
## @author wangyan
## @date 2018/04/28 14:53:23
## @version 1.0.0.0 
## @brief: dep bash
#####################################################################

set -x
set -e


#参数
BIN_DIR=./bin_word2vec
CORPUS=../input_seg.txt
MODEL=model_w2v_skip_200
THREAD_NUM=30


#架构：skip-gram（慢、对罕见字有利、效果好些）vs CBOW（快）
#训练算法：分层softmax（对罕见字有利）vs 负采样（对常见词和低纬向量有利）
#欠采样频繁词：可以提高结果的准确性和速度（适用范围1e-3到1e-5）
#文本（window）大小：skip-gram通常在10附近，CBOW通常在5附近
#向量大小：200 ~ 400


echo "-------------------------------------"
echo "Training vectors..."

#大数据量
function big_data()
{
	#skip
	time ${BIN_DIR}/w2v_bin -train ${CORPUS} -output ${MODEL}_skip -cbow 0 -size 200 \
            -window 5 -negative 5 -hs 0 -sample 1e-4  \
            -threads ${THREAD_NUM}  -binary 1 -min-count 5 -iter 20
	
	return 0

	#cbow
	time ${BIN_DIR}/w2v_bin -train ${CORPUS} -output ${MODEL}_cbow -cbow 1 -size 200 \
            -window 8 -negative 5 -hs 0 -sample 1e-4  \
            -threads ${THREAD_NUM}  -binary 0 -min-count 5 -iter 10
	return 0
}


##### run all ######
big_data



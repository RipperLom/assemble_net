#! /bin/bash

set -x
set -e


#### 参数
BIN_DIR=bin_glove
CORPUS=../input_seg.txt
SAVE_FILE=model_glove


VOCAB_FILE=tmp_vocab.txt
COOC_FILE=tmp_cooc.bin
COOC_SHUF_FILE=tmp_cooc.shuf.bin

VERBOSE=2
MEMORY=1.0         #内存大小
VECTOR_SIZE=50     #向量大小
WINDOW_SIZE=15     #窗口大小
MAX_ITER=15        #迭代次数
BINARY=2           #是否二进制
X_MAX=10           #
NUM_THREADS=8      #线程数


#run all
#统计次频
${BIN_DIR}/vocab_count -min-count 5 -verbose ${VERBOSE} < ${CORPUS} > ${VOCAB_FILE}

#统计共现
${BIN_DIR}/cooccur -memory $MEMORY -vocab-file ${VOCAB_FILE} -verbose $VERBOSE \
        -window-size $WINDOW_SIZE < ${CORPUS} > ${COOC_FILE}

#shuffle
${BIN_DIR}/shuffle -memory $MEMORY -verbose $VERBOSE < ${COOC_FILE} > ${COOC_SHUF_FILE}

#cooc2vec
${BIN_DIR}/glove -save-file ${SAVE_FILE} -threads ${NUM_THREADS} -input-file ${COOC_SHUF_FILE} \
        -x-max $X_MAX -iter $MAX_ITER -vector-size $VECTOR_SIZE -binary $BINARY \
        -vocab-file $VOCAB_FILE -verbose $VERBOSE



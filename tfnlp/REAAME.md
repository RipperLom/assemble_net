tfnlp
=====================

## 模块介绍

TensorFlow版本Assemble，提供了

* 多种基础层 layers：embedding_layer 嵌入层、neuron_layer 神经网络层（CNN、RNN、LSTM、GRU、MultiHeadsAttention、TwoStreamAttention）、mezz_layer 中间层（激活、随机失活、批量标准化、池化）、predict_layer 预测层、loss_layer 损失层、metric_layer 度量层（ACC、AUC）等

* 多种模块 blocks：transfer_block 迁移模块、cnn_block cnn模块、rnn_block rnn模块、classic_block 传统模块（LeNet、Inception、ResNet、Transformer）、hybrid_block cnn和rnn结合模块(OCR)等

* 多种网络 net：layer和block的组合，block之间具有不同的学习率和失活率

可以通过配置文件的形式灵活选择您需要的网络结构，损失函数，训练方式。


```
.
├─layers─┐                      基础层
│        ├─embed_layer.py       嵌入层
│        ├─neuron_layer.py      神经网络层
│        ├─mezz_layer.py        中间层
│        ├─predict_layer.py     预测层
│        ├─loss_layer.py        损失层
│        └─metric_layer.py      度量层
│        
├─blocks─┐                      模块
│        ├─transfer_block.py    迁移模块
│        ├─cnn_block.py         cnn模块
│        ├─rnn_block.py         rnn模块
│        ├─hybrid_block.py      cnn和rnn结合模块
│        └─classic_block .py    传统模块
│        
├─nets───┐                      多种网络
│        ├─blstm.py             blstm
│        ├─bgru.py              bgru
│        └─drmm.py              drmm
│        
├─optimizers                    优化器
│        └─multi_optimizers.py  多模块优化器
│        
├─transforms                    数据预处理
│        └─transform.py         预处理
│        
└─trains─┐                      训练
         └─train.py             工具

## 使用说明

运行assemble_net.py
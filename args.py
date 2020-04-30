import os
from os import path as op
file_path = os.path.dirname(__file__)

model_dir = os.path.join(file_path, 'pretrained/bert_wwm')
config_name = os.path.join(model_dir, 'bert_config.json')
ckpt_name = os.path.join(model_dir, 'bert_model.ckpt')
output_dir = os.path.join(file_path, 'result')
vocab_file = os.path.join(model_dir, 'vocab.txt')
data_dir = os.path.join(file_path, 'extra_data/')

do_lower_case=True
num_labels=2
warmup_steps=400
keras_model_path=op.join(output_dir,'keras_model.h5')

num_train_epochs = 10
batch_size = 16
learning_rate = 0.000025
prune_enabled=False
prune_logdir=os.path.join(file_path,'prune_log')

# gpu使用率
gpu_memory_fraction = 0.8

# 默认取倒数第二层的输出值作为句向量
layer_indexes = [-2]

# 序列的最大程度，单文本建议把该值调小
max_seq_len = 55

# graph名字
graph_file = 'tmp/result/graph'

pool_strategy = 'cls'

enable_early_stopping = False
This Tensorflow fork is based on the seq2seq tutorial and attempts to reproduce the archiecture of "Neural Machine Translation by jointly learning to align and translate" (Bahdanau et al 2015).

A typical config.ini file looks like this:  
use_lstm: True  
embedding_size: 620  
hidden_size: 1000  
num_layers: 1  
num_samples: 0  
norm_digits: False  
use_seqlen: True  
use_src_mask: True  
maxout_layer: True  
encoder: bidirectional  
opt_algorithm: sgd  
learning_rate: 1.0  
learning_rate_decay_factor: 0.99  
adjust_lr: False  
max_gradient_norm: 1.0  
batch_size: 80  
src_lang: en  
trg_lang: de  
max_sequence_length: 50  
train_sequential: True  
num_symm_buckets: 5  
add_src_eos: True  
no_pad_symbol: True  
init_backward: True  
variable_prefix: nmt  
max_train_batches: 800000  
eval_bleu: True  
eval_bleu_start: 10000  
eval_frequency: 10  
single_src_embedding: True  
src_vocab_size: 53981  
trg_vocab_size: 53971  

The trainer is called like this (all data is expected to have been integer-mapped):  
python $tensorflow/tensorflow/models/rnn/translate/train.py \  
--train_dir=$train_dir \  
--train_src_idx=train.ids.src \  
--train_trg_idx=train.ids.trg \  
--dev_src_idx=dev.ids.src \  
--dev_trg_idx=dev.ids.trg \  
--config_file=config.ini

The bag-to-sequence model presented in the paper "A Comparison of Neural Models for Word Ordering" (Hasler et al. 2017, http://aclweb.org/anthology/W17-3531) can be trained by setting the encoder to 'bow'. A sample config.ini file for the bag-to-sequence model is shown below. The trainer is called in the same way as above with source and target data are in the same language. The source data can optionally be shuffled as source sequence information is not encoded in the model.

use_lstm: True  
embedding_size: 620  
hidden_size: 1000  
num_layers: 1  
num_samples: 0  
norm_digits: False  
use_src_mask: True  
maxout_layer: True  
encoder: bow  
opt_algorithm: sgd  
learning_rate: 1.0  
learning_rate_decay_factor: 0.99  
adjust_lr: False  
max_gradient_norm: 1.0  
batch_size: 80  
src_vocab_size: 50003  
trg_vocab_size: 50003  
src_lang: de  
trg_lang: de  
max_sequence_length: 50  
train_sequential: True  
num_symm_buckets: 5  
add_src_eos: False  
no_pad_symbol: True  
bow_init_const: False  
use_bow_mask: True  

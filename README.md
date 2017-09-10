This Tensorflow fork is based on the seq2seq tutorial and attempts to reproduce the archiecture of "Neural Machine Translation by jointly learning to align and translate" (Bahdanau et al 2015).

A typical config file looks like this:
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

The trainer is called like this:
python $tensorflow/tensorflow/models/rnn/translate/train.py
--train_dir=$train_dir
--train_src_idx=train.ids.src
--train_trg_idx=train.ids.trg
--dev_src_idx=dev.ids.src
--dev_trg_idx=dev.ids.trg
--config_file=config.ini

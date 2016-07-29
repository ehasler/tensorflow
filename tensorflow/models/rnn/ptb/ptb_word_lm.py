# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Example / benchmark for building a PTB LSTM model.

Trains the model described in:
(Zaremba, et. al.) Recurrent Neural Network Regularization
http://arxiv.org/abs/1409.2329

There are 3 supported model configurations:
===========================================
| config | epochs | train | valid  | test
===========================================
| small  | 13     | 37.99 | 121.39 | 115.91
| medium | 39     | 48.45 |  86.16 |  82.07
| large  | 55     | 37.87 |  82.62 |  78.29
The exact results may vary depending on the random initialization.

The hyperparameters used in the model:
- init_scale - the initial scale of the weights
- learning_rate - the initial value of the learning rate
- max_grad_norm - the maximum permissible norm of the gradient
- num_layers - the number of LSTM layers
- num_steps - the number of unrolled steps of LSTM
- hidden_size - the number of LSTM units
- max_epoch - the number of epochs trained with the initial learning rate
- max_max_epoch - the total number of epochs for training
- keep_prob - the probability of keeping weights in the dropout layer
- lr_decay - the decay of the learning rate for each epoch after "max_epoch"
- batch_size - the batch size

The data required for this example is in the data/ dir of the
PTB dataset from Tomas Mikolov's webpage:

$ wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
$ tar xvf simple-examples.tgz

To run:

$ python ptb_word_lm.py --data_dir=simple-examples/data/

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import datetime
import logging
import pickle
import copy
import tensorflow as tf

from tensorflow.models.rnn.ptb import reader
from tensorflow.models.rnn.ptb.utils import model_utils, train_utils

flags = tf.flags
flags.DEFINE_string("model", None, "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("config_file", None, "Instead of selecting a predefined model, pass options in a config file")
flags.DEFINE_string("data_dir", None, "Path to dir containing PTB training data")
flags.DEFINE_string("train_dir", None, "Training directory.")
flags.DEFINE_string("train_idx", None, "Path to training data (integer-mapped)")
flags.DEFINE_string("dev_idx", None, "Path to development data (integer-mapped)")
flags.DEFINE_string("test_idx", None, "Path to test data (integer-mapped)")
flags.DEFINE_integer("max_train_data_size", 0, "Limit on the size of training data (0: no limit).") 
flags.DEFINE_string("device", None, "Device to be used")
flags.DEFINE_string("optimizer", "sgd", "Optimizer: sgd/adadelta/adagrad/adam/rmsprop (default: sgd)")
flags.DEFINE_integer("steps_per_checkpoint", 200, "How many training steps to do per checkpoint.")
flags.DEFINE_boolean("score", False, "Run rnnlm on test sentence and report logprobs")
flags.DEFINE_boolean("fixed_random_seed", False, "If True, use a fixed random seed to make training reproducible (affects matrix initialization)")

flags.DEFINE_string("variable_prefix", "model", "Set variable prefix for all model variables")
flags.DEFINE_string("rename_variable_prefix", None, "Rename variable prefix for all model variables")
flags.DEFINE_string("model_path", None, "Path to current model")
flags.DEFINE_string("new_model_path", None, "Path to model with renamed variables")

FLAGS = flags.FLAGS

def main(_):
  if FLAGS.rename_variable_prefix:
    if not FLAGS.model_path or not FLAGS.new_model_path:
      logging.error("Must set --model_path and --new_model_path to rename model variables")
      exit(1)
  else:
    if not FLAGS.train_dir:
      logging.error("Must set --train_dir")
      exit(1)
    if not FLAGS.data_dir and (not FLAGS.train_idx or not FLAGS.dev_idx):
      logging.error("Must set --data_dir to PTB data directory or specify data using --train_idx,--dev_idx")
      exit(1)

  logging.getLogger().setLevel(logging.INFO)
  logging.info("Start: {}".format(datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')))

  device = "/gpu:0"
  log_device_placement = False
  allow_soft_placement = True
  if FLAGS.device:
    device = '/'+FLAGS.device
  logging.info("Use device %s" % device)

  with tf.Graph().as_default(), tf.Session(config=tf.ConfigProto(allow_soft_placement=allow_soft_placement, log_device_placement=log_device_placement)) \
    as session, tf.device(device):

    if FLAGS.rename_variable_prefix:
      model_utils.rename_variable_prefix(session, FLAGS.config_file, FLAGS.model_path, FLAGS.new_model_path,
                           FLAGS.variable_prefix, FLAGS.rename_variable_prefix)
    elif FLAGS.score:
      logging.info("Run model in scoring mode")
      use_log_probs = True
      train_dir = "train.rnn.de"
      model, _ = model_utils.load_model(session, "large50k", train_dir, use_log_probs)

      #test_path = os.path.join(FLAGS.data_dir, "test15/test15.ids50003.de")
      #test_data = reader.read_indexed_data(test_path)
      #test_sentences = [ test_data ]
  
      # Add eos symbol to the beginning to score first word as well
      test_sentences = [[2, 5, 3316, 7930, 7, 7312, 9864, 30, 8, 10453, 4, 2],
                        [2, 7, 5, 30, 8, 10453, 7930, 3316, 7312, 9864, 4, 2],
                        [2, 5, 8, 30, 7, 4, 9864, 3316, 7312, 7930, 10453, 2],
                        [2, 8, 10453, 9864, 30, 5, 3316, 7312, 7, 7930, 4]]
      for test_data in test_sentences:
        # using log probs or cross entropies gives the same perplexities
        if use_log_probs:
          # Run model as in training, with an iterator over inputs
          train_utils.run_epoch_eval(session, model, test_data, tf.no_op(), use_log_probs=use_log_probs)
          # Run model step by step (yields the same result)
          #score_sentence(session, model, test_data)      
        else:
          train_utils.run_epoch_eval(session, model, test_data, tf.no_op(), use_log_probs=use_log_probs)
    else:
      logging.info("Run model in training mode")
      if FLAGS.fixed_random_seed:
        tf.set_random_seed(1234)

      if FLAGS.model:
        config = model_utils.get_config(FLAGS.model)
        eval_config = model_utils.get_config(FLAGS.model)
      elif FLAGS.config_file:
        config = model_utils.read_config(FLAGS.config_file)
        eval_config = copy.copy(config)
      else:
        logging.error("Must specify either model name or config file.")
        exit(1)

      eval_config.batch_size = 1
      eval_config.num_steps = 1
      model, mvalid, mtest = model_utils.create_model(session, config, eval_config, FLAGS.train_dir, FLAGS.optimizer)

      # Restore saved train variable
      start_epoch = 1
      start_idx = 0
      start_state = None
      tmpfile = FLAGS.train_dir+"/tmp_idx.pkl"
      if model.global_step.eval() >= FLAGS.steps_per_checkpoint and \
        os.path.isfile(tmpfile):
          with open(tmpfile, "rb") as f:
            start_epoch, start_idx, start_state = pickle.load(f)
            logging.info("Restore saved train variables from %s, resume from epoch=%i and train idx=%i and last state" % (tmpfile, start_epoch, start_idx))

      if FLAGS.data_dir:
        raw_data = reader.ptb_raw_data(FLAGS.data_dir)
        train_data, valid_data, test_data, _ = raw_data
      else:
        train_data = reader.read_indexed_data(FLAGS.train_idx, FLAGS.max_train_data_size, config.vocab_size)
        valid_data = reader.read_indexed_data(FLAGS.dev_idx, vocab_size=config.vocab_size)
        if FLAGS.test_idx:
          test_data = reader.read_indexed_data(FLAGS.test_idx, vocab_size=config.vocab_size)

      for epoch in range(start_epoch, config.max_max_epoch+1):
        if not (FLAGS.optimizer == "adadelta" or FLAGS.optimizer == "adam"):
          if start_idx == 0:
            lr_decay = config.lr_decay ** max(epoch - config.max_epoch+1, 0.0)
            model.assign_lr(session, config.learning_rate * lr_decay)
        logging.info("Epoch: %d Learning rate: %.3f" % (epoch, session.run(model.lr)))

        train_perplexity = train_utils.run_epoch(session, model, train_data, model.train_op, FLAGS.train_dir, FLAGS.steps_per_checkpoint,
                                                 train=True, start_idx=start_idx, start_state=start_state, tmpfile=tmpfile, m_valid=mvalid,
                                                 valid_data=valid_data, epoch=epoch)
        if start_idx == 0:
          logging.info("Epoch: %d Train Perplexity: %.3f" % (epoch, train_perplexity))
        else:
          logging.info("Epoch: %d Train Perplexity: %.3f (incomplete)" % (epoch, train_perplexity))
        start_idx = 0
        start_state = None

        valid_perplexity = train_utils.run_epoch(session, mvalid, valid_data, tf.no_op(), FLAGS.train_dir, FLAGS.steps_per_checkpoint)
        logging.info("Epoch: %d Full Valid Perplexity: %.3f" % (epoch, valid_perplexity))

      logging.info("Training finished.")
      if FLAGS.data_dir or FLAGS.test_idx:
        test_perplexity = train_utils.run_epoch(session, mtest, test_data, tf.no_op(), FLAGS.train_dir, FLAGS.steps_per_checkpoint)
        logging.info("Test Perplexity: %.3f" % test_perplexity)

      checkpoint_path = os.path.join(FLAGS.train_dir, "rnn.ckpt")
      logging.info("Save final model to path=%s" % checkpoint_path)
      model.saver.save(session, checkpoint_path, global_step=model.global_step)

    logging.info("End: {}".format(datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')))

if __name__ == "__main__":
  tf.app.run()

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

import sys, os
import datetime
import logging
import pickle
import copy
import tensorflow as tf

from tensorflow.models.rnn.ptb import reader
from tensorflow.models.rnn.ptb.utils import model_utils, train_utils
from tensorflow.models.rnn import rnn

flags = tf.flags
flags.DEFINE_string("model", None, "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("config_file", None, "Instead of selecting a predefined model, pass options in a config file")
flags.DEFINE_string("data_dir", None, "Path to dir containing PTB training data")
flags.DEFINE_string("train_dir", "/tmp", "Training directory.")
flags.DEFINE_string("train_idx", None, "Path to training data (integer-mapped)")
flags.DEFINE_string("dev_idx", None, "Path to development data (integer-mapped)")
flags.DEFINE_string("test_idx", None, "Path to test data (integer-mapped)")
flags.DEFINE_integer("max_train_data_size", 0, "Limit on the size of training data (0: no limit).") 
flags.DEFINE_string("device", None, "Device to be used")
flags.DEFINE_boolean("use_adadelta", False, "Use AdaDelta instead of GradientDescent")
flags.DEFINE_boolean("use_adagrad", False, "Use AdaGrad instead of GradientDescent")
flags.DEFINE_boolean("use_adam", False, "Use Adam instead of GradientDescent")
flags.DEFINE_boolean("use_rmsprop", False, "Use RmsProp instead of GradientDescent")
flags.DEFINE_integer("steps_per_checkpoint", 200, "How many training steps to do per checkpoint.")
flags.DEFINE_boolean("score", False, "Run rnnlm on test sentence and report logprobs")
flags.DEFINE_boolean("fixed_random_seed", False, "If True, use a fixed random seed to make training reproducible (affects matrix initialization)")

FLAGS = flags.FLAGS

def log10(x):
  numerator = tf.log(x)
  denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
  return numerator / denominator

class RNNLMModel(object):
  """The RNNLM model. To use the model in decoding where we need probabilities, pass use_log_probs=True.
  """
  def __init__(self, is_training, config, use_log_probs=False):
    self.batch_size = batch_size = config.batch_size
    self.num_steps = num_steps = config.num_steps
    hidden_size = config.hidden_size
    vocab_size = config.vocab_size

    self._input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
    self._targets = tf.placeholder(tf.int32, [batch_size, num_steps])

    self.global_step = tf.Variable(0, trainable=False)

    # Slightly better results can be obtained with forget gate biases
    # initialized to 1 but the hyperparameters of the model would need to be
    # different than reported in the paper.
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias=0.0)
    if is_training and config.keep_prob < 1:
      lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
          lstm_cell, output_keep_prob=config.keep_prob)
    cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers)
    
    if is_training or use_log_probs:
      logging.info("Using LSTM cells of size={}".format(hidden_size))
      logging.info("Model with %d layer(s)" % config.num_layers)
      logging.info("Model with %i unrolled step(s)" % config.num_steps)

    self._initial_state = cell.zero_state(batch_size, tf.float32)

    with tf.device("/cpu:0"):
      embedding = tf.get_variable("embedding", [vocab_size, hidden_size])
      inputs = tf.nn.embedding_lookup(embedding, self._input_data)

    if is_training and config.keep_prob < 1:
      inputs = tf.nn.dropout(inputs, config.keep_prob)

    outputs = []
    state = self._initial_state
    # Simplified version of tensorflow.models.rnn.rnn.py's rnn().
    # This builds an unrolled LSTM for tutorial purposes only.
    # In general, use the rnn() or state_saving_rnn() from rnn.py.
    #with tf.variable_scope("RNN"):
    #  for time_step in range(num_steps):
    #    if time_step > 0: tf.get_variable_scope().reuse_variables()
    #    (cell_output, state) = cell(inputs[:, time_step, :], state)
    #    outputs.append(cell_output)
    inputs = [tf.squeeze(input_, [1])
              for input_ in tf.split(1, num_steps, inputs)]
    outputs, state = rnn.rnn(cell, inputs, initial_state=self._initial_state)
    self._final_state = state

    output = tf.reshape(tf.concat(1, outputs), [-1, hidden_size])
    softmax_w = tf.get_variable("softmax_w", [hidden_size, vocab_size])
    softmax_b = tf.get_variable("softmax_b", [vocab_size])
    logits = tf.matmul(output, softmax_w) + softmax_b
    
    if use_log_probs:
      logging.info("Softmax")
      probs = tf.nn.softmax(logits)
      self._log_probs = tf.log(probs)
    else:
      loss = tf.nn.seq2seq.sequence_loss_by_example(
        [logits],
        [tf.reshape(self._targets, [-1])],
        [tf.ones([batch_size * num_steps])])
      self._cost = cost = tf.reduce_sum(loss) / batch_size

    if is_training:
      self._lr = tf.Variable(0.0, trainable=False)
      tvars = tf.trainable_variables()
      grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                        config.max_grad_norm)
      if FLAGS.use_adadelta:
        lr = 1.0
        rho = 0.95
        epsilon = 1e-6
        logging.info("Use AdaDeltaOptimizer with lr={}".format(lr))
        optimizer = tf.train.AdadeltaOptimizer(lr, rho=rho, epsilon=epsilon)
      elif FLAGS.use_adagrad:
        logging.info("Use AdaGradOptimizer with lr={}".format(self.lr))
        optimizer = tf.train.AdagradOptimizer(self.lr)
      elif FLAGS.use_adam:
        # default values are same as in Keras library
        logging.info("Use AdamOptimizer with default values")
        optimizer = tf.train.AdamOptimizer()
      elif FLAGS.use_rmsprop:
        lr = 0.5
        logging.info ("Use RMSPropOptimizer with lr={}".format(lr))
        optimizer = tf.train.RMSPropOptimizer(self.lr)
      else:
        logging.info("Use GradientDescentOptimizer")
        optimizer = tf.train.GradientDescentOptimizer(self.lr)
      self._train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)
    
    variable_prefix = "model"
    self.saver = tf.train.Saver({ v.op.name: v for v in tf.all_variables() if v.op.name.startswith(variable_prefix) })

  def assign_lr(self, session, lr_value):
    session.run(tf.assign(self.lr, lr_value))

  @property
  def input_data(self):
    return self._input_data

  @property
  def targets(self):
    return self._targets

  @property
  def initial_state(self):
    return self._initial_state

  @property
  def cost(self):
    return self._cost

  @property
  def final_state(self):
    return self._final_state

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op
    
  @property
  def log_probs(self):
    return self._log_probs 

def load_model(session, model_config, train_dir, use_log_probs=False):
  # Create and load model for decoding
  # If model_config is a path, read config from that path, else treat as config name
  if os.path.exists(model_config):
    config = model_utils.read_config(model_config)
  else:
    config = model_utils.get_config(model_config)
  config.batch_size = 1
  config.num_steps = 1
  with tf.variable_scope("model", reuse=None):
    model = RNNLMModel(is_training=False, config=config, use_log_probs=use_log_probs)

  ckpt = tf.train.get_checkpoint_state(train_dir)
  if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
    logging.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    model.saver.restore(session, ckpt.model_checkpoint_path)
  else:
    logging.error("Could not find model in directory %s." % train_dir)
    sys.exit(1)
  return model, config

def main(_):
  if not FLAGS.data_dir and (not FLAGS.train_idx or not FLAGS.dev_idx or not FLAGS.test_idx):
    raise ValueError("Must set --data_dir to PTB data directory or specify data using --train_idx,--dev_idx,--test_idx")

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
      
    if FLAGS.score:
      use_log_probs = True
      logging.info("Run model in scoring mode")
      train_dir = "train.rnn.de"
      model, _ = load_model(session, "large50k", train_dir, use_log_probs)

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
      if FLAGS.model:
        config = model_utils.get_config(FLAGS.model)
        eval_config = model_utils.get_config(FLAGS.model)
      elif FLAGS.config_file:
        config = model_utils.read_config(FLAGS.config_file)
        eval_config = copy.copy(config)
      else:
        logging.error("Must specify either model name or config file.")
        sys.exit(1)
      eval_config.batch_size = 1
      eval_config.num_steps = 1

      if FLAGS.data_dir:
        raw_data = reader.ptb_raw_data(FLAGS.data_dir)
        train_data, valid_data, test_data, _ = raw_data
      else:
        train_data = reader.read_indexed_data(FLAGS.train_idx, FLAGS.max_train_data_size, config.vocab_size)
        valid_data = reader.read_indexed_data(FLAGS.dev_idx, vocab_size=config.vocab_size)
        test_data = reader.read_indexed_data(FLAGS.test_idx, vocab_size=config.vocab_size)

      if FLAGS.use_adagrad:
        config.learning_rate = 0.5
      if FLAGS.fixed_random_seed:
        tf.set_random_seed(1234)

      initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)
      with tf.variable_scope("model", reuse=None, initializer=initializer):
        model = RNNLMModel(is_training=True, config=config)
      with tf.variable_scope("model", reuse=True, initializer=initializer):
        mvalid = RNNLMModel(is_training=False, config=config)
        mtest = RNNLMModel(is_training=False, config=eval_config)

      ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
      if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
        logging.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
      else:
        logging.info("Created model with fresh parameters.")
        session.run(tf.initialize_all_variables())

      # Restore saved train variable
      start_idx = 0
      tmpfile = FLAGS.train_dir+"/tmp_idx.pkl"
      if model.global_step.eval() >= FLAGS.steps_per_checkpoint and \
        os.path.isfile(tmpfile):
          with open(tmpfile, "rb") as f:
            start_idx = pickle.load(f)     
            logging.info("Restore saved train variable from %s, resume from train idx=%i" % (tmpfile, start_idx))

      for i in range(config.max_max_epoch):
        if not (FLAGS.use_adadelta or FLAGS.use_adam):
          lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
          model.assign_lr(session, config.learning_rate * lr_decay)

        logging.info("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(model.lr)))

        train_perplexity = train_utils.run_epoch(session, model, train_data, model.train_op, FLAGS.train_dir, FLAGS.steps_per_checkpoint,
                                                 train=True, start_idx=start_idx, tmpfile=tmpfile, m_valid=mvalid, valid_data=valid_data)
        start_idx = 0
        logging.info("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))

        valid_perplexity = train_utils.run_epoch(session, mvalid, valid_data, tf.no_op(), FLAGS.train_dir, FLAGS.steps_per_checkpoint)
        logging.info("Epoch: %d Full Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

      logging.info("Training finished.")
      test_perplexity = train_utils.run_epoch(session, mtest, test_data, tf.no_op(), FLAGS.train_dir, FLAGS.steps_per_checkpoint)
      logging.info("Test Perplexity: %.3f" % test_perplexity)

      checkpoint_path = os.path.join(FLAGS.train_dir, "rnn.ckpt")
      logging.info("Save final model to path=%s" % checkpoint_path)
      model.saver.save(session, checkpoint_path, global_step=model.global_step)

    logging.info("End: {}".format(datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')))

if __name__ == "__main__":
  tf.app.run()

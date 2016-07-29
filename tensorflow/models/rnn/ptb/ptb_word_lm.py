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
import time, datetime

import numpy as np
import tensorflow as tf

from tensorflow.models.rnn.ptb import reader
from tensorflow.models.rnn import rnn

flags = tf.flags
#logging = tf.logging
import logging
import pickle

flags.DEFINE_string(
    "model", "small",
    "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("data_dir", "/tmp", "data_dir")
flags.DEFINE_string("train_dir", "/tmp", "Training directory.")
flags.DEFINE_integer("max_train_data_size", 0, "Limit on the size of training data (0: no limit).") 
flags.DEFINE_string("device", None, "Device to be used")
flags.DEFINE_boolean("use_adadelta", False, "Use AdaDelta instead of GradientDescent")
flags.DEFINE_boolean("use_adagrad", False, "Use AdaGrad instead of GradientDescent")
flags.DEFINE_boolean("use_adam", False, "Use Adam instead of GradientDescent")
flags.DEFINE_boolean("use_rmsprop", False, "Use RmsProp instead of GradientDescent")
flags.DEFINE_integer("steps_per_checkpoint", 200, "How many training steps to do per checkpoint.")
flags.DEFINE_boolean("score", False, "Run rnnlm on test sentence and report logprobs")

FLAGS = flags.FLAGS

def log10(x):
  numerator = tf.log(x)
  denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
  return numerator / denominator

class RNNLMModel(object):
  """The PTB model."""

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
        #lr = 0.5
        #logging.info("Use AdaGradOptimizer with lr={}".format(lr))
        #optimizer = tf.train.AdagradOptimizer(lr)
        
          #lr = 0.1
        #logging.info("Use AdaGradOptimizer with lr={}".format(lr))
        #optimizer = tf.train.AdagradOptimizer(lr)

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
    
    #self.saver = tf.train.Saver(tf.all_variables())
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

class SmallConfig(object):
  """Small config."""
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 20
  hidden_size = 200
  max_epoch = 4
  max_max_epoch = 13
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  vocab_size = 10000


class MediumConfig(object):
  """Medium config."""
  init_scale = 0.05
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 35
  hidden_size = 650
  max_epoch = 6
  max_max_epoch = 39
  keep_prob = 0.5
  lr_decay = 0.8
  batch_size = 20
  vocab_size = 10000


class LargeConfig(object):
  """Large config."""
  init_scale = 0.04
  learning_rate = 1.0
  max_grad_norm = 10
  num_layers = 2
  num_steps = 35
  hidden_size = 1500
  max_epoch = 14
  max_max_epoch = 55
  keep_prob = 0.35
  lr_decay = 1 / 1.15
  batch_size = 20
  vocab_size = 10000

class LargeConfig50k(object):
  """Large config."""
  init_scale = 0.04
  learning_rate = 1.0
  max_grad_norm = 10
  num_layers = 2
  num_steps = 35
  hidden_size = 1500
  max_epoch = 14
  max_max_epoch = 55
  keep_prob = 0.35
  lr_decay = 1 / 1.15
  batch_size = 80
  vocab_size = 50003

class TestConfig(object):
  """Tiny config, for testing."""
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 1
  num_layers = 1
  #num_steps = 2
  num_steps = 5
  hidden_size = 2
  max_epoch = 1
  max_max_epoch = 1
  keep_prob = 1.0
  lr_decay = 0.5
  #batch_size = 20
  batch_size = 1
  #vocab_size = 10000
  vocab_size = 50003

def run_epoch(session, m, data, eval_op, train=False, start_idx=0, tmpfile=None, 
              m_valid=None, valid_data=None):
  """Runs the model on the given data."""
  epoch_size = ((len(data) // m.batch_size) - 1) // m.num_steps
  logging.info("Data_size=%s batch_size=%s epoch_size=%s" % (len(data), m.batch_size, epoch_size))
  start_time = time.time()
  costs = 0.0
  iters = 0
  state = m.initial_state.eval()
  for step, (x, y) in enumerate(reader.ptb_iterator(data, m.batch_size,
                                                    m.num_steps, start_idx), start=1+start_idx):
    cost, state, _ = session.run([m.cost, m.final_state, eval_op],
                                 {m.input_data: x,
                                  m.targets: y,
                                  m.initial_state: state})
    costs += cost
    iters += m.num_steps
    if train and step % 100 == 0:                                                      
      logging.info("Global step = %i" % m.global_step.eval())

    #if train and step % (epoch_size // 10) == 10:
    #  logging.info("%.3f perplexity: %.3f speed: %.0f wps" %
    #        (step * 1.0 / epoch_size, np.exp(costs / iters),
    #         iters * m.batch_size / (time.time() - start_time)))

    if train and step % FLAGS.steps_per_checkpoint == 0:
      logging.info("Time: {}".format(datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')))      
      logging.info("%.3f perplexity: %.3f speed: %.0f wps" %
            (step * 1.0 / epoch_size, np.exp(costs / iters),
             iters * m.batch_size / (time.time() - start_time)))
      # Save train variable
      with open(tmpfile, "wb") as f:
        # Training idx = step - 1, so we want to resume from idx = step
        # If we had already restarted from start_idx, this gives the offset
        #resume_from = step + start_idx
        resume_from = step
        pickle.dump(resume_from, f, pickle.HIGHEST_PROTOCOL)  

      checkpoint_path = os.path.join(FLAGS.train_dir, "rnn.ckpt")
      #finished_idx = step + start_idx -1
      finished_idx = step -1
      logging.info("Save model to path=%s after training_idx=%s and global_step=%s" % (checkpoint_path, finished_idx, m.global_step.eval()))
      m.saver.save(session, checkpoint_path, global_step=m.global_step)
      
      # Get a random validation batch and evaluate
      data_len = len(valid_data)
      batch_len = data_len // m_valid.batch_size
      epoch_size = (batch_len - 1) // m_valid.num_steps
      from random import randint
      rand_idx = randint(0,epoch_size-1)
      (x_valid, y_valid) = reader.ptb_iterator(valid_data, m_valid.batch_size, m_valid.num_steps, rand_idx).next()
      cost_valid, _, _ = session.run([m_valid.cost, m_valid.final_state, tf.no_op()],
                                 {m_valid.input_data: x_valid,
                                  m_valid.targets: y_valid,
                                  m_valid.initial_state: m_valid.initial_state.eval()})
      valid_perplexity = np.exp(cost_valid / m_valid.num_steps)
      logging.info("Perplexity for random validation index=%i: %.3f" % (rand_idx, valid_perplexity))

  return np.exp(costs / iters)

def run_epoch_eval(session, m, data, eval_op, use_log_probs=False):
  """Runs the model on the given data."""
  #epoch_size = ((len(data) // m.batch_size) - 1) // m.num_steps
  #logging.info("Data_size=%s batch_size=%s epoch_size=%s" % (len(data), m.batch_size, epoch_size))
  costs = 0.0
  iters = 0
  logp = 0.0
  wordcn = 0
  state = m.initial_state.eval()
  # This feeds one word at a time when batch size and num_steps are both 1
  for step, (x, y) in enumerate(reader.ptb_iterator(data, m.batch_size,
                                                    m.num_steps), start=1):                                                      
    if use_log_probs:
      log_probs, state = session.run([m.log_probs, m.final_state],
                                 {m.input_data: x,
                                  m.initial_state: state})
      logp += (log_probs[0][y[0]])[0]
      wordcn += 1
      #print ("{} log_prob={} y={}".format(step, log_probs[0][y[0]], y[0]))
    else:
      cost, state, _ = session.run([m.cost, m.final_state, eval_op],
                                 {m.input_data: x,
                                  m.targets: y,
                                  m.initial_state: state})
      costs += cost
      #print ("{} cost={}".format(step, cost))
      iters += m.num_steps
  
  if use_log_probs:
    logging.info("Test log probability={}".format(logp))
    logging.info("Test PPL: %f", np.exp(-logp/wordcn))
    return logp
  else:
    logging.info("Test PPL: %f", np.exp(costs / iters))
    return np.exp(costs / iters)

def run_step_eval(session, m, input_word, prev_state):
  """Runs the model given the previous state and the data.
  Model must have been created with argument use_log_probs=True."""
  #logging.info("input word={}".format(input_word))
  x = np.zeros([1, 1], dtype=np.int32)
  x[0] = input_word
  log_probs, state = session.run([m.log_probs, m.final_state],
                                 {m.input_data: x,
                                  m.initial_state: prev_state})
  return log_probs[0], state
  
def score_sentence(session, model, sentence):
  state = model.initial_state.eval()
  logp = 0.0
  wordcn = 0
  for i in range(len(sentence)-1):
    posterior, state = run_step_eval(session, model, sentence[i], state)
    logp += posterior[sentence[i+1]]
    wordcn += 1
    #print ("{} log_prob={} y={}".format(i+1, posterior[sentence[i+1]], sentence[i+1]))
  logging.info("Test log probability={}".format(logp))
  logging.info("Test PPL: %f", np.exp(-logp/wordcn))
  return logp

def get_config(model_config):
  if model_config == "small":
    return SmallConfig()
  elif model_config == "medium":
    return MediumConfig()
  elif model_config == "large":
    return LargeConfig()
  elif model_config == "large50k":
    return LargeConfig50k()
  elif model_config == "test":
    return TestConfig()
  else:
    raise ValueError("Invalid model: %s", FLAGS.model)

def load_model(session, model_config, train_dir, use_log_probs=False):
  # Create and load model for decoding
  config = get_config(model_config)
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
  if not FLAGS.data_dir:
    raise ValueError("Must set --data_dir to PTB data directory")

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
      #use_log_probs = False
      logging.info("Run model in scoring mode")
      train_dir = "train.rnn.de"      
      train_dir = "train.rnn.de.fixed"
      model, _ = load_model(session, "large50k", train_dir, use_log_probs)
  
      #num_test_sents = 2169
      #test_data = reader.indexed_data_test(FLAGS.data_dir, num_test_sents)
      #test_sentences = [ test_data ]
  
      # add eos symbol to the beginning to score first word as well
      test_sentences = [[2, 19, 44, 393, 9, 1834, 4],
                        [2, 19, 393, 9, 1834, 44, 4],
                        [2, 1834, 44, 19, 393, 9, 4],
                        [2, 4, 1834, 9, 393, 44, 19],
                        [2, 393, 393, 393, 393, 393, 4]]
      test_sentences = [[2, 5, 3316, 7930, 7, 7312, 9864, 30, 8, 10453, 4, 2],
                        [2, 7, 5, 30, 8, 10453, 7930, 3316, 7312, 9864, 4, 2],
                        [2, 5, 8, 30, 7, 4, 9864, 3316, 7312, 7930, 10453, 2],
                        [2, 8, 10453, 9864, 30, 5, 3316, 7312, 7, 7930, 4]]
      for test_data in test_sentences:
        # using log probs or cross entropies gives the same perplexities
        if use_log_probs:
          # Run model as in training, with an iterator over inputs
          run_epoch_eval(session, model, test_data, tf.no_op(), use_log_probs=use_log_probs)
          # Run model step by step (yields the same result)
          #score_sentence(session, model, test_data)      
        else:
          run_epoch_eval(session, model, test_data, tf.no_op(), use_log_probs=use_log_probs)                
    else:
      logging.info("Run model in training mode")
      config = get_config(FLAGS.model)
      eval_config = get_config(FLAGS.model)
      eval_config.batch_size = 1
      eval_config.num_steps = 1

      #raw_data = reader.ptb_raw_data(FLAGS.data_dir)
      #train_data, valid_data, test_data, _ = raw_data
      indexed_data = reader.indexed_data(FLAGS.data_dir, FLAGS.max_train_data_size, config.vocab_size)
      train_data, valid_data, test_data = indexed_data

      if FLAGS.use_adagrad:
        config.learning_rate = 0.5

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
        #if not (FLAGS.use_adadelta or FLAGS.use_adagrad or FLAGS.use_adam):
        if not (FLAGS.use_adadelta or FLAGS.use_adam):
          lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
          model.assign_lr(session, config.learning_rate * lr_decay)

        logging.info("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(model.lr)))

        train_perplexity = run_epoch(session, model, train_data, model.train_op, train=True, 
                                   start_idx=start_idx, tmpfile=tmpfile, m_valid=mvalid, valid_data=valid_data)
        start_idx = 0
        logging.info("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))

        valid_perplexity = run_epoch(session, mvalid, valid_data, tf.no_op())
        logging.info("Epoch: %d Full Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

      logging.info("Training finished.")
      test_perplexity = run_epoch(session, mtest, test_data, tf.no_op())
      logging.info("Test Perplexity: %.3f" % test_perplexity)

    logging.info("End: {}".format(datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')))

if __name__ == "__main__":
  tf.app.run()

import logging
import tensorflow as tf

from tensorflow.models.rnn import rnn

def log10(x):
  numerator = tf.log(x)
  denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
  return numerator / denominator

class RNNLMModel(object):
  """The RNNLM model. To use the model in decoding where we need probabilities, pass use_log_probs=True.
  """
  def __init__(self, config, variable_prefix, is_training, use_log_probs=False, optimizer="sgd",
               rename_variable_prefix=None):
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
      if optimizer == "adadelta":
        self.lr = 1.0
        rho = 0.95
        epsilon = 1e-6
        logging.info("Use AdaDeltaOptimizer with lr={}".format(self.lr))
        optimizer = tf.train.AdadeltaOptimizer(self.lr, rho=rho, epsilon=epsilon)
      elif optimizer == "adagrad":
        self.lr = 0.5
        logging.info("Use AdaGradOptimizer with lr={}".format(self.lr))
        optimizer = tf.train.AdagradOptimizer(self.lr)
      elif optimizer == "adam":
        # Default values are same as in Keras library
        logging.info("Use AdamOptimizer with default values")
        optimizer = tf.train.AdamOptimizer()
      elif optimizer == "rmsprop":
        self.lr = 0.5
        logging.info ("Use RMSPropOptimizer with lr={}".format(self.lr))
        optimizer = tf.train.RMSPropOptimizer(self.lr)
      else:
        logging.info("Use GradientDescentOptimizer")
        optimizer = tf.train.GradientDescentOptimizer(self.lr)
      self._train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)

    self.saver = tf.train.Saver({ v.op.name: v for v in tf.all_variables() if v.op.name.startswith(variable_prefix) }, max_to_keep=2)

    if rename_variable_prefix:
      self.saver_prefix = tf.train.Saver({ v.op.name.replace(variable_prefix, rename_variable_prefix): \
                                           v for v in tf.all_variables() if v.op.name.startswith(variable_prefix) }, max_to_keep=2)

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

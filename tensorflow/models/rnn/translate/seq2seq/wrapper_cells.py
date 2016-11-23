"""Module for constructing RNN Cells."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.ops.rnn_cell import RNNCell

from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.math_ops import tanh, sigmoid
import tensorflow as tf

import logging

class BidirectionalRNNCell(RNNCell):
  """RNN cell composed of two cells which process the input in opposite directions."""

  def __init__(self, cells):
    """Store two RNN cells, one forward and one backward cell.

    Args:
      cells: list of two RNNCells.

    Raises:
      ValueError: if there are not exactly two cells (not allowed).
    """
    if not cells or len(cells) != 2:
      raise ValueError("Must specify exactly two cells for BidirectionalRNNCell.")
    self._cells = cells
        
  @property
  def state_size(self):
    return sum([cell.state_size for cell in self._cells])

  @property
  def output_size(self):
    return sum([cell.output_size for cell in self._cells])

  @property
  def fw_state_size(self):
    return self._cells[0].state_size

  @property    
  def fw_output_size(self):
    return self._cells[0].output_size

  def get_fw_cell(self):
    return self._cells[0]

  def get_bw_cell(self):
    return self._cells[1]

  def __call__(self, inputs, state, scope=None):
    """We don't need this function because the bidirectional_rnn will call the rnn 
    separately on the forward and backward states"""    
    raise NotImplementedError

class BOWCell(RNNCell):
  """Wrapper for BOW model which uses the word embeddings instead of an RNN cell."""

  def __init__(self, cell):
    """RNN cell wrapper composed of one cell used to initialize the decoder.

    Args:
      cells: exactly one RNNCell.

    Raises:
      ValueError: if there is not exactly cell (not allowed).
    """
    if not cell:
      raise ValueError("Must specify a single/multi rnn cell for BOWCell.")
    self._cell = cell
        
  @property
  def state_size(self):
    return self._cell.state_size

  @property
  def output_size(self):
    return self._cell.output_size

  def embed(self, func, embedding_classes, embedding_size, inputs, dtype=None, scope=None,
            keep_prob=1.0, initializer=None):
    embedder_cell = func(self._cell, embedding_classes, embedding_size, initializer=initializer)

    # Like rnn(..) in rnn.py, but we call only the Embedder, not the RNN cell
    outputs = []
    with vs.variable_scope(scope or "Embedder") as varscope:
      if varscope.caching_device is None:
        varscope.set_caching_device(lambda op: op.device)

      for time, input_ in enumerate(inputs):
        if time > 0: vs.get_variable_scope().reuse_variables()
        embedding = embedder_cell.__call__(input_, scope)
        if keep_prob < 1:
          embedding = tf.nn.dropout(embedding, keep_prob)

        # annotation = C~_t = tanh ( E(x_t) + b_c)
        b_c = tf.get_variable("annotation_b", [embedding_size])
        annotation = tanh(tf.nn.bias_add(embedding, b_c))

        # weighted annotation = i_t * C~_t
        # i = sigmoid ( E(x_t) + b_i)
        b_i = tf.get_variable("input_b", [embedding_size])
        i = sigmoid(tf.nn.bias_add(embedding, b_i))
        w_annotation = i * annotation
        outputs.append(w_annotation)

      # return empty state, will be initialized by decoder
      batch_size = array_ops.shape(inputs[0])[0]
      state = self._cell.zero_state(batch_size, dtype)
      return (outputs, state)

  def get_cell(self):
    return self._cell

  def __call__(self, inputs, state, scope=None):
      # call underlying cell
      logging.debug("CALL RNN CELL")
      return self._cell.__call__(inputs, state, scope)
  


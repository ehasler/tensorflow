'''TODO: this is a modified version of cam/gnmt/predictors/, should merge!
TensorFlow implementation:
This module contains the main predictor based on neural nets. The
specific network architecture is implemented as engine. This
is the bridge between neural network engines and the high level
decoder which requires a predictor implementation.
'''
import tensorflow as tf

from tensorflow.models.rnn.translate.decoding.core import Predictor
from tensorflow.models.rnn.translate.utils import data_utils
from tensorflow.python.platform import gfile
import logging
import numpy as np

class NMTPredictor(Predictor):
  '''Neural MT predictor'''
  def __init__(self, session, engine, encoder="reverse", use_sequence_length=False,
               use_src_mask=False):
      super(NMTPredictor, self).__init__()
      self.session = session
      self.engine = engine

      self.enc_out = {}
      self.decoder_input = [data_utils.GO_ID]
      self.dec_state = {}
      self.bucket_id = -1
      self.num_heads = 1
      self.word_count = 0

      self.encoder = encoder
      self.use_sequence_length = use_sequence_length
      self.use_src_mask = use_src_mask

      self.set_up_predictor()

  def set_up_predictor(self):
    self.training_graph = self.engine.create_training_graph() # Needed for loading variables
    self.encoding_graph = self.engine.create_encoding_graph()
    self.single_step_decoding_graph = self.engine.create_single_step_decoding_graph(
        self.encoding_graph.outputs)
    self.buckets = self.encoding_graph.buckets # the same for other graphs

    ckpt = tf.train.get_checkpoint_state(tf.app.flags.FLAGS.train_dir)
    if ckpt and gfile.Exists(ckpt.model_checkpoint_path):
      logging.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
      self.training_graph.saver.restore(self.session, ckpt.model_checkpoint_path)
    else:
      logging.fatal("Could not load NN model parameters from %s"
                           % ckpt.model_checkpoint_path)

  def initialize(self, src_sentence):
    self.reset()
    feasible_buckets = [b for b in xrange(len(self.buckets))
                        #if self.buckets[b][0] > len(src_sentence)]
                        if self.buckets[b][0] >= len(src_sentence)]
    if not feasible_buckets: # Sentence too long -> truncate
      max_len = self.buckets[-1][0] - 1
      logging.warn("Truncate sentence of length %d to %d" % (len(src_sentence), max_len))
      src_sentence = src_sentence[:max_len]
      self.bucket_id = len(self.buckets) - 1
    else:
      self.bucket_id = min(feasible_buckets)
    encoder_inputs, _, _, sequence_length, src_mask = self.training_graph.get_batch(
            {self.bucket_id: [(src_sentence, [])]}, self.bucket_id, self.encoder)
    logging.info("bucket={}".format(self.buckets[self.bucket_id]))

    if self.use_sequence_length:
      last_enc_state, self.enc_out = self.encoding_graph.encode(
        self.session, encoder_inputs, self.bucket_id, sequence_length)
    else:
      last_enc_state, self.enc_out = self.encoding_graph.encode(
        self.session, encoder_inputs, self.bucket_id)

    # Initialize decoder state with last encoder state
    self.dec_state["dec_state"] = last_enc_state
    for a in xrange(self.num_heads):
      self.dec_state["dec_attns_%d" % a] = np.zeros((1, self.enc_out['enc_v_0'].size), dtype=np.float32)

    if self.use_src_mask:
      self.dec_state["src_mask"] = src_mask

  def predict_next(self):
    #if (self.word_count >= self.buckets[self.bucket_id][1] or 
    #    self.decoder_input[0] == data_utils.EOS_ID): # Predict EOS
    #    return {data_utils.EOS_ID: 0}
    if self.decoder_input[0] == data_utils.EOS_ID: # Predict EOS
      return {data_utils.EOS_ID: 0}

    output, self.dec_state = self.single_step_decoding_graph.decode(self.session, self.enc_out, 
                                               self.dec_state, self.decoder_input, self.bucket_id,
                                               self.use_src_mask, self.word_count)
    return output[0]

  def get_unk_probability(self, posterior):
    return posterior[data_utils.UNK_ID] if len(posterior) > 1 else float("-inf")

  def consume(self, word):
    self.decoder_input = [word]
    self.word_count = self.word_count + 1
    
  def get_state(self):
    return (self.decoder_input, self.dec_state, self.word_count)
    
  def set_state(self, state):
    self.decoder_input, self.dec_state, self.word_count  = state

  def reset(self):
    self.enc_out = {}
    self.decoder_input = [data_utils.GO_ID]
    self.dec_state = {}
    self.word_count = 0


class NNLMPredictor(Predictor):
  '''Neural LM predictor'''
  def __init__(self, path, reset_at_eos):
    super(NNLMPredictor, self).__init__()
    self.reset_at_eos = reset_at_eos
    if gfile.Exists(path):
      logging.info("Reading NNLM model parameters from %s" % path)
      self.training_graph.saver.restore(None, path) # Todo session
    else:
      logging.fatal("Could not load NN model parameters from %s" % path)
    self.create_model()
    self.reset()
    
  def predict_next(self):
    pass
    
  def get_unk_probability(self, posterior):
    return posterior[data_utils.UNK_ID]
    
  def initialize(self, src_sentence):
    ''' NNLM decoder is not dependent on the source sentence. Just reset 
    internal state here if reset_at_eos is True '''
    if self.reset_at_eos:
      self.reset()
    
  def consume(self, word):
    self.input_word = word
    
  def get_state(self):
    return self.input_word, self.hidden_state
    
  def set_state(self, state):
    self.input_word, self.hidden_state = state

  def reset(self):
    self.input_word = data_utils.EOS_ID
    self.current_state  = self.initial_state

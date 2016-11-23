'''
General note for this module: Code is mainly adopted from tensorflow.models.rnn.seq2seq.
Especially the _tf_* methods are copied from there but then modified by deleting parts
which are not necessary when reducing to a "only encoding" graph (_tf_enc_*) or a single
step decoding graph (_tf_dec_*). Therefore, occasionally the code might contain some
unnecessary assignments. I left them there to maintain the comparability to their
counterparts in tensorflow.models.rnn.seq2seq.
'''

from tensorflow.models.rnn.translate.seq2seq.engine import Engine, TrainGraph, EncodingGraph,\
    SingleStepDecodingGraph
from tensorflow.models.rnn.translate.seq2seq.seq2seq_model import Seq2SeqModel
from tensorflow.models.rnn.translate.seq2seq.wrapper_cells import BidirectionalRNNCell, BOWCell

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import variable_scope

from tensorflow.python.ops.math_ops import tanh

import tensorflow as tf
from itertools import chain

import logging

class TFSeq2SeqEngine(Engine):
    '''
    This engine represents the original tensor flow implementation in
    tensorflow.models.rnn.seq2seq. Note that while we can use the
    tensorflow implementation for the training_graph, we cannot
    use it for the encoding and single_step_decoding graph because
    it needs changes in the most inner function attention_decoder.
    Therefore, this module contains a lot of duplicated code from
    tensorflow.models.rnn.seq2seq with only minor changes.
    '''
    
    def __init__(self, source_vocab_size, target_vocab_size, buckets, embedding_size, hidden_size,
                 num_layers, max_gradient_norm = None, batch_size = None, learning_rate = None,
                 learning_rate_decay_factor = None, use_lstm=False,
                 num_samples=512, forward_only=False, opt_algorithm="sgd", 
                 encoder="reverse", use_sequence_length=False, use_src_mask=False, 
                 maxout_layer=False, init_backward=False, no_pad_symbol=False,
                 variable_prefix=None, init_const=False, use_bow_mask=False,
                 initializer=None):
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.buckets = buckets
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.max_gradient_norm = max_gradient_norm
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learning_rate_decay_factor = learning_rate_decay_factor
        self.use_lstm = use_lstm
        self.num_samples = num_samples       
        self.forward_only = forward_only
        self.opt_algorithm = opt_algorithm
        self.encoder = encoder
        self.use_sequence_length = use_sequence_length
        self.use_src_mask = use_src_mask
        self.maxout_layer = maxout_layer
        self.init_backward = init_backward
        self.no_pad_symbol = no_pad_symbol
        self.variable_prefix = variable_prefix
        self.init_const = init_const
        self.use_bow_mask = use_bow_mask
        self.initializer = initializer
                        
    def update_buckets(self, buckets):
      self.buckets = buckets
      self.encoding_graph.update_buckets(buckets)
      self.decoding_graph.update_buckets(buckets, self.encoding_graph.outputs)
    
    def create_training_graph(self):
        '''
        Build the training graph using tensorflow.models.rnn.seq2seq.
        ''' 
        logging.info("Create training graph")       
        return TFSeq2SeqTrainingGraph(self.source_vocab_size, self.target_vocab_size, self.buckets,
                self.embedding_size, self.hidden_size, self.num_layers, self.max_gradient_norm, self.batch_size, self.learning_rate,
                self.learning_rate_decay_factor, self.use_lstm, self.num_samples, self.forward_only, self.opt_algorithm, self.encoder,
                self.use_sequence_length, self.use_src_mask, self.maxout_layer, self.init_backward, self.no_pad_symbol, self.variable_prefix,
                self.init_const, self.use_bow_mask, self.initializer)
    
    def create_encoding_graph(self):
      '''
      The returned TFGraph represents the encoding. The output tensors must
      be compatible with the input tensors for create_single_step_decoding_graph.
      '''
      logging.info("Create encoding graph")    
      self.encoding_graph = TFSeq2SeqEncodingGraph(self.source_vocab_size, self.buckets, self.embedding_size, self.hidden_size, 
               self.num_layers, self.batch_size, self.use_lstm, self.num_samples, self.encoder, self.use_sequence_length, self.init_backward,
               self.variable_prefix, self.initializer)
      return self.encoding_graph
    
    def create_single_step_decoding_graph(self, enc_out):
      logging.info("Create decoding graph")
      self.decoding_graph = TFSeq2SeqSingleStepDecodingGraph(enc_out, self.target_vocab_size, self.buckets, self.embedding_size, self.hidden_size,
               self.num_layers, self.batch_size, self.use_lstm, self.num_samples, self.encoder, self.use_src_mask, self.maxout_layer, self.init_backward,
               self.variable_prefix, self.init_const, self.use_bow_mask, self.initializer)
      return self.decoding_graph


class TFSeq2SeqTrainingGraph(TrainGraph):
    '''
    This is the adapter class for tensorflow.models.rnn.translate.seq2seq_model.Seq2SeqModel.
    Here we can use the original implementations directly. 
    '''
    
    def __init__(self, source_vocab_size, target_vocab_size, buckets, embedding_size, hidden_size,
                 num_layers, max_gradient_norm, batch_size, learning_rate,
                 learning_rate_decay_factor, use_lstm=False,
                 num_samples=512, forward_only=False, opt_algorithm="sgd", encoder="reverse",
                 use_sequence_length=False, use_src_mask=False, maxout_layer=False, init_backward=False, no_pad_symbol=False,
                 variable_prefix=None, init_const=False, use_bow_mask=False, initializer=None):
        super(TFSeq2SeqTrainingGraph, self).__init__(buckets, batch_size)
        self.seq2seq_model = Seq2SeqModel(source_vocab_size, target_vocab_size, buckets, embedding_size, hidden_size,
                 num_layers, max_gradient_norm, batch_size, learning_rate,
                 learning_rate_decay_factor, use_lstm,
                 num_samples, forward_only, opt_algorithm, encoder,
                 use_sequence_length, use_src_mask, maxout_layer, init_backward, no_pad_symbol, variable_prefix,
                 init_const=init_const, use_bow_mask=use_bow_mask, initializer=initializer)

        self.learning_rate = self.seq2seq_model.learning_rate
        self.global_step = self.seq2seq_model.global_step
        self.learning_rate_decay_op = self.seq2seq_model.learning_rate_decay_op
        self.saver = self.seq2seq_model.saver
    
    def train_step(self, session, encoder_inputs, decoder_inputs, target_weights,
           bucket_id, forward_only):
        _, loss, _ = self.seq2seq_model.step(session, encoder_inputs, decoder_inputs, target_weights,
           bucket_id, forward_only)
        return loss

    def decode_step(self, session, encoder_inputs, decoder_inputs, target_weights,
           bucket_id, forward_only):
        ''' This function is only used by the vanilla decoder - decoding is normally
        done using the single step decoding graphs '''
        _, _, output_logits = self.seq2seq_model.step(session, encoder_inputs, decoder_inputs, target_weights,
           bucket_id, forward_only)
        return output_logits
        
    def get_batch(self, data, bucket_id, encoder="reverse"):
      return self.seq2seq_model.get_batch(data, bucket_id, encoder)

class TFSeq2SeqEncodingGraph(EncodingGraph):
    '''
    This class extracts the encoding graph from the original tensorflow implementation. 
    Lots of duplicated code from seq2seq_model
    '''
    
    def __init__(self, source_vocab_size, buckets, embedding_size, hidden_size,
                 num_layers, batch_size, use_lstm=False, num_samples=512, 
                 encoder="reverse", use_sequence_length=False, init_backward=False,
                 variable_prefix=None, initializer=None):
        super(TFSeq2SeqEncodingGraph, self).__init__(buckets, batch_size)
        self.source_vocab_size = source_vocab_size
        self.num_heads = 1
    
        # Create the internal multi-layer cell for our RNN.
        if use_lstm:
          logging.info("Using LSTM cells of size={}".format(hidden_size))
          if initializer:
            single_cell = rnn_cell.LSTMCell(hidden_size, initializer=initializer)
          else:
            # NOTE: to use peephole connections, cell clipping or a projection layer, use LSTMCell instead
            single_cell = rnn_cell.BasicLSTMCell(hidden_size)
        else:
          logging.info("Using GRU cells of size={}".format(hidden_size))
          single_cell = rnn_cell.GRUCell(hidden_size)
        cell = single_cell

        if encoder == "bidirectional":
          logging.info("Bidirectional model")
          if init_backward:
            logging.info("Use backward encoder state to initialize decoder state")
          cell = BidirectionalRNNCell([single_cell] * 2)
        elif encoder == "bow":
          logging.info("BOW model")
          if num_layers > 1:
            logging.info("Model with %d layers for the decoder" % num_layers)
            cell = BOWCell(rnn_cell.MultiRNNCell([single_cell] * num_layers))
          else:
            cell = BOWCell(single_cell)
        elif num_layers > 1:
          logging.info("Model with %d layers" % num_layers)
          cell = rnn_cell.MultiRNNCell([single_cell] * num_layers)
    
        # The seq2seq function: we use embedding for the input and attention.
        scope = None
        if variable_prefix is not None:
          scope = variable_prefix+"/embedding_attention_seq2seq"
          logging.info("Using variable scope {}".format(scope))    
        def seq2seq_f(encoder_inputs, bucket_length):
          return self._tf_enc_embedding_attention_seq2seq(encoder_inputs, cell, source_vocab_size, embedding_size, 
                                                          encoder=encoder, 
                                                          sequence_length=self.sequence_length,
                                                          bucket_length=bucket_length,
                                                          init_backward=init_backward,
                                                          bow_emb_size=hidden_size,
                                                          scope=scope)
    
        # Feeds for inputs.
        self.encoder_inputs = []
        self.sequence_lengths = []
        for i in xrange(buckets[-1][0]):  # Last bucket is the biggest one.
          self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                    name="encoder{0}".format(i)))
        if use_sequence_length:
          logging.info("Using sequence length for encoder")                                            
          self.sequence_length = tf.placeholder(tf.int32, shape=[None], name="seq_len")          
        else:
          self.sequence_length = None
    
        self.outputs = self._tf_enc_model_with_buckets(self.encoder_inputs, buckets, seq2seq_f)
            
        # for update_buckets            
        self.seq2seq_f = seq2seq_f

    def update_buckets(self, buckets): 
      # Feeds for inputs.
      self.encoder_inputs = []
      for i in xrange(buckets[-1][0]):  # Last bucket is the biggest one.
        self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name="encoder{0}".format(i)))

      self.outputs = self._tf_enc_model_with_buckets(self.encoder_inputs, buckets, self.seq2seq_f)
    
    def encode(self, session, encoder_inputs, bucket_id, sequence_length=None):
        '''Similar to tensorflow.models.rnn.seq2seq.seq2seq_model.step
        Returns last enc state, enc_out'''
        # Check if the sizes match.
        encoder_size, _ = self.buckets[bucket_id]
        if len(encoder_inputs) != encoder_size:
            raise ValueError("Encoder length must be equal to the one in bucket,"
                           " %d != %d." % (len(encoder_inputs), encoder_size))
    
        # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
        input_feed = {}
        for l in xrange(encoder_size):
            input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
            
        if sequence_length is not None:
          logging.debug("Using sequence length for encoder: feed")
          input_feed[self.sequence_length.name] = sequence_length                    
    
        # run the model for given bucket_id
        outputs = session.run(self.outputs[bucket_id], input_feed)
        # Build dict from outputs which can be used as decoder input feed
        ret = {}
        ret["enc_hidden"] = outputs[1]
        for a in xrange(self.num_heads):
            ret["enc_hidden_features_%d" % a] = outputs[a + 2]
            ret["enc_v_%d" % a] = outputs[a + 2 + self.num_heads]
        return outputs[0], ret


    def _tf_enc_embedding_attention_seq2seq(self, encoder_inputs, cell,
                                    num_encoder_symbols,
                                    embedding_size,
                                    num_heads=1,
                                    dtype=dtypes.float32,
                                    scope=None,
                                    encoder="reverse",
                                    sequence_length=None,
                                    bucket_length=None,
                                    init_backward=False,
                                    bow_emb_size=None):
        """Embedding sequence-to-sequence model with attention.
        """
        with tf.variable_scope(scope or "embedding_attention_seq2seq", reuse=True):    
            # Encoder.
            if encoder == "bidirectional":
              encoder_cell_fw = rnn_cell.EmbeddingWrapper(
                cell.get_fw_cell(), embedding_classes=num_encoder_symbols,
                embedding_size=embedding_size)
              encoder_cell_bw = rnn_cell.EmbeddingWrapper(
                cell.get_bw_cell(), embedding_classes=num_encoder_symbols,
                embedding_size=embedding_size)        
              encoder_outputs, encoder_state, encoder_state_bw = rnn.bidirectional_rnn(encoder_cell_fw, encoder_cell_bw, 
                                 encoder_inputs, dtype=dtype, 
                                 sequence_length=sequence_length,
                                 bucket_length=bucket_length)
              logging.debug("Bidirectional state size=%d" % cell.state_size) # this shows double the size for lstms
            elif encoder == "reverse": 
              encoder_cell = rnn_cell.EmbeddingWrapper(
                cell, embedding_classes=num_encoder_symbols,
                embedding_size=embedding_size)
              encoder_outputs, encoder_state = rnn.rnn(
                encoder_cell, encoder_inputs, dtype=dtype, sequence_length=sequence_length, bucket_length=bucket_length, reverse=True)
              logging.debug("Unidirectional state size=%d" % cell.state_size)
            elif encoder == "bow":
              encoder_outputs, encoder_state = cell.embed(rnn_cell.Embedder, num_encoder_symbols,
                                                  bow_emb_size, encoder_inputs, dtype=dtype)               
        
            # First calculate a concatenation of encoder outputs to put attention on.
            if encoder == "bow":
              top_states = [array_ops.reshape(e, [-1, 1, bow_emb_size])
                  for e in encoder_outputs]
            else:
              top_states = [array_ops.reshape(e, [-1, 1, cell.output_size])
                          for e in encoder_outputs]
            attention_states = array_ops.concat(1, top_states)

            initial_state = encoder_state
            if encoder == "bidirectional" and init_backward:
              initial_state = encoder_state_bw

            return self._tf_enc_embedding_attention_decoder(
                attention_states, initial_state, cell, num_heads=num_heads)     
    
    def _tf_enc_embedding_attention_decoder(self, attention_states, last_enc_state,
                                    cell, num_heads=1, dtype=dtypes.float32,
                                    scope=None):
        with tf.variable_scope(scope or "embedding_attention_decoder"):
            with tf.device("/cpu:0"):
                return self._tf_enc_attention_decoder(attention_states, last_enc_state, cell,
                    num_heads=num_heads)
    
    
    def _tf_enc_attention_decoder(self, attention_states, last_enc_state, cell,
                          num_heads=1,
                          dtype=dtypes.float32, scope=None):
        """RNN decoder with attention for the sequence-to-sequence model.
    
        Args:
          return_encodings: If true, return encoder hidden states. Otherwise, return
            single step decoding tensors
        """
        if num_heads < 1:
            raise ValueError("With less than 1 heads, use a non-attention decoder.")
        if not attention_states.get_shape()[1:2].is_fully_defined():
            raise ValueError("Shape[1] and [2] of attention_states must be known: %s"
                         % attention_states.get_shape())
    
        with variable_scope.variable_scope(scope or "attention_decoder"):
          attn_length = attention_states.get_shape()[1].value
          attn_size = attention_states.get_shape()[2].value
      
          # To calculate W1 * h_t we use a 1-by-1 convolution, need to reshape before.
          hidden = array_ops.reshape(
              attention_states, [-1, attn_length, 1, attn_size])
          hidden_features = []
          v = []
          attention_vec_size = attn_size  # Size of query vectors for attention.
          for a in xrange(num_heads):
            k = variable_scope.get_variable("AttnW_%d" % a,
                                            [1, 1, attn_size, attention_vec_size])
            hidden_features.append(nn_ops.conv2d(hidden, k, [1, 1, 1, 1], "SAME")) # Hidden states multiplied with W1
            v.append(variable_scope.get_variable("AttnV_%d" % a,
                                                 [attention_vec_size]))    
        
          return [last_enc_state] + [hidden] + hidden_features + v
    
    def _tf_enc_model_with_buckets(self, encoder_inputs, buckets, seq2seq, name=None):
        """Like model_with_buckets but handles output of seq2seq as single variable.
        This generalizes for using it with encoder_graph and single_step_decoder_graph
        """
        if len(encoder_inputs) < buckets[-1][0]:
            raise ValueError("Length of encoder_inputs (%d) must be at least that of la"
                         "st bucket (%d)." % (len(encoder_inputs), buckets[-1][0]))
    
        all_inputs = encoder_inputs
        outputs = []
        with ops.op_scope(all_inputs, name, "model_with_buckets"):
          for j, bucket in enumerate(buckets):
            with variable_scope.variable_scope(variable_scope.get_variable_scope(),
                                               reuse=True if j > 0 else None):
              logging.debug("enc model for bucket={}".format(bucket))
              # this produces a list of 4 outputs (input: encoder input up to src bucket length)
              bucket_outputs = seq2seq(encoder_inputs[:bucket[0]], 
                                       bucket[0])     
              outputs.append(bucket_outputs)
              
        # a list of length=len(buckets), each a list of 4 outputs
        return outputs

class TFSeq2SeqSingleStepDecodingGraph(SingleStepDecodingGraph):
    
    def __init__(self, enc_out, target_vocab_size, buckets, embedding_size, hidden_size,
                 num_layers, batch_size, use_lstm=False, num_samples=512, 
                 encoder="reverse", use_src_mask=False, maxout_layer=False, init_backward=False,
                 variable_prefix=None, init_const=False, use_bow_mask=False, initializer=None):
        super(TFSeq2SeqSingleStepDecodingGraph, self).__init__(buckets, batch_size)
        self.target_vocab_size = target_vocab_size
        self.num_heads = 1
    
        # If we use sampled softmax, we need an output projection.
        output_projection = None
        softmax_loss_function = None
        # Sampled softmax only makes sense if we sample less than vocabulary size.
        if num_samples > 0 and num_samples < self.target_vocab_size:
          with variable_scope.variable_scope(variable_scope.get_variable_scope(),
                                               reuse=True), tf.device("/cpu:0"):
            w = tf.get_variable("proj_w", [hidden_size, self.target_vocab_size])
            w_t = tf.transpose(w)
            b = tf.get_variable("proj_b", [self.target_vocab_size])
          output_projection = (w, b)
            
          def sampled_loss(inputs, labels):
            with tf.device("/cpu:0"):
              labels = tf.reshape(labels, [-1, 1])
              return tf.nn.sampled_softmax_loss(w_t, b, inputs, labels, num_samples,
                                                self.target_vocab_size)
          softmax_loss_function = sampled_loss
        else:
          logging.info("Using maxout_layer=%d and full softmax loss" % maxout_layer)          
    
        # Create the internal multi-layer cell for our RNN.
        if use_lstm:
          logging.info("Using LSTM cells of size={}".format(hidden_size))
          if initializer:
            single_cell = rnn_cell.LSTMCell(hidden_size, initializer=initializer)
          else:
            # NOTE: to use peephole connections, cell clipping or a projection layer, use LSTMCell instead
            single_cell = rnn_cell.BasicLSTMCell(hidden_size)
        else:
          logging.info("Using GRU cells of size={}".format(hidden_size))
          single_cell = rnn_cell.GRUCell(hidden_size)
        cell = single_cell

        if encoder == "bidirectional":
          logging.info("Bidirectional model")
          if init_backward:
            logging.info("Use backward encoder state to initialize decoder state")
          cell = BidirectionalRNNCell([single_cell] * 2)
        elif encoder == "bow":
          logging.info("BOW model")
          if num_layers > 1:
            logging.info("Model with %d layers for the decoder" % num_layers)
            cell = BOWCell(rnn_cell.MultiRNNCell([single_cell] * num_layers))
          else:
            cell = BOWCell(single_cell)
        elif num_layers > 1:
          logging.info("Model with %d layers" % num_layers)
          cell = rnn_cell.MultiRNNCell([single_cell] * num_layers)
    
        # List of placeholders deeper within the decoder (i.e. bucket dependent)
        self.enc_hidden = []
        self.enc_hidden_features = []
        self.enc_v = []
        self.dec_attns = []

        # Placeholder for last state
        if encoder == "bidirectional":
          self.dec_state = tf.placeholder(dtypes.float32, shape=[None, cell.fw_state_size],
                                          name="dec_state")
        elif encoder == "reverse" or encoder == "bow":
          self.dec_state = tf.placeholder(dtypes.float32, shape=[None, cell.state_size],
                                         name="dec_state")                                                                        

        if use_src_mask:
          logging.info("Using source mask for decoder") 
          self.src_mask = tf.placeholder(dtypes.float32, shape=[None, None],
                                         name="src_mask")
        else:
          self.src_mask = None

        if use_bow_mask:
          logging.info("Using bow mask for output layer") 
          self.bow_mask = tf.placeholder(dtypes.float32, shape=[None, None],
                                         name="bow_mask")
        else:
          self.bow_mask = None          

        # placeholder to indicate whether we're at the start of the target sentence
        self.start = tf.placeholder(tf.bool, name="start")

        # The seq2seq function: we use embedding for the input and attention.
        scope = None
        if variable_prefix is not None:
          scope = variable_prefix+"/embedding_attention_seq2seq"
          logging.info("Using variable scope {}".format(scope))
        def seq2seq_f(bucket_enc_out, decoder_input):
            return self._tf_dec_embedding_attention_seq2seq(bucket_enc_out,
                decoder_input, self.dec_state, cell, target_vocab_size, embedding_size, 
                output_projection=output_projection, encoder=encoder, 
                src_mask=self.src_mask, maxout_layer=maxout_layer, init_backward=init_backward,
                start=self.start, scope=scope, init_const=init_const, bow_mask=self.bow_mask)
    
        self.dec_decoder_input = tf.placeholder(tf.int32, shape=[None],
                                                    name="dec_decoder_input")
        self.outputs = self._tf_dec_model_with_buckets(enc_out,
            self.dec_decoder_input, buckets, seq2seq_f)
        # If we use output projection, we need to project outputs for decoding.
        if output_projection is not None:
            # self.outputs contains outputs, new_attns, new_state in flattened list
            for b in xrange(len(buckets)): 
                output = self.outputs[b][0]
                ''' Standard implementation uses following code here to get the previous output (_extract_argmax_and_embed):
                output = tf.nn.xw_plus_b(output, output_projection[0],
                                                 output_projection[1])
                                                 
                However, we have to normalize during decoding using a softmax
                (and then taking a log to produce logprobs),
                as described in nn.py, def sampled_softmax_loss:
                "This operation is for training only.  It is generally an underestimate of the full softmax loss. 
                At inference time, you can compute full softmax probabilities with the expression 
                `tf.nn.softmax(tf.matmul(inputs, weights) + biases)`."
                Note: tf.matmul(i, w) + b does the same as tf.nn.xw_plus_b(i, w, b)
                '''
                output = tf.log(tf.nn.softmax(tf.nn.xw_plus_b(output, output_projection[0],
                                                 output_projection[1])))
                self.outputs[b][0] = output
        else:
          logging.info("Apply full softmax")
          for b in xrange(len(buckets)):
            self.outputs[b][0] = tf.log(tf.nn.softmax(self.outputs[b][0]))
                
        # for update_buckets
        self.enc_out = enc_out
        self.seq2seq_f = seq2seq_f
        self.output_projection = output_projection
        
    def update_buckets(self, buckets, enc_out):
      new_output = self._tf_dec_model_with_buckets(enc_out, self.dec_decoder_input, buckets, self.seq2seq_f, update=True)[0]      
      self.outputs.append(new_output)
      
      if self.output_projection is not None:
        # NOTE: run only for the new bucket
        output = self.outputs[-1][0]
        output = tf.log(tf.nn.softmax(tf.nn.xw_plus_b(output, self.output_projection[0],
                                        self.output_projection[1])))
        self.outputs[-1][0] = output
      else:
        self.outputs[-1][0] = tf.log(tf.nn.softmax(self.outputs[-1][0]))
        
    def decode(self, session, enc_out, dec_state, decoder_input, bucket_id, use_src_mask=False, word_count=0, use_bow_mask=False):
        '''Like seq2seq_model.step()
        Returns output, dec_state where dec_state stores the state of the
        decoder cell and the attention network.
        '''
        input_feed = {}
        input_feed[self.enc_hidden[bucket_id]] = enc_out["enc_hidden"]
        input_feed[self.dec_decoder_input] = decoder_input
        input_feed[self.dec_state] = dec_state["dec_state"]
        for a in xrange(self.num_heads):
            input_feed[self.enc_hidden_features[bucket_id][a]] = enc_out["enc_hidden_features_%d" % a]
            input_feed[self.enc_v[bucket_id][a]] = enc_out["enc_v_%d" % a]
            input_feed[self.dec_attns[bucket_id][a]] = dec_state["dec_attns_%d" % a]
            
        if use_src_mask:
          logging.debug("Using source mask for decoder: feed") 
          input_feed[self.src_mask.name] = dec_state["src_mask"]
        
        if word_count > 0:
          input_feed[self.start] = False
        else:
          input_feed[self.start] = True
        logging.debug("Word count = {} start = {}".format(word_count, input_feed[self.start]))

        if use_bow_mask:
          logging.debug("Using bow mask for output layer: feed") 
          input_feed[self.bow_mask.name] = dec_state["bow_mask"]
            
        # run model for given bucket_id, returns [output] + [new_state] + new_attns
        outputs = session.run(self.outputs[bucket_id], input_feed)
        ret = {}
        ret["dec_state"] = outputs[1]
        for a in xrange(self.num_heads):
            ret["dec_attns_%d" % a] = outputs[a + 2]
        if use_src_mask:
          # pass src mask on
          ret["src_mask"] = dec_state["src_mask"]
        if use_bow_mask:
          # pass bow mask on
          ret["bow_mask"] = dec_state["bow_mask"]          
        return outputs[0], ret

    def _tf_dec_embedding_attention_seq2seq(self, enc_out, decoder_input, last_state, cell,
                                num_decoder_symbols, embedding_size,
                                num_heads=1, output_projection=None,
                                dtype=dtypes.float32,
                                scope=None,
                                encoder="reverse",
                                src_mask=None,
                                maxout_layer=False,
                                init_backward=False,
                                start=None,
                                init_const=False, 
                                bow_mask=None):
        """Decode single step version of tensorflow.models.rnn.seq2seq.embedding_attention_seq2seq
        """
        with tf.variable_scope(scope or "embedding_attention_seq2seq", reuse=True): 
          # Decoder.
          if encoder == "bidirectional":
            if init_backward:
              cell = cell.get_bw_cell()
            else:
              cell = cell.get_fw_cell()
        
          output_size = None
          if output_projection is None:
            #cell = rnn_cell.OutputProjectionWrapper(cell, num_decoder_symbols)
            output_size = num_decoder_symbols
          return self._tf_dec_embedding_attention_decoder(
            enc_out, decoder_input, last_state, cell,
            num_decoder_symbols, embedding_size, num_heads, output_size, output_projection, src_mask=src_mask, maxout_layer=maxout_layer,
            encoder=encoder, start=start, init_const=init_const, bow_mask=bow_mask)
        
    def _tf_dec_embedding_attention_decoder(self, enc_out, decoder_input, last_state,
                                    cell, num_symbols, embedding_size, num_heads=1,
                                    output_size=None, output_projection=None,
                                    dtype=dtypes.float32,
                                    scope=None, src_mask=None, maxout_layer=False, encoder="reverse",
                                    start=None, init_const=False, bow_mask=None):
        """Decode single step version of tensorflow.models.rnn.seq2seq.embedding_attention_decoder
            """
        if output_size is None:
          output_size = cell.output_size
        if output_projection is not None:
          proj_weights = ops.convert_to_tensor(output_projection[0], dtype=dtype)
          proj_weights.get_shape().assert_is_compatible_with([cell.output_size,
                                                                num_symbols])   
          proj_biases = ops.convert_to_tensor(output_projection[1], dtype=dtype)
          proj_biases.get_shape().assert_is_compatible_with([num_symbols])

        with variable_scope.variable_scope(scope or "embedding_attention_decoder"):
          with ops.device("/cpu:0"):
            embedding = variable_scope.get_variable("embedding",
                                                    [num_symbols, embedding_size])
          emb_inp = embedding_ops.embedding_lookup(embedding, decoder_input)
          return self._tf_dec_attention_decoder(
              enc_out, emb_inp, last_state, cell, output_size=output_size,
              num_heads=num_heads, src_mask=src_mask, maxout_layer=maxout_layer, embedding_size=embedding_size,
              encoder=encoder, start=start, init_const=init_const, bow_mask=bow_mask)

    def _tf_dec_attention_decoder(self, enc_out, decoder_input, last_state, cell,
                          output_size=None, num_heads=1, dtype=dtypes.float32, scope=None,
                          src_mask=None, maxout_layer=False, embedding_size=None, 
                          encoder="reverse", start=None, init_const=False, bow_mask=None):
        """Decode single step version of tensorflow.models.rnn.seq2seq.attention_decoder
        """
        if num_heads < 1:
            raise ValueError("With less than 1 heads, use a non-attention decoder.")
        if output_size is None:
            output_size = cell.output_size
            
        with tf.variable_scope(scope or "attention_decoder"):
            # enc_out is a list [last_enc_state, hidden, num_heads*hidden_features, num_heads*v]
            # Note that these are computation graphs. We only use them to get the
            # shape right. During computation time, we use placeholders instead
            # because we want to specify them via input feed   
            hidden_shape = enc_out[1].get_shape()
            hidden_feature_shape = enc_out[2].get_shape()
            v_shape = enc_out[num_heads+2].get_shape()

            hidden = tf.placeholder(dtypes.float32,
                shape=[d.value for d in hidden_shape], 
                name="enc_hidden")            
            hidden_features = [tf.placeholder(dtypes.float32,
                shape=[d.value for d in hidden_feature_shape],
                name="enc_hidden_features_%d" % a) for a in xrange(num_heads)]
            v = [tf.placeholder(dtypes.float32,
                shape=[d.value for d in v_shape],
                name="enc_v_%d" % a) for a in xrange(num_heads)]
            self.enc_hidden.append(hidden)
            self.enc_hidden_features.append(hidden_features)
            self.enc_v.append(v)

            batch_size = 1
            attn_length = hidden.get_shape()[1].value
            attn_size = hidden.get_shape()[3].value
            attention_vec_size = attn_size  # Size of query vectors for attention.
            logging.info("Attn_length=%d attn_size=%d" % (attn_length, attn_size))

            def is_LSTM_cell(cell):
              if isinstance(cell, rnn_cell.LSTMCell) or \
                 isinstance(cell, rnn_cell.BasicLSTMCell):
                   return True
              return False

            def is_LSTM_cell_with_dropout(cell):
              if isinstance(cell, rnn_cell.DropoutWrapper):
                if is_LSTM_cell(cell._cell):
                   return True
              return False

            def init_state():
              logging.info("Init decoder state for bow")
              for a in xrange(num_heads):
                s = array_ops.ones(array_ops.pack([batch_size, attn_length]), dtype=dtype)
                s.set_shape([None, attn_length])
              
              # multiply with source mask, then do softmax
              if src_mask is not None:
                s = s * src_mask
              a = nn_ops.softmax(s)
              
              if isinstance(cell, BOWCell) and \
                 (is_LSTM_cell(cell.get_cell()) or \
                  is_LSTM_cell_with_dropout(cell.get_cell()) or \
                  (isinstance(cell.get_cell(), rnn_cell.MultiRNNCell) and \
                  (is_LSTM_cell(cell.get_cell()._cells[0]) or \
                   is_LSTM_cell_with_dropout(cell.get_cell()._cells[0])))):
                  # C = SUM_t i_t * C~_t (ignore i_t for now)
                  C = math_ops.reduce_sum(
                    array_ops.reshape(a, [-1, attn_length, 1, 1]) * hidden, [1, 2])                         
                  h = tanh(C)

                  if is_LSTM_cell(cell.get_cell()) or \
                    is_LSTM_cell_with_dropout(cell.get_cell()):
                    # single LSTM cell
                    return array_ops.concat(1, [C, h])
                  else:
                    # MultiRNNCell (multi LSTM cell)
                    unit = array_ops.concat(1, [C, h])
                    state = unit
                    count = 1
                    while (count < cell.get_cell().num_layers):
                      state = array_ops.concat(1, [state, unit])
                      count += 1
                    return state
              else:
                raise NotImplementedError("Need to implement decoder state initialization for non-LSTM cells")              

            def init_state_const():
              # TODO: don't hardcode (training) batch size
              b_size = 80              
              state_batch = variable_scope.get_variable("DecInit", [b_size, cell.state_size])
              state = math_ops.reduce_sum(state_batch, [0])
              state = array_ops.reshape(state, [1, cell.state_size])
              logging.info("Init decoder state: {} * {} matrix".format(1, cell.state_size))
              state = init_state()
              return state

            def keep_state():
              logging.info("Keep decoder state for bow")
              return last_state

            if encoder == "bow" and start is not None:
              if init_const:                
                last_state = control_flow_ops.cond(start, init_state_const, keep_state)
                last_state.set_shape([None, cell.state_size])
              else:
                last_state = control_flow_ops.cond(start, init_state, keep_state)
        
            def attention(query):
              """Put attention masks on hidden using hidden_features and query."""
              ds = []  # Results of attention reads will be stored here.
              for i in xrange(num_heads):
                with variable_scope.variable_scope("Attention_%d" % i):                  
                  y = rnn_cell.linear(query, attention_vec_size, True)
                  y = array_ops.reshape(y, [-1, 1, 1, attention_vec_size])
                  # Attention mask is a softmax of v^T * tanh(...).
                  s = math_ops.reduce_sum(
                      v[i] * math_ops.tanh(hidden_features[i] + y), [2, 3])
                  # multiply with source mask, then do softmax
                  if src_mask is not None:
                    s = s * src_mask
                  a = nn_ops.softmax(s)
                  # Now calculate the attention-weighted vector d.
                  d = math_ops.reduce_sum(
                      array_ops.reshape(a, [-1, attn_length, 1, 1]) * hidden,
                      [1, 2])                  
                  ds.append(array_ops.reshape(d, [-1, attn_size]))
              return ds            

            attns = [tf.placeholder(dtypes.float32,
                shape=[1, attn_size],
                name="dec_attns_%d" % i) for i in xrange(num_heads)]
            for a in attns:  # Ensure the second shape of attention vectors is set.
                a.set_shape([None, attn_size])

            self.dec_attns.append(attns)
            
            variable_scope.get_variable_scope().reuse_variables()

            # Merge input and previous attentions into one vector of the right size.
            input_size = decoder_input.get_shape().with_rank(2)[1]
            if input_size.value is None:
              raise ValueError("Could not infer input size from input: %s" % decoder_input.name)

            x = rnn_cell.linear([decoder_input] + attns, input_size, True)            
            # Run the RNN.
            cell_output, new_state = cell(x, last_state) # run cell on combination of input and previous attn masks
            # Run the attention mechanism.
            new_attns = attention(new_state) # calculate new attention masks (attention-weighted src vector)

            if maxout_layer:
              # This tries to imitate the blocks Readout layer, consisting of Merge, Bias, Maxout, Linear, Linear
              logging.info("Output layer consists of: Merge, Bias, Maxout, Linear, Linear")
              # Merge
              with tf.variable_scope("AttnMergeProjection"):
                merge_output = rnn_cell.linear([cell_output] + [decoder_input] + new_attns, cell.output_size, True)

              # Bias
              b = tf.get_variable("maxout_b", [cell.output_size])
              merge_output_plus_b = tf.nn.bias_add(merge_output, b)

              # Maxout
              maxout_size = cell.output_size // 2
              segment_id_list = [ [i,i] for i in xrange(maxout_size) ] # make pairs of segment ids to be max-ed over
              segment_id_list = list(chain(*segment_id_list)) # flatten list
              segment_ids = tf.constant(segment_id_list, dtype=tf.int32)
              maxout_output = tf.transpose(tf.segment_max(tf.transpose(merge_output_plus_b), segment_ids)) # transpose to get shape (cell.output_size, batch_size) and reverse         
              maxout_output.set_shape([None, maxout_size])

              # Linear, softmax0 (maxout_size --> embedding_size ), without bias
              with tf.variable_scope("MaxoutOutputProjection_0"):
                output_embed = rnn_cell.linear([maxout_output], embedding_size, False)

              # Linear, softmax1 (embedding_size --> vocab_size), with bias
              with tf.variable_scope("MaxoutOutputProjection_1"):
                output = rnn_cell.linear([output_embed], output_size, True)
            else:
              with variable_scope.variable_scope("AttnOutputProjection"):
                output = rnn_cell.linear([cell_output] + new_attns, output_size, True) # calculate the output

            if bow_mask is not None:
              # Normalize output layer over subset of target words found in input bag-of-words.
              # To do this without changing the architecture, apply a mask over the output layer
              # that sets all logits for words outside the bag to zero.
              logging.info("Use bow mask to locally normalize output layer wrt bow vocabulary")
              output = output * bow_mask

            return [output] + [new_state] + new_attns

    def _tf_dec_model_with_buckets(self, enc_out, decoder_input, buckets, seq2seq,
                       name=None, update=False):
        all_inputs = [decoder_input]
        outputs = []
        ''' Note: it is important that seq2seq is called in order of the buckets, otherwise
        the global placeholders enc_hidden, enc_hidden_features, enc_v, dec_attns
        are in wrong order (see append() statements in _tf_dec_attention_decoder)'''
        with ops.op_scope(all_inputs, name, "model_with_buckets"):
          for j, bucket in enumerate(buckets):
            if update and j < len(buckets)-1:
              # NOTE: run only for the new bucket
              continue
            with variable_scope.variable_scope(variable_scope.get_variable_scope(),
                                               reuse=True if j > 0 else None):
              logging.debug("dec model for bucket={}".format(bucket))
              bucket_decoder_input = decoder_input
              bucket_enc_out = enc_out[j]
              bucket_outputs = seq2seq(bucket_enc_out, bucket_decoder_input)
              outputs.append(bucket_outputs)
        return outputs






'''
An engine defines the specific neural network architecture for the Tensorflow 
implementation
'''

from abc import abstractmethod
import tensorflow as tf

import random
import numpy as np
from tensorflow.models.rnn.translate.utils import data_utils

class TFGraph(object):
    '''
    Represents a tensorflow execution graph represented by its output and input
    feeds (tensors)
    '''
    def __init__(self, buckets, batch_size):
        self.buckets = buckets
        self.batch_size = batch_size       


class TrainGraph(TFGraph):
    '''
    TrainGraph is returned by Engine.create_training_graph
    '''
    
    def __init__(self, buckets, batch_size, learning_rate = None, learning_rate_decay_factor = None):
        '''Initializes the tf variables learning_rate and global_step'''
        super(TrainGraph, self).__init__(buckets, batch_size)
        if learning_rate:
            self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
            self.global_step = tf.Variable(0, trainable=False)
            self.learning_rate_decay_op = self.learning_rate.assign(
                                    self.learning_rate * learning_rate_decay_factor)
            self.saver = None # Must be overridden by subclasses
        
    @abstractmethod
    def train_step(self, session, encoder_inputs, decoder_inputs, target_weights,
           bucket_id, forward_only):
        '''
        Performs a training step for a single batch.
        Note: During training, we evaluate on the dev_set with forward_only=True.
        This is the only time when we use the train graph for decoding, later we
        use a Decoder instance with EncodingGraph and SingleStepDecodingGraph
        
        return: step_loss
        '''
        raise NotImplementedError()

class EncodingGraph(TFGraph):
    '''
    EncodingGraph is returned by Engine.create_encoding_graph
    '''
    
    def __init__(self, buckets, batch_size):
        super(EncodingGraph, self).__init__(buckets, batch_size)
        
    @abstractmethod
    def encode(self, session, encoder_inputs, bucket_id):
        '''
        Returns hidden states of the encoder
        '''
        raise NotImplementedError()

class SingleStepDecodingGraph(TFGraph):
    '''
    SingleDecodingGraph is returned by Engine.create_single_step_decoding_graph
    '''
    
    def __init__(self, buckets, batch_size):
        super(SingleStepDecodingGraph, self).__init__(buckets, batch_size)
        
    @abstractmethod
    def decode(self):
        '''
        Performs a single step in the decoder network to generate the new state
        and output
        '''
        raise NotImplementedError()

class Engine(object):
    '''
    An engine defines the principle architecture of the translation system. This
    includes training graphs, encoding graphs, and single step decoding graphs.
    This class decouples the underlying network architectures and designs from
    the beam search / A* decoding algorithm
    '''
    
    @abstractmethod
    def create_training_graph(self):
        '''
        This should return a TFGraph instance which represents the training
        process - i.e. it contains both encoder and decoder. Source sentences
        are given by encoder_inputs and target sentences by decoder_inputs
        Compare tensorflow.models.rnn.seq2seq
        '''
        raise NotImplementedError()
    
    @abstractmethod
    def create_encoding_graph(self):
        '''
        The returned TFGraph represents the encoding. The output tensors must
        be compatible with the input tensors for create_single_step_decoding_graph.
        '''
        raise NotImplementedError()
    
    @abstractmethod
    def create_single_step_decoding_graph(self, enc_out):
        '''
        The returned TFGraph should represent a single step in the decoder network.
        It requires the tensors from create_encoding_graph and the current state
        as input, and outputs the posteriors of the next word
        '''
        raise NotImplementedError()

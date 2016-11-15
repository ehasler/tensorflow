# NOTE: This file is deprecated, it has been replaced by train.py and decode.py!!
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

"""Binary for training translation models and decoding from them.

Running this program without --decode will download the WMT corpus into
the directory specified as --data_dir and tokenize it in a very basic way,
and then start training a model saving checkpoints to --train_dir.

Running with --decode starts an interactive loop so you can see how
the current checkpoint translates English sentences into French.

See the following papers for more information on neural translation models.
 * http://arxiv.org/abs/1409.3215
 * http://arxiv.org/abs/1409.0473
 * http://arxiv.org/abs/1412.2007
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time, datetime
import logging, pickle, re

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.models.rnn.translate.utils import data_utils
from tensorflow.models.rnn.translate.seq2seq import seq2seq_model, tf_seq2seq
from tensorflow.models.rnn.translate.decoding.decoder import VanillaDecoder, GreedyDecoder, BeamDecoder
from tensorflow.models.rnn.translate.predictors.tf_neural import NMTPredictor

# todo: factor out decoding
from tensorflow.models.rnn.translate.decoding import core

import subprocess
from collections import defaultdict

# Training settings
tf.app.flags.DEFINE_string("src_lang", "en", "Source language")
tf.app.flags.DEFINE_string("trg_lang", "de", "Target language")
tf.app.flags.DEFINE_integer("batch_size", 80, "Batch size to use during training.")
tf.app.flags.DEFINE_string("data_dir", "/tmp", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "/tmp", "Training directory.")
tf.app.flags.DEFINE_boolean("default_filenames", False, "Whether to use the default filenames under the data directory (train.ids.LANG, dev.ids.LANG")
tf.app.flags.DEFINE_integer("max_train_data_size", 0, "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("max_train_batches", 0, "Limit on the number of training batches.")
tf.app.flags.DEFINE_integer("max_train_epochs", 0, "Limit on the number of training epochs.")
tf.app.flags.DEFINE_integer("max_epoch", 0, "Number of training epochs with original learning rate.")
tf.app.flags.DEFINE_boolean("adjust_lr", False, "Adjust learning rate independent of performance.")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 200, "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_string("device", None, "Device to be used")
tf.app.flags.DEFINE_boolean("train_sequential", False, "Shuffle training indices every epoch, then go through set sequentially")
tf.app.flags.DEFINE_integer("num_symm_buckets", 5, "Use x buckets of equal source/target length, with the largest bucket of length=50 (max_seq_len=50")
tf.app.flags.DEFINE_boolean("use_default_data", False, "If set, download and prepare Gigaword data set instead of providing custom train/dev/test data. That data will be tokenized first.")
tf.app.flags.DEFINE_boolean("add_src_eos", False, "Add EOS symbol to all source sentences.")
tf.app.flags.DEFINE_boolean("swap_pad_unk", False, "Swap UNK and PAD indices, such that UNK: 0 PAD: 3")
tf.app.flags.DEFINE_boolean("no_pad_symbol", False, "Only use GO, EOS, UNK, set PAD=-1")
tf.app.flags.DEFINE_boolean("init_backward", False, "When using the bidirectional encoder, initialise the hidden decoder state from the backward encoder state (default: forward).")
tf.app.flags.DEFINE_boolean("bow_init_const", False, "Learn an initialisation matrix for the decoder instead of taking the average of source embeddings")
tf.app.flags.DEFINE_boolean("use_bow_mask", False, "Normalize decoder output layer over per-sentence BOW vocabulary")
tf.app.flags.DEFINE_string("variable_prefix", None, "Suffix to add to graph variable names")
tf.app.flags.DEFINE_boolean("fixed_random_seed", False, "If True, use a fixed random seed to make training reproducible (affects matrix initialization)")
tf.app.flags.DEFINE_boolean("shuffle_data", True, "If False, do not shuffle the training data to make training reproducible")
tf.app.flags.DEFINE_integer("max_to_keep", 5, "Number of saved models to keep (set to 0 to keep all models)")

# Optimization settings
tf.app.flags.DEFINE_string("opt_algorithm", "sgd", "Optimization algorithm: sgd, adagrad, adadelta")
tf.app.flags.DEFINE_float("learning_rate", 1.0, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99, "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm.")

# Model configuration
tf.app.flags.DEFINE_integer("src_vocab_size", 40000, "Source vocabulary size.")
tf.app.flags.DEFINE_integer("trg_vocab_size", 40000, "Target vocabulary size.")
tf.app.flags.DEFINE_boolean("use_lstm", False, "Use LSTM cells instead of GRU cells")
tf.app.flags.DEFINE_integer("size", -1, "Size of each model layer (sets both embedding and hidden size).")
tf.app.flags.DEFINE_integer("embedding_size", 620, "Size of the word embeddings.")
tf.app.flags.DEFINE_integer("hidden_size", 1000, "Size of the hidden model layer.")
tf.app.flags.DEFINE_integer("num_layers", 1, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("num_samples", 512, "Number of samples if using sampled softmax (0 to not use it).")
tf.app.flags.DEFINE_boolean("norm_digits", False, "Normalise all digits to 0s")
tf.app.flags.DEFINE_boolean("use_seqlen", False, "Use sequence length for encoder inputs.")
tf.app.flags.DEFINE_boolean("use_src_mask", False, "Use source mask over for decoder attentions.")
tf.app.flags.DEFINE_boolean("maxout_layer", False, "If > 0, use a maxout layer of given size and full softmax instead of sampled softmax")
tf.app.flags.DEFINE_string("encoder", "reverse",
                            "Select encoder from 'reverse', 'bidirectional', 'bow'. The 'reverse' encoder is unidirectional and reverses the input "
                            "(default for tensorflow), the bidirectional encoder creates both forward and backward states and "
                            "concatenates them (like the Bahdanau model)")
tf.app.flags.DEFINE_string("model", "tensorflow", "Choose between: tensorflow,bahdanau,bahdanau_sgd. This will activate a bunch of flags to define "
                            "the model configuration (will override conflicting individual settings).")
tf.app.flags.DEFINE_string("config_file", None, "Pass options in a config file (will override conflicting command line settings)")

# Decoder settings
tf.app.flags.DEFINE_string("decode", None, "Set to stdin/dev/test, stdin evokes interactive decoding.")
tf.app.flags.DEFINE_string("decoder", "beam", "Select decoder from vanilla/greedy/beam.")
tf.app.flags.DEFINE_integer("beam_size", 12, "Beam size for beam decoder")
tf.app.flags.DEFINE_string("output_path", None, "Output path for decoding of dev/test set")
tf.app.flags.DEFINE_string("range", None, "Range for decoding of dev/test set")
tf.app.flags.DEFINE_boolean("length_norm", True, "Normalize hypothesis scores by length")
tf.app.flags.DEFINE_string("predictors", "nmt", "Comma separated scoring modules: nmt, nnlm, nplm, fst.")
tf.app.flags.DEFINE_string("predictor_weights", "1.0", "Comma separated scoring modules: nmt, nnlm, nplm, fst.")
tf.app.flags.DEFINE_string("fst_path", "fst/%d.fst", "Path to FSTs. Only required for predictor fst")
tf.app.flags.DEFINE_boolean("use_fst_weights", True, "Whether to use weights in FSTs for fst predictor.")
tf.app.flags.DEFINE_boolean("normalize_fst_weights", True, "Whether to normalize the weights in the FSTs.")

# Evaluate settings
tf.app.flags.DEFINE_string("evaluate", None, "Set to dev/test to compute multibleu scores of a previously decoded set.")

# Rename model variables
tf.app.flags.DEFINE_boolean("rename_model_vars", False, "Rename model variables with variable_prefix")

FLAGS = tf.app.flags.FLAGS

# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
_buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]

def process_config():
  # This overrides existing FLAG settings
  if FLAGS.config_file is None:
    return
  with open(FLAGS.config_file) as config:
    for line in config:
      key, value = line.strip().split(": ")
      if re.match("^\d+$", value):
        value = int(value)
      elif re.match("^[\d\.]+$", value):
        value = float(value)
      logging.info("Setting {} from config file: {}".format(key, value))
      FLAGS.__setattr__(key, value)

def process_flags():
  # This overrides individual settings for these config options
  if FLAGS.model.startswith("bahdanau"):
    FLAGS.opt_algorithm = "adadelta"
    FLAGS.learning_rate = 1.0
    FLAGS.max_gradient_norm = 1.0
    FLAGS.norm_digits = False
    FLAGS.encoder = "bidirectional"
    FLAGS.use_seqlen = True
    FLAGS.use_src_mask = True
    FLAGS.train_sequential = True
    FLAGS.use_lstm = False

    if FLAGS.model == "bahdanau_lstm_sgd_eos_maxout":
      FLAGS.use_lstm = True
      FLAGS.opt_algorithm = "sgd"
      FLAGS.add_src_eos = True
      FLAGS.maxout_layer = True
      FLAGS.num_samples = 0
    elif FLAGS.model == "bahdanau_lstm_sgd_eos_maxout_initbw":
      FLAGS.use_lstm = True
      FLAGS.opt_algorithm = "sgd"
      FLAGS.add_src_eos = True
      FLAGS.maxout_layer = True
      FLAGS.num_samples = 0
      FLAGS.init_backward = True
    elif FLAGS.model == "bahdanau_lstm_sgd_eos_maxout_initbw_small":
      FLAGS.use_lstm = True
      FLAGS.opt_algorithm = "sgd"
      FLAGS.add_src_eos = True
      FLAGS.maxout_layer = True
      FLAGS.num_samples = 0
      FLAGS.init_backward = True
      FLAGS.embedding_size = 300
      FLAGS.hidden_size = 500
      FLAGS.no_pad_symbol = True

  # BOW model setup
  if FLAGS.model.startswith("bow"):
    FLAGS.learning_rate = 1.0
    FLAGS.max_gradient_norm = 1.0
    FLAGS.norm_digits = False
    #FLAGS.use_seqlen = True # not needed
    FLAGS.use_src_mask = True
    FLAGS.train_sequential = True
    FLAGS.encoder = "bow"
    FLAGS.use_lstm = True
    FLAGS.opt_algorithm = "sgd"
    FLAGS.add_src_eos = False
    FLAGS.maxout_layer = True
    FLAGS.num_samples = 0
    FLAGS.no_pad_symbol = True

    if FLAGS.model == "bow_ptb":
      FLAGS.embedding_size = 300
      FLAGS.hidden_size = 500

  elif FLAGS.model == "tensorflow_1layer":
    FLAGS.use_lstm = True
    FLAGS.learning_rate = 0.5
  elif FLAGS.model == "tensorflow_2layers":
    FLAGS.use_lstm = True
    FLAGS.learning_rate = 0.5
    FLAGS.num_layers = 2

  if FLAGS.opt_algorithm not in [ "sgd", "adagrad", "adadelta" ]:
    raise Exception("Unknown optimization algorithm: {}".format(FLAGS.opt_algorithm))
  if FLAGS.size != -1:
    FLAGS.embedding_size = FLAGS.size
    FLAGS.hidden_size = FLAGS.size
  if FLAGS.num_symm_buckets > 0:
    make_x_buckets(FLAGS.num_symm_buckets, True if FLAGS.decode is None else False)

  # Combination schemes
  if FLAGS.length_norm:
    core.breakdown2score_partial = core.breakdown2score_length_norm
    core.breakdown2score_full = core.breakdown2score_length_norm

  if FLAGS.swap_pad_unk:
    data_utils.swap_pad_unk()
    logging.info("UNK_ID=%d" % data_utils.UNK_ID)

  if FLAGS.no_pad_symbol:
    data_utils.no_pad_symbol()
    logging.info("UNK_ID=%d" % data_utils.UNK_ID)
    logging.info("PAD_ID=%d" % data_utils.PAD_ID)

def make_x_buckets(x, train=True):
  global _buckets
  max_seq_len = 50
  if train:
    # Make target buckets 2 tokens longer because of GO and EOS symbols
    _buckets = [ (int(max_seq_len/x)*i, int(max_seq_len/x)*i+2) for i in range(1,x+1) ]
  else:
    _buckets = [ (int(max_seq_len/x)*i, int(max_seq_len/x)*i+5) for i in range(1,x+1) ]
  logging.info("Use buckets={}".format(_buckets))

def read_data(source_path, target_path, max_size=None, src_vcb_size=None, trg_vcb_size=None):
  """Read data from source and target files and put into buckets.

  Args:
    source_path: path to the files with token-ids for the source language.
    target_path: path to the file with token-ids for the target language;
      it must be aligned with the source file: n-th line contains the desired
      output for n-th line from the source_path.
    max_size: maximum number of lines to read, all other will be ignored;
      if 0 or None, data files will be read completely (no limit).

  Returns:
    data_set: a list of length len(_buckets); data_set[n] contains a list of
      (source, target) pairs read from the provided data files that fit
      into the n-th bucket, i.e., such that len(source) < _buckets[n][0] and
      len(target) < _buckets[n][1]; source and target are lists of token-ids.
  """
  if FLAGS.add_src_eos:
    logging.info("Add EOS symbol to all source sentences")
  if src_vcb_size:
    logging.info("Replace OOV words with id={} for src_vocab_size={}".format(data_utils.UNK_ID, src_vcb_size))
  if trg_vcb_size:
    logging.info("Replace OOV words with id={} for trg_vocab_size={}".format(data_utils.UNK_ID, trg_vcb_size))

  data_set = [[] for _ in _buckets]
  with tf.gfile.GFile(source_path, mode="r") as source_file:
    with tf.gfile.GFile(target_path, mode="r") as target_file:
      source, target = source_file.readline(), target_file.readline()
      counter = 0
      while source and target and (not max_size or counter < max_size):
        counter += 1
        if counter % 100000 == 0:
          logging.info("  reading data line %d" % counter)
          sys.stdout.flush()

        source_ids = [int(x) for x in source.split()]
        if FLAGS.add_src_eos:
          source_ids.append(data_utils.EOS_ID)
        target_ids = [int(x) for x in target.split()]
        target_ids.append(data_utils.EOS_ID)

        if src_vcb_size:
          # replace source OOV words with unk (in case this has not been done on the source side)
          source_ids = [ wid if wid < src_vcb_size else data_utils.UNK_ID for wid in source_ids ]

        if trg_vcb_size:
          # replace target OOV words with unk (in case this has not been done on the target side)
          target_ids = [ wid if wid < trg_vcb_size else data_utils.UNK_ID for wid in target_ids ]

        for bucket_id, (source_size, target_size) in enumerate(_buckets):
          #if len(source_ids) < source_size and len(target_ids) < target_size:
          if len(source_ids) <= source_size and len(target_ids) < target_size:
            data_set[bucket_id].append([source_ids, target_ids])
            break
        source, target = source_file.readline(), target_file.readline()
  return data_set


def create_model(session, forward_only, buckets, opt_algorithm="sgd",
                 single_steps=False, rename_variable_prefix=None):
  """Create translation model and initialize or load parameters in session."""
  if single_steps:
    model = tf_seq2seq.TFSeq2SeqEngine(
      FLAGS.src_vocab_size, FLAGS.trg_vocab_size, buckets,
      FLAGS.embedding_size,FLAGS.hidden_size,
      FLAGS.num_layers, FLAGS.max_gradient_norm, 1, # Batch size is 1, not FLAGS.batch_size,
      FLAGS.learning_rate, FLAGS.learning_rate_decay_factor, use_lstm=FLAGS.use_lstm,
      num_samples=FLAGS.num_samples, forward_only=True,
      opt_algorithm=opt_algorithm, encoder=FLAGS.encoder,
      use_sequence_length=FLAGS.use_seqlen, use_src_mask=FLAGS.use_src_mask, maxout_layer=FLAGS.maxout_layer,
      init_backward=FLAGS.init_backward, no_pad_symbol=FLAGS.no_pad_symbol,
      variable_prefix=FLAGS.variable_prefix,
      init_const=FLAGS.bow_init_const, use_bow_mask=FLAGS.use_bow_mask)
  else:
    model = seq2seq_model.Seq2SeqModel(
      FLAGS.src_vocab_size, FLAGS.trg_vocab_size, buckets,
      FLAGS.embedding_size,FLAGS.hidden_size,
      FLAGS.num_layers, FLAGS.max_gradient_norm, FLAGS.batch_size,
      FLAGS.learning_rate, FLAGS.learning_rate_decay_factor, use_lstm=FLAGS.use_lstm,
      num_samples=FLAGS.num_samples, forward_only=forward_only,
      opt_algorithm=opt_algorithm, encoder=FLAGS.encoder,
      use_sequence_length=FLAGS.use_seqlen, use_src_mask=FLAGS.use_src_mask, maxout_layer=FLAGS.maxout_layer,
      init_backward=FLAGS.init_backward, no_pad_symbol=FLAGS.no_pad_symbol,
      variable_prefix=FLAGS.variable_prefix, rename_variable_prefix=rename_variable_prefix,
      init_const=FLAGS.bow_init_const, use_bow_mask=FLAGS.use_bow_mask,
      max_to_keep=FLAGS.max_to_keep)

    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
      logging.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
      model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
      logging.info("Created model with fresh parameters.")
      session.run(tf.initialize_all_variables())
  return model, ckpt


def train():
  """Train a en->fr translation model using WMT data."""
  # Prepare WMT data.
  logging.info("Preparing data in %s" % FLAGS.data_dir)
  if (FLAGS.encoder == "bow") and FLAGS.src_lang == FLAGS.trg_lang:
    logging.info("Preparing monolingual training data")
    if FLAGS.default_filenames:
      trg_train = os.path.join(FLAGS.data_dir, "train.ids." + FLAGS.trg_lang)
      src_train = trg_train
      trg_dev = os.path.join(FLAGS.data_dir, "dev.ids." + FLAGS.trg_lang)
      src_dev = trg_dev
    else:
      src_train, trg_train, src_dev, trg_dev = data_utils.get_mono_news_data(
        FLAGS.data_dir, FLAGS.trg_vocab_size, FLAGS.trg_lang)
  else:
    logging.info("Preparing bilingual training data")
    if FLAGS.default_filenames:
      src_train = os.path.join(FLAGS.data_dir, "train.ids." + FLAGS.src_lang)
      trg_train = os.path.join(FLAGS.data_dir, "train.ids." + FLAGS.trg_lang)
      src_dev = os.path.join(FLAGS.data_dir, "dev.ids." + FLAGS.src_lang)
      trg_dev = os.path.join(FLAGS.data_dir, "dev.ids." + FLAGS.trg_lang)
    else:
      src_train, trg_train, src_dev, trg_dev, _, _ = data_utils.prepare_wmt_data(
        FLAGS.data_dir, FLAGS.src_vocab_size, FLAGS.trg_vocab_size, FLAGS.src_lang,
        FLAGS.trg_lang, FLAGS.use_default_data, tokenizer=None, normalize_digits=FLAGS.norm_digits)

  device = "/cpu:0"
  log_device_placement = False
  allow_soft_placement = True
  if FLAGS.device:
    device = '/'+FLAGS.device
  logging.info("Use device %s" % device)

  with tf.Session(config=tf.ConfigProto(allow_soft_placement=allow_soft_placement, log_device_placement=log_device_placement)) as sess, tf.device(device):
    # Create model.
    if FLAGS.fixed_random_seed:
      tf.set_random_seed(1234)
    logging.info("Creating %d layers of %d units, encoder=%s." % (FLAGS.num_layers, FLAGS.hidden_size, FLAGS.encoder))
    model, _ = create_model(sess, False, _buckets, FLAGS.opt_algorithm)

    # Read data into buckets and compute their sizes.
    logging.info ("Reading development and training data (limit: %d)."
           % FLAGS.max_train_data_size)
    dev_set = read_data(src_dev, trg_dev, 
                        src_vcb_size=FLAGS.src_vocab_size, trg_vcb_size=FLAGS.trg_vocab_size)
    train_set = read_data(src_train, trg_train, FLAGS.max_train_data_size, 
                          src_vcb_size=FLAGS.src_vocab_size, trg_vcb_size=FLAGS.trg_vocab_size)
    train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_buckets))]
    train_size = float(sum(train_bucket_sizes))
    logging.info ("Training bucket sizes: {}".format(train_bucket_sizes))

    # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
    # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
    # the size if i-th training bucket, as used later.
    train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_size
                           for i in xrange(len(train_bucket_sizes))]

    if FLAGS.train_sequential:
      # Create a list of (bucket_id, off_set) tuples and walk through it sequentially,
      # shuffling the buckets every epoch
      bucket_offset_pairs = []
      for b in xrange(len(_buckets)):
        for idx in xrange(len(train_set[b])):
          if idx % model.batch_size == 0:
            bucket_offset_pairs.append((b, idx))

        # Make sure every bucket has num_items % batch_size == 0 by adding random samples
        if len(train_set[b]) % model.batch_size > 0:
          num_extra = model.batch_size - (len(train_set[b]) % model.batch_size)
          if FLAGS.shuffle_data:
            samples = [ int(r * 10000) for r in np.random.random_sample(num_extra) ]
            for s in samples:
              while s >= len(train_set[b]):
                s = int(s/10)
              train_set[b].append(train_set[b][s])
          else:
            # For reproducibility, append the first n examples from a given bucket to its end
            for i in xrange(num_extra):
              train_set[b].append(train_set[b][i])
          assert len(train_set[b]) % model.batch_size == 0, "len(train_set[b])=%i mod model.batch_size=%i != 0" % (len(train_set[b]), model.batch_size)

      # For each bucket, create a list of indices which we can shuffle and therefore use as a mapping instead of shuffling the data
      train_idx_map = [ [] for b in xrange(len(_buckets))]
      for b in xrange(len(_buckets)):
        for idx in xrange(len(train_set[b])):
          train_idx_map[b].append(idx)
        assert len(train_idx_map[b]) == len(train_set[b]), "Inconsistent train idx map for bucket %i" % b

      train_size_old = train_size
      train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_buckets))]
      train_size = float(sum(train_bucket_sizes))
      num_train_batches = int(train_size / model.batch_size)
      num_batch_pointers = len(bucket_offset_pairs)
      assert num_batch_pointers == num_train_batches, "Inconsistent number of batch pointers: %i != %i" % (num_batch_pointers, num_train_batches)
      logging.info("Total size of training set adjusted from %i to %i" % (train_size_old, train_size))
      logging.info("Total number of training batches=%i" % num_train_batches)
    else:
      logging.info ("Total size of training set=%i" % train_size)

    # This is the training loop.
    step_time, loss = 0.0, 0.0
    current_step = 0
    previous_losses = []
    logging.info("Time: {}".format(datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')))
    sys.stdout.flush()

    bookk = defaultdict(dict)

    # Restore saved shuffled train variables
    tmpfile = FLAGS.train_dir+"/tmp_shuffled.pkl"
    if FLAGS.train_sequential and model.global_step.eval() >= FLAGS.steps_per_checkpoint:
        if tf.gfile.Exists(tmpfile):
          logging.info("Restore saved shuffled train variables from %s" % tmpfile)
          with open(tmpfile, "rb") as f:
            train_idx_map, bucket_offset_pairs, bookk = pickle.load(f)
        else:
          logging.info("No file with saved shuffled train variables available, using new shuffling.")

    store_shuffled_train_vars = False
    epoch = 0
    while True:
      if FLAGS.train_sequential:
        global_step = model.global_step.eval() # a step is one batch
        current_batch_idx = global_step % num_train_batches

        if current_batch_idx == 0:
          epoch = int(global_step / num_train_batches)

          if epoch > 0 and bookk is not None:
            # This is for debugging only: check if all training examples have been processed
            lengths = [ len(bookk[b].keys()) for b in bookk.keys() ]
            logging.info("After epoch %i: Total examples = %i, processed examples = %i" % (epoch, train_size, sum(lengths)))
            #assert train_size == sum(lengths), "ERROR: training set has not been fully processed"
            if train_size != sum(lengths):
              logging.info("ERROR: training set has not been fully processed")
            bookk = defaultdict(dict)

          if FLAGS.shuffle_data:
            if not (FLAGS.max_train_epochs > 0 and epoch >= FLAGS.max_train_epochs):
              logging.info("Epoch = %i, shuffling train idx maps and batch pointers" % (int(epoch)+1))
            store_shuffled_train_vars = True
            for b in xrange(len(_buckets)):
              random.shuffle(train_idx_map[b]) # shuffle training idx map for each bucket
            random.shuffle(bucket_offset_pairs) # shuffle the bucket_id, offset pairs

          if FLAGS.adjust_lr and FLAGS.max_epoch > 0:
            # adjust learning rate independent of performance
            lr_decay_factor = FLAGS.learning_rate_decay_factor ** max(epoch - FLAGS.max_epoch, 0.0)
            sess.run(model.learning_rate.assign(FLAGS.learning_rate * lr_decay_factor))
            logging.info("Learning rate = {}".format(model.learning_rate.eval()))

        # bucket_offset_pair holds a bucket_id and train_idx
        # train_idx is the offset for the batch in the given bucket: all subsequent indices belong to the batch as well
        bucket_offset_pair = bucket_offset_pairs[current_batch_idx]
        bucket_id = bucket_offset_pair[0]
        train_offset = bucket_offset_pair[1]
        # idx_map is used to map the train indices to randomly assigned indices in the same bucket
        idx_map = train_idx_map[bucket_id]

        # This is for debugging only: making sure the order is preserved after reloading the model
        if current_batch_idx+2 < len(bucket_offset_pairs) and (current_step+1) % FLAGS.steps_per_checkpoint == 0:
          bucket_offset_pair_1 = bucket_offset_pairs[current_batch_idx+1]
          bucket_offset_pair_2 = bucket_offset_pairs[current_batch_idx+2]
          idx_map_1 = train_idx_map[bucket_offset_pair_1[0]]
          idx_map_2 = train_idx_map[bucket_offset_pair_2[0]]
          logging.info("Global step = {}, current batch idx = {} current batch = {} -> {}, next two batches = {} -> {}, {} -> {}" .format(model.global_step.eval(), current_batch_idx, \
                        bucket_offset_pair, idx_map[train_offset], \
                        bucket_offset_pair_1, idx_map_1[bucket_offset_pair_1[1]], \
                        bucket_offset_pair_2, idx_map_2[bucket_offset_pair_2[1]] ))
        else:
          logging.info("Global step = {}, current batch idx = {} current batch = {} -> {}".format(model.global_step.eval(), current_batch_idx, bucket_offset_pair, idx_map[train_offset] ))

        batch_ptr = { "offset": train_offset, "idx_map": idx_map }
      else:
        # Choose a bucket according to data distribution. We pick a random number
        # in [0, 1] and use the corresponding interval in train_buckets_scale.
        random_number_01 = np.random.random_sample()
        bucket_id = min([i for i in xrange(len(train_buckets_scale))
                       if train_buckets_scale[i] > random_number_01])
        logging.debug("bucket_id=%d" % bucket_id)

      if (FLAGS.max_train_batches > 0 and model.global_step.eval() >= FLAGS.max_train_batches) or \
         (FLAGS.max_train_epochs > 0 and epoch >= FLAGS.max_train_epochs):
        # Save final model
        checkpoint_path = os.path.join(FLAGS.train_dir, "translate.ckpt")
        print("Save final model to path=%s after global step=%d" % (checkpoint_path, model.global_step.eval()))
        model.saver.save(sess, checkpoint_path, global_step=model.global_step)
        print("Time: {}".format(datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')))
        return

      # Get a batch and make a step.
      start_time = time.time()
      encoder_inputs, decoder_inputs, target_weights, sequence_length, src_mask, bow_mask = model.get_batch(
        train_set, bucket_id, FLAGS.encoder, batch_ptr=batch_ptr if FLAGS.train_sequential else None, bookk=bookk)

      _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                   target_weights, bucket_id, False,
                                   sequence_length, src_mask, bow_mask)

      step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
      loss += step_loss / FLAGS.steps_per_checkpoint
      current_step += 1

      # Once in a while, we save checkpoint, print statistics, and run evals.
      if current_step % FLAGS.steps_per_checkpoint == 0:
        print("Time: {}".format(datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')))
        # Print statistics for the previous epoch.
        perplexity = math.exp(loss) if loss < 300 else float('inf')
        if FLAGS.opt_algorithm == "sgd":
          print ("global step %d learning rate %.4f step-time %.2f perplexity "
                 "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                           step_time, perplexity))
        else:
          print ("global step %d step-time %.2f perplexity "
                 "%.2f" % (model.global_step.eval(),
                           step_time, perplexity))

        # Decrease learning rate if no improvement was seen over last 3 times.
        if len(previous_losses) > 2 and loss > max(previous_losses[-3:]) and \
           FLAGS.opt_algorithm == "sgd":
          sess.run(model.learning_rate_decay_op)
          print ("Decrease learning rate --> {}".format(model.learning_rate.eval()))
        previous_losses.append(loss)

        # Save shuffled train variables
        if FLAGS.train_sequential and store_shuffled_train_vars:
          with open(tmpfile, "wb") as f:
            print("Saving shuffled train variables to path=%s" % tmpfile)
            pickle.dump((train_idx_map, bucket_offset_pairs, bookk), f, pickle.HIGHEST_PROTOCOL)
            store_shuffled_train_vars = False

        # Save checkpoint and zero timer and loss.
        checkpoint_path = os.path.join(FLAGS.train_dir, "translate.ckpt")
        print("Save model to path=%s after global step=%d" % (checkpoint_path, model.global_step.eval()))
        model.saver.save(sess, checkpoint_path, global_step=model.global_step)
        step_time, loss = 0.0, 0.0
        sys.stdout.flush()

      if current_step % (FLAGS.steps_per_checkpoint * 6) == 0:
        # Run evals on development set and print their perplexity.
        logging.info("Run eval on development set")
        for bucket_id in xrange(len(_buckets)):
          if len(dev_set[bucket_id]) == 0:
            print("  eval: empty bucket %d" % (bucket_id))
            continue

          encoder_inputs, decoder_inputs, target_weights, sequence_length, src_mask, bow_mask = model.get_batch(
            dev_set, bucket_id, FLAGS.encoder)

          _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                       target_weights, bucket_id, True,
                                       sequence_length, src_mask, bow_mask)
          eval_ppx = math.exp(eval_loss) if eval_loss < 300 else float('inf')
          print("  eval: global step %d bucket %d perplexity %.2f" % (model.global_step.eval(), bucket_id, eval_ppx))
        sys.stdout.flush()

def decode(filetype, decoder_type=FLAGS.decoder):
    # Load vocabularies.
    src_vocab_path = os.path.join(FLAGS.data_dir,
                              "vocab%d.%s" % (FLAGS.src_vocab_size, FLAGS.src_lang))
    trg_vocab_path = os.path.join(FLAGS.data_dir,
                              "vocab%d.%s" % (FLAGS.trg_vocab_size, FLAGS.trg_lang))
    src_vocab, _ = data_utils.initialize_vocabulary(src_vocab_path)
    _, rev_trg_vocab = data_utils.initialize_vocabulary(trg_vocab_path)

    is_tokenized = True
    with tf.Session() as session:
      if FLAGS.decode == "stdin":
        # Decode from standard input
        decode_stdin(session, decoder_type, src_vocab, rev_trg_vocab, is_tokenized)
      else:
        # Decode dev/test set
        if FLAGS.use_default_data:
          is_tokenized = False
        decode_set(session, decoder_type, filetype, src_vocab, rev_trg_vocab, is_tokenized)

def add_predictors(decoder, session, model):
    if decoder.name == "vanilla":
        return
    weights = None
    if FLAGS.predictor_weights:
      weights = FLAGS.predictor_weights.strip().split(",")
    preds = FLAGS.predictors.split(",")
    if not preds:
        logging.fatal("Require at least one predictor!")
    for idx,pred in enumerate(preds):
        if pred == "nmt":
            p = NMTPredictor(session, model, FLAGS.encoder, FLAGS.use_seqlen, FLAGS.use_src_mask)
        elif pred == "fst":
            p = FstPredictor(FLAGS.fst_path, FLAGS.use_fst_weights, FLAGS.normalize_fst_weights)
        else:
            logging.fatal("Predictor %s not available." % pred)
        if weights:
          weight = float(weights[idx])
          logging.info("Add predictor %s with weight %f" % (pred, weight))
          decoder.add_predictor(pred, p, weight)
        else:
          decoder.add_predictor(pred, p)

def create_model_and_decoder(session, buckets, decoder_type):
    single_steps = False if decoder_type == "vanilla" else True
    model, _ = create_model(session, True, buckets, FLAGS.opt_algorithm, single_steps)
    model.batch_size = 1  # We decode one sentence at a time.

    if decoder_type == "vanilla":
        decoder = VanillaDecoder(session, model)
    elif decoder_type == "greedy":
        decoder = GreedyDecoder()
    elif decoder_type == "beam":
        decoder = BeamDecoder(FLAGS.beam_size, closed_vocab_norm=core.CLOSED_VOCAB_SCORE_NORM_NONE)
    else:
        raise Exception("Unknown decoder type={}".format(FLAGS.decoder))
        sys.exit(1)
    add_predictors(decoder, session, model)
    return model, decoder

def prepare_sentence(sentence, src_vocab, is_tokenized, indexed=False, dynamic=False):
  """ If input is raw sentence, tokenize (if not already) and index it. Otherwise just split it.
  Return its bucket id."""
  # If not indexed, get token-ids for the input sentence
  token_ids = sentence.split() if indexed else \
    data_utils.sentence_to_token_ids(tf.compat.as_bytes(sentence), src_vocab, is_tokenized,
                                     normalize_digits=FLAGS.norm_digits)
  if FLAGS.add_src_eos:
    token_ids.append(data_utils.EOS_ID)

  # Which bucket does it belong to?
  #if _buckets[-1][0] > len(token_ids):
  if _buckets[-1][0] >= len(token_ids):
    bucket_id = min([b for b in xrange(len(_buckets))
        #if _buckets[b][0] > len(token_ids)])
        if _buckets[b][0] >= len(token_ids)])
    logging.info("bucket={}".format(_buckets[bucket_id]))
    # only return new buckets
    bucket = None
  else:
    # there is no suitable bucket
    #bucket = (len(token_ids)+1, len(token_ids)+6)
    bucket = (len(token_ids), len(token_ids)+5)

  return token_ids, bucket

def decode_stdin(session, decoder_type, src_vocab, rev_trg_vocab, is_tokenized):
  model, decoder = create_model_and_decoder(session, _buckets, decoder_type)
  sys.stdout.write("Input (lang=%s, tokenized=%s) > " % (FLAGS.src_lang, is_tokenized))
  sys.stdout.flush()
  sentence = sys.stdin.readline()
  while sentence:
    target, score = decode_extend(model, decoder, sentence, src_vocab, rev_trg_vocab, is_tokenized, indexed=False)
    print(target, score)
    print("> ", end="")
    sentence = sys.stdin.readline()

def decode_set(session, decoder_type, filetype, src_vocab, rev_trg_vocab, is_tokenized):
  model, decoder = create_model_and_decoder(session, _buckets, decoder_type)
  out_filename = os.path.join(FLAGS.train_dir, filetype+".out") if FLAGS.output_path == None \
                  else FLAGS.output_path
  src_filename, _, src_size = data_utils.get_data_set(
  filetype, FLAGS.data_dir, FLAGS.src_vocab_size, FLAGS.trg_vocab_size, FLAGS.src_lang,
  FLAGS.trg_lang, FLAGS.use_default_data)
  logging.info("Decoder input={}".format(src_filename))
  logging.info("Writing decoder output to file {}".format(out_filename))
  with open(src_filename) as srcfile, open(out_filename, "w") as outfile:
    for idx in xrange(1,src_size+1):
      line = srcfile.readline().rstrip()
      if FLAGS.range != None and not in_range(FLAGS.range, idx):
        continue
      target, _ = decode_extend(model, decoder, line, src_vocab, rev_trg_vocab, is_tokenized, indexed=True)
      outfile.write("{}\n".format(target))
      outfile.flush()

def decode_extend(model, decoder, line, src_vocab, rev_trg_vocab, is_tokenized, indexed):
  token_ids, bucket = prepare_sentence(line, src_vocab, is_tokenized, indexed)
  if bucket is not None:
    logging.info("Add new bucket={} and update model".format(bucket))
    global _buckets
    _buckets.append(bucket)
    model.update_buckets(_buckets)

  hypo = decoder.decode(token_ids)[0]
  target = " ".join([tf.compat.as_str(rev_trg_vocab[tok]) for tok in hypo.trgt_sentence[:-1]])
  score = hypo.total_score
  return target, score

def in_range(range, idx):
  start, end = range.split(':')
  return True if idx >= int(start) and idx <= int(end) else False

def eval_set(filetype):
  # Evaluate dev/test set: multi-bleu.pl [-lc] reference < hypothesis
  filename = os.path.join(FLAGS.train_dir, filetype+".out")
  reference = os.path.join(FLAGS.data_dir, filetype, filetype+"."+FLAGS.trg_lang)
  cat = subprocess.Popen(("cat", filename), stdout=subprocess.PIPE)
  multibleu = subprocess.check_output(("/home/mifs/ech57/code/tensorflow/tensorflow/models/rnn/translate/eval/multi-bleu.perl",
                                       reference), stdin=cat.stdout)
  print("{}".format(multibleu))
  import re
  m = re.match("BLEU = ([\d.]+),", multibleu)
  return float(m.group(1))

def rename_model_vars():
  logging.info("Rename model variables with prefix %s" % FLAGS.variable_prefix)
  with tf.Session() as session:
    # Create model and restore variable
    variable_prefix = FLAGS.variable_prefix
    FLAGS.variable_prefix = None
    logging.info("Creating %d layers of %d units, encoder=%s." % (FLAGS.num_layers, FLAGS.hidden_size, FLAGS.encoder))
    model, ckpt = create_model(session, False, _buckets, FLAGS.opt_algorithm, rename_variable_prefix=variable_prefix)

    # Save model with new variable names
    checkpoint_path = os.path.join(FLAGS.train_dir+"_mult", os.path.basename(ckpt.model_checkpoint_path))
    print("Save model to path=%s using saver_prefix" % checkpoint_path)
    model.saver_prefix.save(session, checkpoint_path)

def main(_):
  process_config()
  process_flags()

  if not FLAGS.use_default_data:
    logging.info ("Basic tokenizer will be suppressed (all data assumed to be tokenized)")
  if FLAGS.decode != None:
    decode(FLAGS.decode)
  elif FLAGS.evaluate != None:
    eval_set(FLAGS.evaluate)
  elif FLAGS.rename_model_vars:
    rename_model_vars()
  else:
    train()

if __name__ == "__main__":
  logging.getLogger().setLevel(logging.INFO)

  logging.info("Start: {}".format(datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')))
  tf.app.run()
  logging.info("End: {}".format(datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')))

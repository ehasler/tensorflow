# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

"""Utilities for downloading data from WMT, tokenizing, vocabularies."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import re
import tarfile

from six.moves import urllib

import tensorflow as tf
from tensorflow.python.platform import gfile
import logging

# Special vocabulary symbols - we always put them at the start.
_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")
_DIGIT_RE = re.compile(br"\d")

# URLs for WMT data.
_WMT_ENFR_TRAIN_URL = "http://www.statmt.org/wmt10/training-giga-fren.tar"
_WMT_ENFR_DEV_URL = "http://www.statmt.org/wmt15/dev-v2.tgz"

def no_pad_symbol():
  global PAD_ID
  global UNK_ID
  UNK_ID = 0
  PAD_ID = -1

def maybe_download(directory, filename, url):
  """Download filename from url unless it's already in directory."""
  if not os.path.exists(directory):
    print("Creating directory %s" % directory)
    os.mkdir(directory)
  filepath = os.path.join(directory, filename)
  if not os.path.exists(filepath):
    print("Downloading %s to %s" % (url, filepath))
    filepath, _ = urllib.request.urlretrieve(url, filepath)
    statinfo = os.stat(filepath)
    print("Succesfully downloaded", filename, statinfo.st_size, "bytes")
  return filepath


def gunzip_file(gz_path, new_path):
  """Unzips from gz_path into new_path."""
  print("Unpacking %s to %s" % (gz_path, new_path))
  with gzip.open(gz_path, "rb") as gz_file:
    with open(new_path, "wb") as new_file:
      for line in gz_file:
        new_file.write(line)


def get_wmt_enfr_train_set(directory):
  """Download the WMT en-fr training corpus to directory unless it's there."""
  train_path = os.path.join(directory, "giga-fren.release2.fixed")
  if not (gfile.Exists(train_path +".fr") and gfile.Exists(train_path +".en")):
    corpus_file = maybe_download(directory, "training-giga-fren.tar",
                                 _WMT_ENFR_TRAIN_URL)
    print("Extracting tar file %s" % corpus_file)
    with tarfile.open(corpus_file, "r") as corpus_tar:
      corpus_tar.extractall(directory)
    gunzip_file(train_path + ".fr.gz", train_path + ".fr")
    gunzip_file(train_path + ".en.gz", train_path + ".en")
  return train_path


def get_wmt_enfr_dev_set(directory):
  """Download the WMT en-fr training corpus to directory unless it's there."""
  dev_name = "newstest2013"
  dev_path = os.path.join(directory, dev_name)
  if not (gfile.Exists(dev_path + ".fr") and gfile.Exists(dev_path + ".en")):
    dev_file = maybe_download(directory, "dev-v2.tgz", _WMT_ENFR_DEV_URL)
    print("Extracting tgz file %s" % dev_file)
    with tarfile.open(dev_file, "r:gz") as dev_tar:
      fr_dev_file = dev_tar.getmember("dev/" + dev_name + ".fr")
      en_dev_file = dev_tar.getmember("dev/" + dev_name + ".en")
      fr_dev_file.name = dev_name + ".fr"  # Extract without "dev/" prefix.
      en_dev_file.name = dev_name + ".en"
      dev_tar.extract(fr_dev_file, directory)
      dev_tar.extract(en_dev_file, directory)
  return dev_path


def basic_tokenizer(sentence):
  """Very basic tokenizer: split the sentence into a list of tokens."""
  words = []
  for space_separated_fragment in sentence.strip().split():
    words.extend(_WORD_SPLIT.split(space_separated_fragment))
  return [w for w in words if w]


def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size,
                      tokenizer=None, normalize_digits=True):
  """Create vocabulary file (if it does not exist yet) from data file.
  Data file is assumed to contain one sentence per line. Each sentence is
  tokenized and digits are normalized (if normalize_digits is set).
  Vocabulary contains the most-frequent tokens up to max_vocabulary_size.
  We write it to vocabulary_path in a one-token-per-line format, so that later
  token in the first line gets id=0, second line gets id=1, and so on.
  Args:
    vocabulary_path: path where the vocabulary will be created.
    data_path: data file that will be used to create vocabulary.
    max_vocabulary_size: limit on the size of the created vocabulary.
    tokenizer: a function to use to tokenize each data sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.
  """
  if not gfile.Exists(vocabulary_path):
    print("Creating vocabulary %s from data %s" % (vocabulary_path, data_path))
    vocab = {}
    with gfile.GFile(data_path, mode="rb") as f:
      counter = 0
      for line in f:
        counter += 1
        if counter % 100000 == 0:
          print("  processing line %d" % counter)
        line = tf.compat.as_bytes(line)
        tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
        for w in tokens:
          word = _DIGIT_RE.sub(b"0", w) if normalize_digits else w
          if word in vocab:
            vocab[word] += 1
          else:
            vocab[word] = 1
      vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
      if len(vocab_list) > max_vocabulary_size:
        vocab_list = vocab_list[:max_vocabulary_size]
      with gfile.GFile(vocabulary_path, mode="wb") as vocab_file:
        for w in vocab_list:
          vocab_file.write(w + b"\n")


def initialize_vocabulary(vocabulary_path):
  """Initialize vocabulary from file.
  We assume the vocabulary is stored one-item-per-line, so a file:
    dog
    cat
  will result in a vocabulary {"dog": 0, "cat": 1}, and this function will
  also return the reversed-vocabulary ["dog", "cat"].
  Args:
    vocabulary_path: path to the file containing the vocabulary.
  Returns:
    a pair: the vocabulary (a dictionary mapping string to integers), and
    the reversed vocabulary (a list, which reverses the vocabulary mapping).
  Raises:
    ValueError: if the provided vocabulary_path does not exist.
  """
  if gfile.Exists(vocabulary_path):
    rev_vocab = []
    with gfile.GFile(vocabulary_path, mode="rb") as f:
      rev_vocab.extend(f.readlines())
    rev_vocab = [line.strip() for line in rev_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
    return vocab, rev_vocab
  else:
    raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def sentence_to_token_ids(sentence, vocabulary,
                          tokenizer=None, normalize_digits=True):
  """Convert a string to list of integers representing token-ids.
  For example, a sentence "I have a dog" may become tokenized into
  ["I", "have", "a", "dog"] and with vocabulary {"I": 1, "have": 2,
  "a": 4, "dog": 7"} this function will return [1, 2, 4, 7].
  Args:
    sentence: the sentence in bytes format to convert to token-ids.
    vocabulary: a dictionary mapping tokens to integers.
    tokenizer: a function to use to tokenize each sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.
  Returns:
    a list of integers, the token-ids for the sentence.
  """

  if tokenizer:
    words = tokenizer(sentence)
  else:
    words = basic_tokenizer(sentence)
  if not normalize_digits:
    return [vocabulary.get(w, UNK_ID) for w in words]
  # Normalize digits by 0 before looking words up in the vocabulary.
  return [vocabulary.get(_DIGIT_RE.sub(b"0", w), UNK_ID) for w in words]


def data_to_token_ids(data_path, target_path, vocabulary_path,
                      tokenizer=None, normalize_digits=True):
  """Tokenize data file and turn into token-ids using given vocabulary file.
  This function loads data line-by-line from data_path, calls the above
  sentence_to_token_ids, and saves the result to target_path. See comment
  for sentence_to_token_ids on the details of token-ids format.
  Args:
    data_path: path to the data file in one-sentence-per-line format.
    target_path: path where the file with token-ids will be created.
    vocabulary_path: path to the vocabulary file.
    tokenizer: a function to use to tokenize each sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.
  """
  if not gfile.Exists(target_path):
    print("Tokenizing data in %s" % data_path)
    vocab, _ = initialize_vocabulary(vocabulary_path)
    with gfile.GFile(data_path, mode="rb") as data_file:
      with gfile.GFile(target_path, mode="w") as tokens_file:
        counter = 0
        for line in data_file:
          counter += 1
          if counter % 100000 == 0:
            print("  tokenizing line %d" % counter)
          token_ids = sentence_to_token_ids(line, vocab, tokenizer,
                                            normalize_digits)
          tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")


def prepare_wmt_data(data_dir, en_vocabulary_size, fr_vocabulary_size, tokenizer=None):
  """Get WMT data into data_dir, create vocabularies and tokenize data.
  Args:
    data_dir: directory in which the data sets will be stored.
    en_vocabulary_size: size of the English vocabulary to create and use.
    fr_vocabulary_size: size of the French vocabulary to create and use.
    tokenizer: a function to use to tokenize each data sentence;
      if None, basic_tokenizer will be used.
  Returns:
    A tuple of 6 elements:
      (1) path to the token-ids for English training data-set,
      (2) path to the token-ids for French training data-set,
      (3) path to the token-ids for English development data-set,
      (4) path to the token-ids for French development data-set,
      (5) path to the English vocabulary file,
      (6) path to the French vocabulary file.
  """
  # Get wmt data to the specified directory.
  train_path = get_wmt_enfr_train_set(data_dir)
  dev_path = get_wmt_enfr_dev_set(data_dir)

  # Create vocabularies of the appropriate sizes.
  fr_vocab_path = os.path.join(data_dir, "vocab%d.fr" % fr_vocabulary_size)
  en_vocab_path = os.path.join(data_dir, "vocab%d.en" % en_vocabulary_size)
  create_vocabulary(fr_vocab_path, train_path + ".fr", fr_vocabulary_size, tokenizer)
  create_vocabulary(en_vocab_path, train_path + ".en", en_vocabulary_size, tokenizer)

  # Create token ids for the training data.
  fr_train_ids_path = train_path + (".ids%d.fr" % fr_vocabulary_size)
  en_train_ids_path = train_path + (".ids%d.en" % en_vocabulary_size)
  data_to_token_ids(train_path + ".fr", fr_train_ids_path, fr_vocab_path, tokenizer)
  data_to_token_ids(train_path + ".en", en_train_ids_path, en_vocab_path, tokenizer)

  # Create token ids for the development data.
  fr_dev_ids_path = dev_path + (".ids%d.fr" % fr_vocabulary_size)
  en_dev_ids_path = dev_path + (".ids%d.en" % en_vocabulary_size)
  data_to_token_ids(dev_path + ".fr", fr_dev_ids_path, fr_vocab_path, tokenizer)
  data_to_token_ids(dev_path + ".en", en_dev_ids_path, en_vocab_path, tokenizer)

  return (en_train_ids_path, fr_train_ids_path,
          en_dev_ids_path, fr_dev_ids_path,
          en_vocab_path, fr_vocab_path)

def get_training_data(config):
  if config['use_default_data']:
    """Train a en->fr translation model using WMT data."""
    logging.info("Preparing data in dir=%s" % config['data_dir'])
    src_train, trg_train, src_dev, trg_dev, _, _ = prepare_wmt_data(
      config['data_dir'], config['src_vocab_size'], config['trg_vocab_size'], tokenizer=None)
  elif config['save_npz']:
    # do not need data
    return None, None, None, None
  else:
    if config['train_src_idx'] != None and config['train_trg_idx'] != None and \
      config['dev_src_idx'] != None and config['dev_trg_idx'] != None:
        logging.info("Get indexed training and dev data")
        src_train, trg_train, src_dev, trg_dev = config['train_src_idx'], config['train_trg_idx'], \
                                                 config['dev_src_idx'], config['dev_trg_idx'] 
    elif config['train_src'] != None and config['train_trg'] != None and \
      config['dev_src'] != None and config['dev_trg'] != None:
        logging.info("Index tokenized training and dev data and write to dir=%s" % config['data_dir'])
        src_train, trg_train, src_dev, trg_dev, _, _ =  prepare_data(
          config['data_dir'], config['src_vocab_size'], config['trg_vocab_size'],
          config['train_src'], config['train_trg'], config['dev_src'], config['dev_trg'],
          config['src_lang'], config['trg_lang'])
    else:
      logging.error("You have to provide either tokenized or integer-mapped training and dev data usinig " \
        "--train_src, --train_trg, --dev_src, --dev_trg or --train_src_idx, --train_trg_idx, --dev_src_idx, --dev_trg_idx")
      exit(1)
    return src_train, trg_train, src_dev, trg_dev

def prepare_data(data_dir, src_vocabulary_size, trg_vocabulary_size,
                     train_src, train_trg, dev_src, dev_trg, src_lang, trg_lang):
  """Create vocabularies and index data, data assumed to be tokenized.
  Args:
    data_dir: directory in which the data will be stored.
    src_vocabulary_size: size of the source vocabulary to create and use.
    trg_vocabulary_size: size of the target vocabulary to create and use.
    train_src: Tokenized source training data
    train_trg: Tokenized target training data
    dev_src: Tokenized source dev data
    dev_trg: Tokenized target dev data
    src_lang: Source language
    trg_lang: Target language
  Returns:
    A tuple of 6 elements:
      (1) path to the token-ids for source training data-set,
      (2) path to the token-ids for target training data-set,
      (3) path to the token-ids for source development data-set,
      (4) path to the token-ids for target development data-set,
      (5) path to the source vocabulary file,
      (6) path to the target vocabulary file.
  """
  # Output paths
  train_path = os.path.join(data_dir, "train")
  dev_path = os.path.join(data_dir, "dev")

  # Create vocabularies of the appropriate sizes.
  src_vocab_path = os.path.join(data_dir, "vocab%d" % src_vocabulary_size + "." + src_lang)
  trg_vocab_path = os.path.join(data_dir, "vocab%d" % trg_vocabulary_size + "." + trg_lang)
  create_vocabulary(src_vocab_path, train_src, src_vocabulary_size)
  create_vocabulary(trg_vocab_path, train_trg, trg_vocabulary_size)

  # Create token ids for the training data.
  src_train_ids_path = train_path + (".ids%d" % src_vocabulary_size + "." + src_lang)
  trg_train_ids_path = train_path + (".ids%d" % trg_vocabulary_size + "." + trg_lang)
  data_to_token_ids(train_path + "." + src_lang, src_train_ids_path, src_vocab_path)
  data_to_token_ids(train_path + "." + trg_lang, trg_train_ids_path, trg_vocab_path)

  # Create token ids for the development data.
  src_dev_ids_path = dev_path + (".ids%d" % src_vocabulary_size + "." + src_lang)
  trg_dev_ids_path = dev_path + (".ids%d" % trg_vocabulary_size + "." + trg_lang)
  data_to_token_ids(dev_path + "." + src_lang, src_dev_ids_path, src_vocab_path)
  data_to_token_ids(dev_path + "." + trg_lang, trg_dev_ids_path, trg_vocab_path)

  return (src_train_ids_path, trg_train_ids_path,
          src_dev_ids_path, trg_dev_ids_path,
          src_vocab_path, trg_vocab_path)

def read_data(buckets, source_path, target_path, max_size=None, src_vcb_size=None, trg_vcb_size=None, add_src_eos=True):
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
  if add_src_eos:
    logging.info("Add EOS symbol to all source sentences")
  if src_vcb_size:
    logging.info("Replace OOV words with id={} for src_vocab_size={}".format(UNK_ID, src_vcb_size))
  if trg_vcb_size:
    logging.info("Replace OOV words with id={} for trg_vocab_size={}".format(UNK_ID, trg_vcb_size))

  data_set = [[] for _ in buckets]
  with tf.gfile.GFile(source_path, mode="r") as source_file:
    with tf.gfile.GFile(target_path, mode="r") as target_file:
      source, target = source_file.readline(), target_file.readline()
      counter = 0
      while source and target and (not max_size or counter < max_size):
        counter += 1
        if counter % 100000 == 0:
          logging.info("  reading data line %d" % counter)

        source_ids = [int(x) for x in source.split()]
        if add_src_eos:
          source_ids.append(EOS_ID)
        target_ids = [int(x) for x in target.split()]
        target_ids.append(EOS_ID)

        if src_vcb_size:
          # Replace source OOV words with unk (in case this has not been done on the source side)
          source_ids = [ wid if wid < src_vcb_size else UNK_ID for wid in source_ids ]

        if trg_vcb_size:
          # Replace target OOV words with unk (in case this has not been done on the target side)
          target_ids = [ wid if wid < trg_vcb_size else UNK_ID for wid in target_ids ]

        for bucket_id, (source_size, target_size) in enumerate(buckets):
          # Target will get additional GO symbol
          if len(source_ids) <= source_size and len(target_ids) < target_size:
            data_set[bucket_id].append([source_ids, target_ids])
            break # skips training example if it fits in no bucket
        source, target = source_file.readline(), target_file.readline()
  return data_set

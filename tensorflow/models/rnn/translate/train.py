"""Binary for training translation models based on tensorflow/models/rnn/translate/translate.py.

Running this program with --use_default_data will download the WMT corpus into
the directory specified as --data_dir and tokenize it in a very basic way,
and then start training a model saving checkpoints to --train_dir.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import time, datetime
import logging, pickle

import tensorflow as tf
from tensorflow.models.rnn.translate.utils import data_utils, train_utils, model_utils

tf.app.flags.DEFINE_string("config_file", None, "Pass options in a config file (will override conflicting command line settings)")

# Training settings
tf.app.flags.DEFINE_string("src_lang", "en", "Source language")
tf.app.flags.DEFINE_string("trg_lang", "de", "Target language")
tf.app.flags.DEFINE_string("train_dir", None, "Training directory.")
tf.app.flags.DEFINE_string("data_dir", "/tmp", "Data directory if data needs to be prepared")
tf.app.flags.DEFINE_boolean("use_default_data", False, "If set, download and prepare Gigaword data set instead of providing custom train/dev/test data. That data will be tokenized first.")
tf.app.flags.DEFINE_string("train_src", None, "Source side of training data")
tf.app.flags.DEFINE_string("train_src_idx", None, "Source side of training data (integer-mapped)")
tf.app.flags.DEFINE_string("train_trg", None, "Target side of training data")
tf.app.flags.DEFINE_string("train_trg_idx", None, "Target side of training data (integer-mapped)")
tf.app.flags.DEFINE_string("dev_src", None, "Source side of development data")
tf.app.flags.DEFINE_string("dev_src_idx", None, "Source side of development data (integer-mapped)")
tf.app.flags.DEFINE_string("dev_trg", None, "Target side of development data")
tf.app.flags.DEFINE_string("dev_trg_idx", None, "Target side of development data (integer-mapped)")
tf.app.flags.DEFINE_integer("max_sequence_length", 50, "Maximum length of source/target training sentences")
tf.app.flags.DEFINE_integer("max_train_data_size", 0, "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("max_train_batches", 0, "Limit on the number of training batches.")
tf.app.flags.DEFINE_integer("max_train_epochs", 0, "Limit on the number of training epochs.")
tf.app.flags.DEFINE_integer("max_epoch", 0, "Number of training epochs with original learning rate.")
tf.app.flags.DEFINE_boolean("train_sequential", False, "Shuffle training indices every epoch, then go through set sequentially")
tf.app.flags.DEFINE_integer("num_symm_buckets", 5, "Use x buckets of equal source/target length, with the largest bucket of length=50 (max_seq_len=50")
tf.app.flags.DEFINE_boolean("add_src_eos", False, "Add EOS symbol to all source sentences.")
tf.app.flags.DEFINE_boolean("no_pad_symbol", False, "Only use GO, EOS, UNK, set PAD=-1")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 200, "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_string("device", None, "Device to be used")
tf.app.flags.DEFINE_string("variable_prefix", "nmt", "Prefix to add to graph variable names")
tf.app.flags.DEFINE_integer("max_to_keep", 5, "Number of saved models to keep (set to 0 to keep all models)")
tf.app.flags.DEFINE_float("keep_prob", 1.0, "Probability of applying dropout to parameters")
tf.app.flags.DEFINE_boolean("fixed_random_seed", False, "If True, use a fixed random seed to make training reproducible (affects matrix initialization)")
tf.app.flags.DEFINE_float("init_scale", None, "Set the initial scale of the weights using tf.random_uniform_initializer")
tf.app.flags.DEFINE_boolean("shuffle_data", True, "If False, do not shuffle the training data to make training reproducible")
tf.app.flags.DEFINE_boolean("debug", True, "Add checks to make sure all examples in an epoch have been processed even if training was interrupted")
tf.app.flags.DEFINE_boolean("self_test", False, "Run a self-test if this is set to True.")

# Dev evaluation settings
tf.app.flags.DEFINE_integer("eval_frequency", 6, "How often performance should be evaluated on dev batches relative to checkpoints (default: evaluate every 6 checkpoints)")
tf.app.flags.DEFINE_boolean("eval_random", True, "If True, choose eval examples at random from each bucket, otherwise read sequentially")
tf.app.flags.DEFINE_integer("eval_size", 80, "The number of examples to evaluate in each bucket. If set to -1, evaluate on all examples in each bucket. This setting is only "
                            "used when random_eval=False.")
tf.app.flags.DEFINE_boolean("eval_bleu", False, "If True, decode dev set and measure BLEU instead of measuring perplexities")
tf.app.flags.DEFINE_integer("eval_bleu_size", 0, "The number of dev sentences to translate (all if set to 0)")
tf.app.flags.DEFINE_integer("eval_bleu_start", 10000, "Number of batches before starting BLEU evaluation on dev")

# Model configuration
tf.app.flags.DEFINE_integer("src_vocab_size", 40000, "Source vocabulary size.")
tf.app.flags.DEFINE_integer("trg_vocab_size", 40000, "Target vocabulary size.")
tf.app.flags.DEFINE_boolean("use_lstm", False, "Use LSTM cells instead of GRU cells")
tf.app.flags.DEFINE_integer("embedding_size", 620, "Size of the word embeddings.")
tf.app.flags.DEFINE_integer("hidden_size", 1000, "Size of the hidden model layer.")
tf.app.flags.DEFINE_integer("num_layers", 1, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("num_samples", 512, "Number of samples if using sampled softmax (0 to not use it).")
tf.app.flags.DEFINE_boolean("norm_digits", False, "Normalise all digits to 0s")
tf.app.flags.DEFINE_boolean("use_seqlen", True, "Use sequence length for encoder inputs.")
tf.app.flags.DEFINE_boolean("use_src_mask", True, "Use source mask over for decoder attentions.")
tf.app.flags.DEFINE_boolean("maxout_layer", False, "If > 0, use a maxout layer of given size and full softmax instead of sampled softmax")
tf.app.flags.DEFINE_string("encoder", "reverse", "Select encoder from 'reverse', 'bidirectional', 'bow'. The 'reverse' encoder is unidirectional and reverses the input "
                            "(default for tensorflow), the bidirectional encoder creates both forward and backward states and "
                            "concatenates them (like the Bahdanau model)")
tf.app.flags.DEFINE_boolean("init_backward", False, "When using the bidirectional encoder, initialise the hidden decoder state from the backward encoder state (default: forward).")                            
tf.app.flags.DEFINE_boolean("legacy", False, "Read legacy models with slightly different variable scopes")

# Extra model configuration for BOW model
tf.app.flags.DEFINE_boolean("bow_init_const", False, "Learn an initialisation matrix for the decoder instead of taking the average of source embeddings")
tf.app.flags.DEFINE_boolean("use_bow_mask", False, "Normalize decoder output layer over per-sentence BOW vocabulary")

# Optimization settings
tf.app.flags.DEFINE_string("opt_algorithm", "sgd", "Optimization algorithm: sgd, adagrad, adadelta")
tf.app.flags.DEFINE_float("learning_rate", 1.0, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99, "Learning rate decays by this much.")
tf.app.flags.DEFINE_boolean("adjust_lr", False, "Adjust learning rate independent of performance.")
tf.app.flags.DEFINE_float("max_gradient_norm", 1.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 80, "Batch size to use during training.")

# Rename model variables
tf.app.flags.DEFINE_bool("rename_variable_prefix", False, "Rename model variables with variable_prefix (assuming the model was saved with default prefix)")
tf.app.flags.DEFINE_string("model_path", None, "Path to trained model")
tf.app.flags.DEFINE_string("new_model_path", None, "Path to trained model with renamed variables")

# Model saving
tf.app.flags.DEFINE_string("filetype", "ckpt", "File type of saved model, will be set to 'npz' internally if save_npz=true")
tf.app.flags.DEFINE_boolean("save_npz", False, "Save model in npz format")

FLAGS = tf.app.flags.FLAGS

def train(config):
  logging.info("Train BOW model") if (config['encoder'] == "bow" and config['src_lang'] == config['trg_lang']) \
    else logging.info("Train NMT model")  
  
  # Get training data.
  src_train, trg_train, src_dev, trg_dev = data_utils.get_training_data(config)

  # Set device
  device = "/cpu:0"
  log_device_placement = False
  allow_soft_placement = True
  if config['device']:
    device = '/'+config['device']
  logging.info("Use device %s" % device)

  with tf.Session(config=tf.ConfigProto(allow_soft_placement=allow_soft_placement, log_device_placement=log_device_placement)) as session, tf.device(device):
    # Create model
    if config['fixed_random_seed']:
      tf.set_random_seed(1234)
    model = model_utils.create_model(session, config, forward_only=False)

    if config['save_npz']:
      config['filetype'] = 'npz'
      model_utils.save_model(session, config, model, epoch=-1)
      logging.info("Saved model as npz file.")
      exit(0)

    # Read data into buckets and prepare buckets for training
    logging.info ("Reading development and training data (limit: %d)." % config['max_train_data_size'])
    train_set = data_utils.read_data(model_utils._buckets, src_train, trg_train, config['max_train_data_size'],
                                     src_vcb_size=config['src_vocab_size'],
                                     trg_vcb_size=config['trg_vocab_size'],
                                     add_src_eos=config['add_src_eos'])
    dev_set = data_utils.read_data(model_utils._buckets, src_dev, trg_dev,
                                   src_vcb_size=config['src_vocab_size'],
                                   trg_vcb_size=config['trg_vocab_size'],
                                   add_src_eos=config['add_src_eos'])
    tmpfile = config['train_dir']+"/tmp_idx.pkl"
    tmpfile_bookk = config['train_dir']+"/tmp_bookk.pkl"
    train_buckets_scale, train_idx_map, bucket_offset_pairs, train_size, num_train_batches, bookk, epoch = \
      train_utils.prepare_buckets(model,
                                  train_set,
                                  tmpfile,
                                  tmpfile_bookk if config['debug'] else None,
                                  config['train_sequential'],
                                  config['steps_per_checkpoint'],
                                  config['shuffle_data'])
    # This is the training loop.
    step_time, loss = 0.0, 0.0
    current_step = model.global_step.eval()
    previous_losses = [] # used for updating learning rate (train loss)
    current_eval_ppxs = [] # used for model saving
    current_bleu = 0 # used for model saving
    current_batch_idx = None
    while True:
      current_batch_idx = model.global_step.eval() % num_train_batches
      if current_batch_idx == 0:
        # New epoch
        epoch = int(model.global_step.eval() / num_train_batches) + 1
        logging.info("Epoch=%i" % epoch)

        # Shuffle train variables and save result
        if config['train_sequential']:
          if config['shuffle_data']:
            logging.info("Shuffle train idx maps and batch pointers")
            for b in xrange(len(model_utils._buckets)):
              random.shuffle(train_idx_map[b]) # shuffle training idx map for each bucket
            random.shuffle(bucket_offset_pairs) # shuffle the bucket_id, offset pairs

          if os.path.isfile(tmpfile):
            os.rename(tmpfile, tmpfile+".old")
          with open(tmpfile, "wb") as f:
            logging.info("Epoch %i, save training example permutation to path=%s" % (epoch,tmpfile))
            pickle.dump((train_idx_map, bucket_offset_pairs, epoch), f, pickle.HIGHEST_PROTOCOL)

        # Debugging: check if all training examples have been processed in the past epoch
        if config['debug']:
          if epoch > 1 and bookk is not None:
            lengths = [ len(bookk[b].keys()) for b in bookk.keys() ]
            logging.info("After epoch %i: Total examples=%i, processed examples=%i" % (epoch-1, train_size, sum(lengths)))
            #assert train_size == sum(lengths), "ERROR: training set has not been fully processed"
            if train_size != sum(lengths):
              logging.error("Training set has not been fully processed")
            bookk.clear()

        # Adjust learning rate independent of performance
        if config['adjust_lr'] and config['max_epoch'] > 0:
          lr_decay_factor = config['learning_rate_decay_factor'] ** max(epoch - config['max_epoch'], 0.0)
          session.run(model.learning_rate.assign(config['learning_rate'] * lr_decay_factor))
          logging.info("Learning rate={}".format(model.learning_rate.eval()))
      #endif new epoch

      # Get a bucket_id or bucket_id + batch_ptr (if train_sequential)
      bucket_id, batch_ptr = train_utils.get_bucket_or_batch_ptr(model, train_buckets_scale, train_idx_map,
                                                                 bucket_offset_pairs, current_batch_idx, current_step,
                                                                 config['train_sequential'], config['steps_per_checkpoint'])

      # Make a step on a random sequential batch (processing one batch is a global step)
      start_time = time.time()
      encoder_inputs, decoder_inputs, target_weights, sequence_length, src_mask, bow_mask = model.get_batch(
        train_set, bucket_id, config['encoder'], batch_ptr=batch_ptr if config['train_sequential'] else None,
        bookk=bookk if config['debug'] else None)

      _, step_loss, _ = model.step(session, encoder_inputs, decoder_inputs,
                                   target_weights, bucket_id, False,
                                   sequence_length, src_mask, bow_mask)

      step_time += (time.time() - start_time) / config['steps_per_checkpoint']
      loss += step_loss / config['steps_per_checkpoint']
      current_step += 1

      # Once in a while, we save a checkpoint, print statistics, and run evals.
      if current_step % config['steps_per_checkpoint'] == 0:
        train_utils.print_stats(model, loss, step_time, config['opt_algorithm'])
        model_utils.save_model(session, config, model, epoch)

        # Debugging: save book keeping dict
        if config['debug']:
          if os.path.isfile(tmpfile_bookk):
            os.rename(tmpfile_bookk, tmpfile_bookk+".old")
          with open(tmpfile_bookk, "wb") as f:
            logging.info("Epoch %i, save book keeping variable to path=%s" % (epoch, tmpfile_bookk))
            pickle.dump(bookk, f, pickle.HIGHEST_PROTOCOL)

        # Decrease learning rate if no improvement was seen over last 3 times.
        if len(previous_losses) > 2 and loss > max(previous_losses[-3:]) and config['opt_algorithm'] == "sgd":
          session.run(model.learning_rate_decay_op)
          logging.info("Decrease learning rate to {}".format(model.learning_rate.eval()))
        previous_losses.append(loss)

        # Zero timer and loss.
        step_time, loss = 0.0, 0.0

        # Run evals on development set and print their perplexity.
        if current_step % (config['steps_per_checkpoint'] * config['eval_frequency']) == 0:
          if config['eval_bleu']:
            if model.global_step.eval() >= config['eval_bleu_start']:
              current_bleu = train_utils.decode_dev(config, model, current_bleu)
            else:
              logging.info("Waiting until global step %i for BLEU evaluation on dev" % config['eval_bleu_start'])
          else:
            current_eval_ppxs = train_utils.run_eval(config, session, model, dev_set, current_eval_ppxs)
        logging.info("Time: {}".format(datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')))
      #endif save checkpoint

      if (config['max_train_batches'] > 0 and model.global_step.eval() >= config['max_train_batches']) or \
        (config['max_train_epochs'] > 0 and epoch == config['max_train_epochs'] and current_batch_idx + 1 == num_train_batches):
          if current_step % config['steps_per_checkpoint'] != 0:
            model_utils.save_model(session, config, model, epoch)
          logging.info("Stopped training after %i epochs" % epoch)
          logging.info("Time: {}".format(datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')))
          break

def self_test():
  """Test the translation model."""
  import random
  from tensorflow.models.rnn.translate.seq2seq import seq2seq_model
  with tf.Session() as sess:
    print("Self-test for neural translation model.")
    # Create model with vocabularies of 10, 2 small buckets, 2 layers of 32.
    model = seq2seq_model.Seq2SeqModel(10, 10, [(3, 3), (6, 6)], 32, 32, 2,
                                       5.0, 32, 0.3, 0.99, num_samples=8)
    sess.run(tf.initialize_all_variables())

    # Fake data set for both the (3, 3) and (6, 6) bucket.
    data_set = ([([1, 1], [2, 2]), ([3, 3], [4]), ([5], [6])],
                [([1, 1, 1, 1, 1], [2, 2, 2, 2, 2]), ([3, 3, 3], [5, 6])])
    for _ in xrange(5):  # Train the fake model for 5 steps.
      bucket_id = random.choice([0, 1])
      encoder_inputs, decoder_inputs, target_weights, _, _, _ = model.get_batch(
          data_set, bucket_id)
      print ("Make training step..")
      model.step(sess, encoder_inputs, decoder_inputs, target_weights,
                 bucket_id, False)
    print ("Done.")

def main(_):
  if FLAGS.self_test:
    self_test()
  else:
    config = model_utils.process_args(FLAGS, train=False if FLAGS.rename_variable_prefix \
                                                         else True)
    if config['rename_variable_prefix']:
      model_utils.rename_variable_prefix(config)
    else:
      train(config)

if __name__ == "__main__":
  logging.getLogger().setLevel(logging.INFO)
  logging.info("Start: {}".format(datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')))
  tf.app.run()
  logging.info("End: {}".format(datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')))

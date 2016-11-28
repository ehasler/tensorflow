"""
Original approach:
- training data is organized in buckets
- at each step, select a random bucket id
- then repeatedly select instances from that bucket until the batch is full

Sequential approach:
- training data is organized in buckets
- create a list of (bucket_id, offset) tuples and walk through that list sequentially ==> bucket_offset_pairs
- at each step, we pick the bucket_id from the current tuple and fill the batch starting from the offset in sequential order until it is full: data[bucket_id][idx_map[offset]], data[bucket_id][idx_map[offset+1]], ..
- at the beginning of every epoch, the buckets are shuffled: this is done implicitly by generating a mapping from integers to indices within each bucket ==> train_ixd_map

--> the original data is unchanged and we only need the bucket_offset_pairs and the train_idx_map

For debugging: 
Tick off which training examples we have processed: bookk[bucket_id][idx_map[train_idx]] = 1
"""

import os
import numpy as np
import math
import pickle
import shutil
from collections import defaultdict
import logging

from tensorflow.models.rnn.translate.utils import model_utils
import tensorflow as tf

def prepare_buckets(model, train_set, tmpfile, tmpfile_bookk, train_sequential, steps_per_checkpt, shuffle_data):
    # Compute buckets sizes
    buckets = model_utils._buckets
    train_bucket_sizes = [len(train_set[b]) for b in xrange(len(buckets))]
    train_size = float(sum(train_bucket_sizes))
    logging.info ("Training bucket sizes: {}".format(train_bucket_sizes))

    # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
    # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
    # the size if i-th training bucket, as used later.
    train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_size
                           for i in xrange(len(train_bucket_sizes))]     

    if train_sequential:
      train_size_old = train_size
      train_idx_map, bucket_offset_pairs, train_size, num_train_batches = prepare_sequential(train_set, model.batch_size, shuffle_data)
      logging.info("Total size of training set adjusted from %i to %i" % (train_size_old, train_size))
    else:
      logging.info ("Total size of training set=%i" % train_size)

    bookk = defaultdict(dict)
    epoch = 1
    # Try to restore saved shuffled train variables
    if train_sequential and model.global_step.eval() >= steps_per_checkpt:
      if tf.gfile.Exists(tmpfile):
        logging.info("Restore training example permutation from %s" % tmpfile)
        with open(tmpfile, "rb") as f:
          train_idx_map, bucket_offset_pairs, epoch = pickle.load(f)
        if tmpfile_bookk is not None:
          logging.info("Restore book keeping variable from %s" % tmpfile_bookk)
          with open(tmpfile_bookk, "rb") as f:
            bookk = pickle.load(f)
            logging.info("Training examples processed so far: %i" % sum([ len(bookk[b].keys()) for b in bookk.keys() ]))
      else:
        logging.info("No file with training example permutation available, using new permutation.")
    return train_buckets_scale, train_idx_map, bucket_offset_pairs, train_size, num_train_batches, bookk, epoch

def prepare_sequential(train_set, batch_size, shuffle_data):
  # Create a list of (bucket_id, off_set) tuples and walk through it sequentially,
  # shuffling the buckets every epoch
  buckets = model_utils._buckets
  bucket_offset_pairs = []
  for b in xrange(len(buckets)):
    for idx in xrange(len(train_set[b])):
      if idx % batch_size == 0:
        bucket_offset_pairs.append((b, idx))

    # Make sure every bucket has num_items % batch_size == 0 by adding random samples
    if len(train_set[b]) % batch_size > 0:
      num_extra = batch_size - (len(train_set[b]) % batch_size)
      if shuffle_data:
        samples = [ int(r * 10000) for r in np.random.random_sample(num_extra) ]
        for s in samples:
          while s >= len(train_set[b]):
            s = int(s/10)
          train_set[b].append(train_set[b][s])
      else:
        # For reproducibility (given the serialized data), just append the first n examples from a given bucket to its end
        for i in xrange(num_extra):
          train_set[b].append(train_set[b][i])
      assert len(train_set[b]) % batch_size == 0, "len(train_set[b])=%i mod batch_size=%i != 0" % (len(train_set[b]), batch_size)

  # For each bucket, create a list of indices which we can shuffle and therefore use as a mapping instead of shuffling the data
  train_idx_map = [ [] for b in xrange(len(buckets))]
  for b in xrange(len(buckets)):
    for idx in xrange(len(train_set[b])):
      train_idx_map[b].append(idx)
    assert len(train_idx_map[b]) == len(train_set[b]), "Inconsistent train idx map for bucket %i" % b

  train_bucket_sizes = [len(train_set[b]) for b in xrange(len(buckets))]
  train_size = float(sum(train_bucket_sizes))
  num_train_batches = int(train_size / batch_size)
  num_batch_pointers = len(bucket_offset_pairs)
  assert num_batch_pointers == num_train_batches, "Inconsistent number of batch pointers: %i != %i" % (num_batch_pointers, num_train_batches)
  logging.info("Total number of training batches=%i with batch size=%i" % (num_train_batches, batch_size))
  return train_idx_map, bucket_offset_pairs, train_size, num_train_batches

def get_bucket_or_batch_ptr(model, train_buckets_scale, train_idx_map, bucket_offset_pairs, current_batch_idx, current_step, train_sequential, steps_per_checkpt):
  if train_sequential:
    # Get next sequential batch pointer
    return get_batch_ptr(model, train_idx_map, bucket_offset_pairs, current_batch_idx, current_step, steps_per_checkpt)
  else:
    # Choose a bucket according to data distribution. We pick a random number
    # in [0, 1] and use the corresponding interval in train_buckets_scale.
    random_number_01 = np.random.random_sample()
    bucket_id = min([i for i in xrange(len(train_buckets_scale))
                   if train_buckets_scale[i] > random_number_01])
    logging.debug("bucket_id=%d" % bucket_id)
    return bucket_id, None

def get_batch_ptr(model, train_idx_map, bucket_offset_pairs, current_batch_idx, current_step, steps_per_checkpt):
  # bucket_offset_pair holds a bucket_id and train_idx
  # train_idx is the offset for the batch in the given bucket: all subsequent indices belong to the batch as well
  bucket_offset_pair = bucket_offset_pairs[current_batch_idx]
  bucket_id = bucket_offset_pair[0]
  train_offset = bucket_offset_pair[1]
  # idx_map is used to map the train indices to randomly assigned indices in the same bucket
  idx_map = train_idx_map[bucket_id]

  # This is for debugging only: make sure the order is preserved after reloading the model
  global_step = model.global_step.eval()+1
  if global_step % 100 == 0:
    logging.info("Global step=%i" % global_step)
  if current_batch_idx+2 < len(bucket_offset_pairs) and (current_step+1) % steps_per_checkpt == 0:
    bucket_offset_pair_1 = bucket_offset_pairs[current_batch_idx+1]
    bucket_offset_pair_2 = bucket_offset_pairs[current_batch_idx+2]
    idx_map_1 = train_idx_map[bucket_offset_pair_1[0]]
    idx_map_2 = train_idx_map[bucket_offset_pair_2[0]]
    logging.debug("Global step={}, current batch idx={} bucket_id={}, offset={}-->{}, next two batch ptrs={},{}, {},{}" .format(global_step, current_batch_idx, \
                  bucket_id, train_offset, idx_map[train_offset], \
                  bucket_offset_pair_1, idx_map_1[bucket_offset_pair_1[1]], \
                  bucket_offset_pair_2, idx_map_2[bucket_offset_pair_2[1]] ))
  else:
    logging.debug("Global step={}, current batch idx={} bucket_id={}, offset={}-->{}".format(global_step, current_batch_idx, bucket_id, train_offset, idx_map[train_offset] ))
  
  batch_ptr = { "offset": train_offset, "idx_map": idx_map }
  return bucket_id, batch_ptr

def print_stats(model, loss, step_time, opt_algorithm):
  # Print statistics for the previous epoch.
  perplexity = math.exp(loss) if loss < 300 else float('inf')
  if opt_algorithm == "sgd":
    logging.info("global step %d learning rate %.4f step-time %.2f perplexity "
           "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                     step_time, perplexity))
  else:
    logging.info("global step %d step-time %.2f perplexity "
           "%.2f" % (model.global_step.eval(),
                     step_time, perplexity))

def save_checkpoint(session, model, train_dir, epoch):
 checkpoint_path = os.path.join(train_dir, "translate.ckpt")
 logging.info("Epoch %i, save model to path=%s after global step=%d" % (epoch, checkpoint_path, model.global_step.eval()))
 model.epoch = epoch
 model.saver.save(session, checkpoint_path, global_step=model.global_step)

def run_eval(config, session, model, dev_set, current_eval_ppxs):
  logging.info("Run eval on development set")
  buckets = model_utils._buckets
  eval_ppxs = []

  if not config['eval_random']:
    dev_bucket_sizes = [len(dev_set[b]) for b in xrange(len(buckets))]
    logging.info ("Dev bucket sizes: {}".format(dev_bucket_sizes))

  for bucket_id in xrange(len(buckets)):
    if len(dev_set[bucket_id]) == 0:
      logging.info("  eval: empty bucket %d" % (bucket_id))
      continue

    if config['eval_random']:
      encoder_inputs, decoder_inputs, target_weights, sequence_length, src_mask, bow_mask = model.get_batch(
        dev_set, bucket_id, config['encoder'])

      _, eval_loss, _ = model.step(session, encoder_inputs, decoder_inputs,
                                   target_weights, bucket_id, True,
                                   sequence_length, src_mask, bow_mask)

      eval_ppx = math.exp(eval_loss) if eval_loss < 300 else float('inf')
      logging.info("  eval: global step %d bucket %d perplexity %.2f" % (model.global_step.eval(), bucket_id, eval_ppx))
    else:
      eval_set_size = min(len(dev_set[bucket_id]), config['eval_size']) if config['eval_size'] >= 0 else len(dev_set[bucket_id])
      iters = int(math.ceil(eval_set_size / model.batch_size))
      loss = 0
      # If eval_set_size > batch_size, run several dev steps, then average losses
      for it in xrange(iters):
        offset = it * model.batch_size
        idx_map = dict()
        for i in xrange(offset, offset+model.batch_size):
          idx_map[i] = i # identity mapping
        batch_ptr = { "offset": offset, "idx_map": idx_map }
        encoder_inputs, decoder_inputs, target_weights, sequence_length, src_mask, bow_mask = model.get_batch(
          dev_set, bucket_id, config['encoder'], batch_ptr=batch_ptr)

        _, eval_loss, _ = model.step(session, encoder_inputs, decoder_inputs,
                                     target_weights, bucket_id, True,
                                     sequence_length, src_mask, bow_mask)
        loss += eval_loss

      avg_loss = loss / iters
      eval_ppx = math.exp(avg_loss) if avg_loss < 300 else float('inf')
      logging.info("  eval: global step %d bucket %d eval_size %d perplexity %.2f" % (model.global_step.eval(), bucket_id, eval_set_size, eval_ppx))

    eval_ppxs.append(eval_ppx)

  # If the current model improves over the results of the previous dev eval, overwrite the dev_ppx model
  if len(current_eval_ppxs) > 0:
    num_improved = 0
    for b in xrange(len(current_eval_ppxs)):
      if eval_ppxs[b] < current_eval_ppxs[b]:
        num_improved += 1

    checkpoint_path = os.path.join(config['train_dir'], "translate.ckpt-")
    current_model = checkpoint_path + str(model.global_step.eval())
    dev_ppx_model = checkpoint_path + "dev_ppx"
    if num_improved == len(current_eval_ppxs):
      shutil.copy(current_model, dev_ppx_model)
      shutil.copy(current_model+".meta", dev_ppx_model+".meta")
      logging.info("Model %s achieves lower dev perplexity, updating %s" % (current_model, dev_ppx_model))
      return eval_ppxs
    else:
      logging.info("Model %s does not achieve lower dev perplexity, not updating %s" % (current_model, dev_ppx_model))
      return current_eval_ppxs
  else:
    return eval_ppxs

def decode_dev(config, model, current_bleu):
  # Greedily decode dev set
  inp = config['dev_src_idx']
  out = os.path.join(config['train_dir'], "dev.out")
  ref = config['dev_trg_idx']
  g2 = tf.Graph() # use a separate graph to avoid variable collision when loading model for decoding
  with g2.as_default() as g:
    from tensorflow.models.rnn.translate.decode import decode
    decode(config, input=inp, output=out, max_sentences=config['eval_bleu_size'])
  bleu = eval_set(out, ref)

  # If the current model improves over the results of the previous dev eval, overwrite the dev_bleu model
  checkpoint_path = os.path.join(config['train_dir'], "translate.ckpt-")
  current_model = checkpoint_path + str(model.global_step.eval())
  dev_bleu_model = checkpoint_path + "dev_bleu"
  if bleu > current_bleu:
    current_bleu = bleu
    shutil.copy(current_model, dev_bleu_model)
    shutil.copy(current_model+".meta", dev_bleu_model+".meta")
    logging.info("Model %s achieves new best BLEU=%.2f, updating %s" % (current_model, bleu, dev_bleu_model))
    return bleu
  else:
    logging.info("Model %s does not achieve higher BLEU, not updating %s" % (current_model, dev_bleu_model))
    return current_bleu

def eval_set(out, ref):
  # multi-bleu.pl [-lc] reference < hypothesis
  import subprocess
  cat = subprocess.Popen(("cat", out), stdout=subprocess.PIPE)
  try:
    multibleu = subprocess.check_output(("/home/mifs/ech57/code/tensorflow/tensorflow/models/rnn/translate/eval/multi-bleu.perl", "-lc",
                                       ref), stdin=cat.stdout)
    logging.info("{}".format(multibleu))
    import re
    m = re.match("BLEU = ([\d.]+),", multibleu)
    return float(m.group(1))
  except Exception, e:
    logging.info("Multi-bleu error: {}".format(e))
    return 0.0
  

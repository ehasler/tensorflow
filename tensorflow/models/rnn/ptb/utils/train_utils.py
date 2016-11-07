import os
import time, datetime
import pickle
import logging
import numpy as np
import tensorflow as tf

from tensorflow.models.rnn.ptb import reader

def run_epoch(session, m, data, eval_op, train_dir, steps_per_ckpt, train=False, 
              start_idx=0, start_state=None, tmpfile=None, m_valid=None, valid_data=None, epoch=None):
  """Runs the model on the given data."""
  epoch_size = ((len(data) // m.batch_size) - 1) // m.num_steps
  if train:
    logging.info("Training data_size=%s batch_size=%s epoch_size=%s start_idx=%i global_step=%s" % \
      (len(data), m.batch_size, epoch_size, start_idx, m.global_step.eval()))
  else:
    logging.info("Val/Test data_size=%s batch_size=%s epoch_size=%s start_idx=%i" % (len(data), m.batch_size, epoch_size, start_idx))
  start_time = time.time()
  costs = 0.0
  iters = 0
  if start_idx == 0:
    state = m.initial_state.eval()
  else:
    state = start_state
  for step, (x, y) in enumerate(reader.ptb_iterator(data, m.batch_size,
                                                    m.num_steps, start_idx), start=1+start_idx):
    if train:
      logging.debug("Epoch=%i start_idx=%i step=%i global_step=%i " % (epoch, start_idx, step, m.global_step.eval()))

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

    if train and step % steps_per_ckpt == 0:
      logging.info("Time: {}".format(datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')))      
      logging.info("%.3f perplexity: %.3f speed: %.0f wps" %
            (step * 1.0 / epoch_size, np.exp(costs / iters),
             iters * m.batch_size / (time.time() - start_time)))
      checkpoint_path = os.path.join(train_dir, "rnn.ckpt")
      finished_idx = step - 1
      logging.info("Save model to path=%s after training_idx=%s and global_step=%s" % (checkpoint_path, finished_idx, m.global_step.eval()))
      m.saver.save(session, checkpoint_path, global_step=m.global_step)

      # Save train variables
      with open(tmpfile, "wb") as f:
        # Training idx = step - 1, so we want to resume from idx = step
        # If we had already restarted from start_idx, this gives the offset
        training_idx = step
        logging.info("Save epoch=%i and training_idx=%i and state to resume from" % (epoch, training_idx))
        pickle.dump((epoch, training_idx, state), f, pickle.HIGHEST_PROTOCOL)
      
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
    else:
      cost, state, _ = session.run([m.cost, m.final_state, eval_op],
                                 {m.input_data: x,
                                  m.targets: y,
                                  m.initial_state: state})
      costs += cost
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
  logging.info("Test log probability={}".format(logp))
  logging.info("Test PPL: %f", np.exp(-logp/wordcn))
  return logp
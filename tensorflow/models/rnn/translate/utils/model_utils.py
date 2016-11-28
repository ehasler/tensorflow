from tensorflow.models.rnn.translate.utils import data_utils
from tensorflow.models.rnn.translate.seq2seq import seq2seq_model, tf_seq2seq
from tensorflow.python.platform import gfile
import tensorflow as tf
import os,re
import logging

# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
# Default buckets:
_buckets = [(5, 10), (10, 15), (20, 25), (40, 50)] 

def read_config(config_file, config):
  def is_bool(v):
    return v.lower() in ('true', 't', 'false', 'f')

  def str2bool(v):
    return v.lower() in ('true', 't')  
  
  if not config_file or not os.path.isfile(config_file):
    raise ValueError("Cannot load config file %s" % config_file)

  logging.info("Settings from tensorflow config file:")    
  with open(config_file) as f:
    for line in f:
      key,value = line.strip().split(": ")
      if value == "None":
        value = None
      elif is_bool(value):
        value = str2bool(value)      
      elif re.match("^\d+$", value):
        value = int(value)
      elif re.match("^[\d\.]+$", value):
        value = float(value)
      config[key] = value
      logging.info("{}: {}".format(key, value))

def process_args(FLAGS, train=True, greedy_decoder=False):
  config = dict()

  # First read command line flags
  for key,value in FLAGS.__dict__['__flags'].iteritems():
    config[key] = value

  # Then read config file if available
  if config['config_file']:
    read_config(config['config_file'], config)

  if train and not config['train_dir']:
     raise ValueError("Must set --train_dir")

  # Process specific args
  if config['opt_algorithm'] not in [ "sgd", "adagrad", "adadelta" ]:
    raise Exception("Unknown optimization algorithm: {}".format(config['opt_algorithm']))
  if config['num_symm_buckets'] > 0:
    global _buckets
    _buckets = make_buckets(config['num_symm_buckets'], config['max_sequence_length'],
                            config['add_src_eos'], train, greedy_decoder)

  if config['no_pad_symbol']:
    data_utils.no_pad_symbol()
    logging.info("UNK_ID=%d" % data_utils.UNK_ID)
    logging.info("PAD_ID=%d" % data_utils.PAD_ID)
    
  return config

def make_buckets(num_buckets, max_seq_len=50, add_src_eos=True, train=True, greedy_decoder=False):
  # Bucket length: +1 for EOS, +1 for GO symbol
  src_offset = 0 if not add_src_eos else 1
  if train:
    # Buckets for training
    buckets = [ (int(max_seq_len/num_buckets)*i + src_offset, int(max_seq_len/num_buckets)*i + 2) for i in range(1,num_buckets+1) ]
  else:
    if greedy_decoder:
      # Buckets for decoding with training graph (greedy decoder)
      buckets = [ (int(max_seq_len/num_buckets)*i + src_offset, int(max_seq_len/num_buckets)*i + 5) for i in range(1,num_buckets+1) ]
    else:
      # Buckets for decoding with single-step decoding graph: input length=1 on the target side)
      buckets = [ (int(max_seq_len/num_buckets)*i + src_offset, 1) for i in range(1,num_buckets+1) ]

  logging.info("Use buckets={}".format(buckets))
  return buckets

def make_bucket(src_length, greedy_decoder=False):
  if greedy_decoder:
    # Additional bucket for decoding with training graph
    return (src_length, src_length + 5)
  else:
    # Additional bucket for decoding with single-step decoding graph: input length=1 on the target side)
    return (src_length, 1)

def get_model_path(config):
  if 'model_path' in config and config['model_path']:
    return config['model_path']
  elif 'train_dir' in config and config['train_dir']:
    ckpt = tf.train.get_checkpoint_state(config['train_dir'])
    if ckpt:
      return ckpt.model_checkpoint_path
    else:
      return None
  else:
    logging.error("You have to specify either --train_dir or --model_path")
    exit(1)

def get_initializer(config):
  if 'init_scale' in config and config['init_scale']:
    logging.info("Using initializer tf.random_uniform_initializer(-{},{})".format(config['init_scale'],config['init_scale']))
    initializer = tf.random_uniform_initializer(-config['init_scale'], config['init_scale'])
    return initializer
  return None

def create_model(session, config, forward_only, rename_variable_prefix=None, buckets=None):
  """Create or load translation model for training or greedy decoding"""
  if not forward_only:
    logging.info("Creating %d layers of %d units, encoder=%s." % (config['num_layers'], config['hidden_size'], config['encoder']))
  if not buckets:
    buckets = _buckets
  model = get_Seq2SeqModel(config, buckets, forward_only, rename_variable_prefix)

  model_path = get_model_path(config)
  if model_path and tf.gfile.Exists(model_path):
    logging.info("Reading model parameters from %s" % model_path)
    model.saver.restore(session, model_path)
  else:
    logging.info("Created model with fresh parameters.")
    session.run(tf.initialize_all_variables())
  return model

def load_model(session, config):
  """Load translation model with single-step graph for decoding"""
  buckets = make_buckets(config['num_symm_buckets'], config['max_sequence_length'], config['add_src_eos'], train=False)
  model = get_singlestep_Seq2SeqModel(config, buckets)
  training_graph = model.create_training_graph() # Needed for loading variables
  encoding_graph = model.create_encoding_graph()
  single_step_decoding_graph = model.create_single_step_decoding_graph(encoding_graph.outputs)

  model_path = get_model_path(config)
  if model_path and gfile.Exists(model_path):
    logging.info("Reading model parameters from %s" % model_path)
    training_graph.saver.restore(session, model_path)
  else:
    logging.fatal("Could not load model parameters from %s" % model_path)

  return model, training_graph, encoding_graph, single_step_decoding_graph, buckets

def get_Seq2SeqModel(config, buckets, forward_only, rename_variable_prefix=None):
  return seq2seq_model.Seq2SeqModel(
      config['src_vocab_size'], config['trg_vocab_size'], buckets,
      config['embedding_size'], config['hidden_size'],
      config['num_layers'], config['max_gradient_norm'], config['batch_size'],
      config['learning_rate'], config['learning_rate_decay_factor'], use_lstm=config['use_lstm'],
      num_samples=config['num_samples'], forward_only=forward_only,
      opt_algorithm=config['opt_algorithm'], encoder=config['encoder'],
      use_sequence_length=config['use_seqlen'], use_src_mask=config['use_src_mask'], 
      maxout_layer=config['maxout_layer'], init_backward=config['init_backward'], 
      no_pad_symbol=config['no_pad_symbol'], variable_prefix=config['variable_prefix'], 
      init_const=config['bow_init_const'], use_bow_mask=config['use_bow_mask'],
      max_to_keep=config['max_to_keep'],
      keep_prob=config['keep_prob'],
      initializer=get_initializer(config),
      rename_variable_prefix=rename_variable_prefix)

def get_singlestep_Seq2SeqModel(config, buckets):
  return tf_seq2seq.TFSeq2SeqEngine(
      config['src_vocab_size'], config['trg_vocab_size'], buckets,
      config['embedding_size'], config['hidden_size'],
      config['num_layers'], config['max_gradient_norm'], 1, # Batch size is 1
      config['learning_rate'], config['learning_rate_decay_factor'], use_lstm=config['use_lstm'],
      num_samples=config['num_samples'], forward_only=True,
      opt_algorithm=config['opt_algorithm'], encoder=config['encoder'],
      use_sequence_length=config['use_seqlen'], use_src_mask=config['use_src_mask'],
      maxout_layer=config['maxout_layer'], init_backward=config['init_backward'],
      no_pad_symbol=config['no_pad_symbol'],
      variable_prefix=config['variable_prefix'],
      init_const=config['bow_init_const'], use_bow_mask=config['use_bow_mask'],
      initializer=get_initializer(config))

def rename_variable_prefix(config):
  logging.info("Rename model variables with prefix %s" % config['variable_prefix'])
  with tf.Session() as session:
    # Create model and restore variable
    logging.info("Creating %d layers of %d units, encoder=%s." % (config['num_layers'], config['hidden_size'], config['encoder']))
    rename_variable_prefix=config['variable_prefix']
    config['variable_prefix'] = None
    model, ckpt = create_model(session, config, forward_only=False, rename_variable_prefix=rename_variable_prefix)

    # Save model with new variable names
    checkpoint_path = os.path.join(config['train_dir']+"_mult", os.path.basename(ckpt.model_checkpoint_path))
    print("Save model to path=%s using saver_prefix" % checkpoint_path)
    model.saver_prefix.save(session, checkpoint_path)

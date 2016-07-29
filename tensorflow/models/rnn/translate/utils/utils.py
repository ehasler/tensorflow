'''TODO: this has been copied from gnmt/cam/gnmt/, could just import it from there!
Common basic functionality
'''
#from scipy.misc import logsumexp
import numpy
import operator

'''
Reserved indices
'''

PAD_ID = None
GO_ID = 1
EOS_ID = 2
UNK_ID = 0
NOTAPPLICABLE_ID = 3 # This is for factored models only


def switch_to_old_indexing():
    global PAD_ID
    global GO_ID
    global EOS_ID
    global UNK_ID
    PAD_ID = 0
    GO_ID = 1
    EOS_ID = 2
    UNK_ID = 3

'''
Log summation
'''


def log_sum_tropical_semiring(vals):
    return max(vals)

def log_sum_log_semiring(vals):
    return logsumexp(numpy.asarray([val for val in vals]))
#log_sum = log_sum_log_semiring
log_sum = log_sum_log_semiring


'''
Argmax
'''

def argmax_n(arr, n):
    ''' Get indices of n maximum entries in arr. The array can
    be a dictionary. The returned index set is not guaranteed
    to be sorted '''
    if isinstance(arr, dict):
        return sorted(arr, key=arr.get, reverse=True)[:n]
    else:
        return numpy.argpartition(arr, -n)[-n:]
    
def argmax(arr):
    ''' Get index of maximum entry in arr. The array can
    be a dictionary. '''
    if isinstance(arr, dict):
        return max(arr.iteritems(), key=operator.itemgetter(1))[0]
    else:
        return numpy.argmax(arr)
    

'''
Functions for common access to numpy arrays, lists, and dicts
'''
    
def common_viewkeys(obj):
    ''' See http://stackoverflow.com/questions/12325608/iterate-over-a-dict-or-list-in-python '''
    if isinstance(obj, dict):
        return obj.viewkeys()
    else:
        return xrange(len(obj))

def common_iterable(obj):
    ''' See http://stackoverflow.com/questions/12325608/iterate-over-a-dict-or-list-in-python '''
    if isinstance(obj, dict):
        for key, value in obj.iteritems():
            yield key, value
    else:
        for index, value in enumerate(obj):
            yield index, value

def common_get(obj, key, default):
    if isinstance(obj, dict):
        return obj.get(key, default)
    else:
        return obj[key] if key < len(obj) else default

def common_contains(obj, key):
    if isinstance(obj, dict):
        return key in obj
    else:
        return key < len(obj)

''' Very simple Trie implementation '''

class SimpleNode:
    def __init__(self):
        self.edges = {} # outgoing edges with terminal symbols
        self.element = None # rules at this node
        
class SimpleTrie:
    ''' This Trie implementation is simpler than the one in cam.gnmt.predictors.grammar
    because it does not support non-terminals or removal. However, for the cache in the
    greedy heuristic its enough. '''
    
    def __init__(self):
        self.root = SimpleNode()
    
    def _get_node(self, seq):
        cur_node = self.root
        for token_id in seq:
            children = cur_node.edges
            if not token_id in children:
                children[token_id] = SimpleNode()
            cur_node = children[token_id]
        return cur_node
    
    def add(self, seq, element):
        self._get_node(seq).element = element
        
    def get(self, seq):
        return self._get_node(seq).element

'''TODO: this has been copied from gnmt/cam/gnmt/decoding/, could just import it from there!
Contains all the basic interfaces for the gnmt.decoding package
'''
from abc import abstractmethod
import numpy as np
import operator
from tensorflow.models.rnn.translate.utils import utils

NEG_INF = float("-inf")

'''
Some constants to define the normalization behavior for closed vocab predictor scores.

Closed vocabulary predictors (e.g. NMT) do have a predefined normally very limited 
vocabulary. In contrast, open vocabulary predictors (see UnboundedPredictor) are
defined over a much larger vocabulary (e.g. FST) s.t. it is easier to consider them
as having an open vocabulary. When combining open and closed vocabulary predictors,
we use the unk probability of closed vocabulary predictors for words outside their
vocabulary. The following flags decide (as argument to Decoder) what to do with
the closed vocabulary predictor scores on combination.
None: Do not apply any normalization
Exact: Normalize by 1 plus the number of words outside the vocabulary to make
   it a proper distributuion again
Reduced: Always normalize the closed vocabulary scores to the vocabulary which
   is defined by the open vocabulary predictors at each time step.
'''
CLOSED_VOCAB_SCORE_NORM_NONE = 1
CLOSED_VOCAB_SCORE_NORM_EXACT = 2
CLOSED_VOCAB_SCORE_NORM_REDUCED = 3

class Predictor(object):
    ''' A predictor produces the predictive probability distribution of the next
    word given the state of the predictor. The state may change during 
    predict_next() and consume(). get_state() and set_state() can be used for
    non-greedy decoding. Note: The state describes the predictor given the
    current source sentence. You cannot recover a predictor state if
    initialize() was called in between.
    '''
    
    def __init__(self):
        self.current_sen_id = 0
        pass

    def set_current_sen_id(self, cur_sen_id):
        ''' Called between initialize() calls, e.g. to increment the sentence id
        counter. It can also be used to skip sentences for the --range argument '''
        self.current_sen_id = cur_sen_id
    
    @abstractmethod
    def predict_next(self):
        ''' Return predictive distribution over the target vocabulary for the
        next word. Note that the prediction itself can change the state of the
        predictor. For example, the neural predictor updates the decoder network
        state and its attention to predict the next word. Two calls of 
        predict_next() must be separated by a consume() call.
        
        return dictionary or list with word log probabilities. All ids which are not
            set are assumed to have probability get_unk_probability()
        '''
        raise NotImplementedError
    
    def estimate_future_cost(self, hypo):
        ''' Predictors can implement their own look-ahead cost functions. They are
        used in A* if the --heuristics is set to predictor. This function should
        return the future log *cost* (i.e. the lower the better) given the current
        predictor state, given that the last word in the partial hypothesis 'hypo' is
        consumed next. This function must not change the internal predictor state. '''
        return 0.0
        
    
    def get_unk_probability(self, posterior):
        ''' Get the probability of all IDs which are not in posterior. Posterior 
        should have been generated with predict_next() '''
        return NEG_INF
    
    def initialize(self, src_sentence):
        ''' Initialize this predictor with the given source sentence '''
        pass
    
    def initialize_heuristic(self, src_sentence):
        ''' This is called after initialize() when the predictor is registered as
        heuristic predictor (i.e. estimate_future_cost() will be called in the future).
        Predictors can implement this function for initialization of their own 
        heuristic mechanisms. '''
        pass
    
    def finalize_posterior(self, scores, use_weights, normalize_scores):
        ''' This method can be used to force parameters use_weights and normalize_scores
        in predictors with dict posteriors.
        scores: unnormalized log valued scores
        use_weights: Set to false to replace all values in scores with 0
        normalize_scores: Set to true to make the exp of elements in scores sum up to 1'''
        if not scores: # empty scores -> pass through
            return scores
        if not use_weights:
            scores = dict.fromkeys(scores, 0.0)
        if normalize_scores:
            log_sum = utils.log_sum(scores.itervalues())
            ret = {k: v - log_sum for k, v in scores.iteritems()}
            return ret
        return scores
    
    @abstractmethod
    def consume(self, word):
        ''' Update internal state according
        
        return target vocabulary sized vector with word probabilities
        '''
        raise NotImplementedError
    
    @abstractmethod
    def get_state(self):
        ''' Get current predictor state '''
        raise NotImplementedError
    
    @abstractmethod
    def set_state(self, state):
        ''' Set predictor state '''
        raise NotImplementedError
    
    @abstractmethod
    def reset(self):
        ''' Reset predictor state to initial configuration '''
        raise NotImplementedError

class UnboundedVocabularyPredictor(Predictor):
    ''' This class of predictors implements models with very large target vocabulary,
    for which it is too large to list the entire posterior. Instead, they are evaluated
    only for a given list of target words. This list is usually created by taking all
    non-zero probability words from the normal predictors. A example predictor of this
    kind is the ngram predictor: Instead of listing the entire ngram vocabulary, we run
    srilm only on the words which are possible according other predictor (e.g. fst or nmt) '''

    def __init__(self):
        super(UnboundedVocabularyPredictor, self).__init__()

    @abstractmethod
    def predict_next(self, trgt_words):
        raise NotImplementedError

class Heuristic(object):
    ''' A heuristic instance can be used to estimate the future costs for a given word
    in a given state. See the heuristics module for implementations. '''
    
    def __init__(self):
        super(Heuristic, self).__init__()
        self.predictors = []

    def set_predictors(self, predictors):
        ''' predictors should be in the form of Decoder.predictors, i.e. a list
        of (predictor, weight) tuples '''
        self.predictors = predictors
    
    def initialize(self, src_sentence):
        ''' Initialize this heuristic with the given source sentence. This is not
        passed through to the heuristic predictors. '''
        pass

    @abstractmethod
    def estimate_future_cost(self, hypo):
        ''' Estimate future cost (i.e. negative score) given the predictor states
        for the given partial hypo. Note that this function is not supposed to change predictor
        states. If (e.g. for greedy heuristic) this is not possible, the predictor states
        must be changed back after execution by the implementation of this method. '''
        raise NotImplementedError
    

''' The following functions implement --combination_scheme. The function breakdown2score_partial is called
at each hypothesis expansion and should only be changed if --combination_scheme is not 'sum' and
--apply_combination_scheme_to_partial_hypos is set to true. The function breakdown2score_full is called
at each creation of a full hypothesis. 

TODO breakdown2score_length_norm and breakdown2score_bayesian could be more efficiently making use of working_score'''

def breakdown2score_sum(working_score, score_breakdown):
    return working_score

def breakdown2score_length_norm(working_score, score_breakdown):
    score = sum([Decoder.combi_arithmetic_unnormalized(s) for s in score_breakdown])
    return score / len(score_breakdown)

def breakdown2score_bayesian(working_score, score_breakdown):
    ''' See Bayesian LM interpolation with K=T'''
    if not score_breakdown:
        return working_score
    acc = []
    prev_alphas = [] # list of all alpha_i,k
    # Write priors to alphas
    for (p,w) in score_breakdown[0]:
        prev_alphas.append(np.log(w))
    for pos in score_breakdown: # for each position in the hypothesis
        alphas = []
        sub_acc = []
        for k,(p,w) in enumerate(pos): # for each predictor (p: p_k(w_i|h_i), w: prior p(k))
            alpha = prev_alphas[k] + p
            alphas.append(alpha)
            sub_acc.append(p + alpha)
        acc.append(utils.log_sum(sub_acc) - utils.log_sum(alphas))
        prev_alphas = alphas
    return sum(acc)

breakdown2score_partial = breakdown2score_sum
breakdown2score_full = breakdown2score_sum


class Decoder(object):    
    ''' A decoder instance represents a particular search strategy such as
    A*, beam search, greedy search etc. Decisions are made based on the
    outputs of one or many predictors.
    '''
    
    def __init__(self, closed_vocab_norm = CLOSED_VOCAB_SCORE_NORM_NONE):
        self.predictors = [] # Tuples (predictor, weight)
        self.heuristics = []
        self.heuristic_predictors = []
        self.predictor_names = []
        self.nbest = 1 # length of n-best list
        self.combi_predictor_method = Decoder.combi_arithmetic_unnormalized
        self.combine_posteriors = self._combine_posteriors_norm_none
        if closed_vocab_norm == CLOSED_VOCAB_SCORE_NORM_EXACT:
            self.combine_posteriors = self._combine_posteriors_norm_exact
        elif closed_vocab_norm == CLOSED_VOCAB_SCORE_NORM_REDUCED:
            self.combine_posteriors = self._combine_posteriors_norm_reduced
        self.current_sen_id = -1
        self.start_sen_id = 0
        self.apply_predictors_count = 0
    
    def add_predictor(self, name, predictor, weight=1.0):
        self.predictors.append((predictor, weight))
        self.predictor_names.append(name)
    
    def set_heuristic_predictors(self, heuristic_predictors):
        ''' Set the list of predictors used by heuristics. '''
        self.heuristic_predictors = heuristic_predictors
    
    def add_heuristic(self, heuristic):
        ''' Use set_heuristic_predictors() before adding heuristics '''
        heuristic.set_predictors(self.heuristic_predictors)
        self.heuristics.append(heuristic)
    
    def estimate_future_cost(self, hypo):
        ''' Use the added heuristics to estimate the future costs
        from the current state given partial hypothesis 'hypo' '''
        return sum([h.estimate_future_cost(hypo) for h in  self.heuristics])
    
    def consume(self, word):
        ''' Consume word with all predictors '''
        for (p, _) in self.predictors:
            p.consume(word) # May change predictor state
    
    def _get_non_zero_words(self, bounded_predictors, posteriors):
        ''' Get set of words from predictor posteriors with non-zero probability '''
        words = None
        for idx, posterior in enumerate(posteriors):
            (p, _) = bounded_predictors[idx]
            if p.get_unk_probability(posterior) == NEG_INF: # Restrict to this one
                if not words:
                    words = set(utils.common_viewkeys(posterior))
                else:
                    words = words & set(utils.common_viewkeys(posterior))
                if not words: # Special case empty set (i.e. no word is possible)
                    return set([utils.EOS_ID])
        if not words: # If no restricting predictor, use union
            words = set(utils.common_viewkeys(posteriors[0]))
            for posterior in posteriors[1:]:
                words = words | set(utils.common_viewkeys(posterior))
        return words
    
    def apply_predictors(self):
        ''' Applies the predictors and get combined prediction using predict_next
        return combined,score_breakdown: two dicts. combined maps target word ids to the combined score,
               and score_breakdown contains the scores for each predictor separately represented as tuple
               (unweighted_score, weight) '''
        self.apply_predictors_count += 1
        bounded_predictors = [el for el in self.predictors if not isinstance(el[0], UnboundedVocabularyPredictor)]
        # Get bounded posteriors
        bounded_posteriors = [p.predict_next() for (p, _) in bounded_predictors] # May change predictor state
        non_zero_words = self._get_non_zero_words(bounded_predictors, bounded_posteriors)
        # Add unbounded predictors and unk probabilities
        posteriors = []
        unk_probs = []
        bounded_idx = 0
        for (p, _) in self.predictors:
            if isinstance(p, UnboundedVocabularyPredictor):
                posterior = p.predict_next(non_zero_words)
            else: # Take it from the bounded_* variables
                posterior = bounded_posteriors[bounded_idx]
                bounded_idx += 1
            posteriors.append(posterior)
            unk_probs.append(p.get_unk_probability(posterior))
        return self.combine_posteriors(non_zero_words, posteriors, unk_probs)
    
    ''' 
    The following _combine_posteriors_* methods are strategies for closed vocabulary
    predictor score normalization, see CLOSED_VOCAB_SCORE_*
    '''
    
    def _combine_posteriors_norm_none(self, non_zero_words, posteriors, unk_probs):
        ''' Combine posteriors according CLOSED_VOCAB_SCORE_NORM_NONE '''
        combined = {}
        score_breakdown = {}
        #print("predictors:")
        #print(self.predictors)
        for trgt_word in non_zero_words:
            preds = [(utils.common_get(posteriors[idx], trgt_word, unk_probs[idx]), w)
                        for idx, (_,w) in enumerate(self.predictors)]
            combined[trgt_word] = self.combi_predictor_method(preds)
            #score_breakdown[trgt_word]  = [bla*w for (bla,w) in preds] 
            score_breakdown[trgt_word] = preds
        #print("COMBINED")
        #print(combined)
        #print("BREAKDOWN")
        #print(score_breakdown)
        return combined, score_breakdown
    
    def _combine_posteriors_norm_exact(self, non_zero_words, posteriors, unk_probs):
        ''' Combine posteriors according CLOSED_VOCAB_SCORE_NORM_EXACT '''
        n_predictors = len(self.predictors)
        score_breakdown_raw = {}
        unk_counts = [0] * n_predictors
        for trgt_word in non_zero_words:
            preds = []
            for idx, (_,w) in enumerate(self.predictors):
                if utils.common_contains(posteriors[idx], trgt_word):
                    preds.append((posteriors[idx][trgt_word], w))
                else:
                    preds.append((unk_probs[idx], w))
                    unk_counts[idx] += 1
            score_breakdown_raw[trgt_word] = preds
        renorm_factors = [0.0] * n_predictors
        for idx in xrange(n_predictors):
            if unk_counts[idx] > 1:
                renorm_factors[idx] = np.log(1.0 + (unk_counts[idx] - 1.0) * np.exp(unk_probs[idx]))  
        return self._combine_posteriors_with_renorm(score_breakdown_raw, renorm_factors)
    
    def _combine_posteriors_norm_reduced(self, non_zero_words, posteriors, unk_probs):
        ''' Combine posteriors according CLOSED_VOCAB_SCORE_NORM_REDUCED '''
        n_predictors = len(self.predictors)
        score_breakdown_raw = {}
        for trgt_word in non_zero_words: 
            score_breakdown_raw[trgt_word] = [(utils.common_get(posteriors[idx], trgt_word, unk_probs[idx]), w)
                        for idx, (_,w) in enumerate(self.predictors)]
        sums = []
        for idx in xrange(n_predictors):
            sums.append(utils.log_sum([preds[idx][0] for preds in score_breakdown_raw.itervalues()]))
        return self._combine_posteriors_with_renorm(score_breakdown_raw, sums)
    
    def _combine_posteriors_with_renorm(self, score_breakdown_raw, renorm_factors):
        n_predictors = len(self.predictors)
        combined = {}
        score_breakdown = {}
        for trgt_word,preds_raw in score_breakdown_raw.iteritems():
            preds = [(preds_raw[idx][0] - renorm_factors[idx], preds_raw[idx][1]) for idx in xrange(n_predictors)]
            combined[trgt_word] = self.combi_predictor_method(preds) 
            score_breakdown[trgt_word] = preds
        return combined, score_breakdown
    
    def set_start_sen_id(self, start_sen_id):
        self.start_sen_id = start_sen_id
        self.reset_predictors()

    def reset_predictors(self):
        for (p, _) in self.predictors:
            p.reset()
        self.current_sen_id = self.start_sen_id-1 # -1 because its incremented in initialize_predictors
            
    def initialize_predictors(self, src_sentence):
        #print("INITIALIZE")
        self.current_sen_id += 1
        for (p, _) in self.predictors:
            p.set_current_sen_id(self.current_sen_id)
            p.initialize(src_sentence)
        for h in self.heuristics:
            h.initialize(src_sentence)
    
    def set_predictor_states(self, states):
        i = 0
        for (p, _) in self.predictors:
            p.set_state(states[i])
            i = i + 1
    
    def get_predictor_states(self):
        return [p.get_state() for (p, _) in self.predictors]
    
    def set_predictor_combi_method(self, method):
        ''' method should be one of the Decoder.combi_* methods '''
        self.predictor_combi_method = method
    
    '''
    The following functions combine predictor outcomes.
    Use set_combination_method() to use one of them. Functions accept a list 
    of tuples [(out1, weight1), ...]
    You can use combi_arithmetic for the geometric mean in log space.
    The combi_* methods are usually not safe to use them with empty lists
    '''
    
    @staticmethod
    def combi_arithmetic_unnormalized(x):
        (fAcc, _) = reduce(lambda (f1, w1), (f2, w2): (f1*w1 + f2*w2, 1.0), x, (0.0, 1.0))
        return fAcc
    
    @staticmethod
    def combi_geometric_unnormalized(x):
        (fAcc, _) = reduce(lambda (f1, w1), (f2, w2): (pow(f1,w1)*pow(f2*w2), 1.0), x, (1.0, 1.0))
        return fAcc

    @abstractmethod
    def decode(self, src_sentence):
        ''' Decodes a single source sentence.
        
        src_sentence: Source sentence, blank separated words (not word ids)
        return: list of Hypothesis instances (n-best), best score first
        '''
        raise NotImplementedError

'''TODO: this is a stripped-down and modified copy of cam/gnmt/decoding/ and modified, should merge!
This module contains high level decoder algorithms such as greedy, A*, beam search.
Note that all decoders are implemented in a monotonic left-to-right way. If
we use features which do not have a natural left-to-right semantics, we
1) Restrict it to a accept/not-accept decision or
2) Change it s.t. it does have left-to-right semantics

For example, to use synchronous grammars, we could
1.) Keep track of all parse trees which match the partial prefix sequence
2.) Transform the grammar into greibach normal form

The reason for this design decision is the emphasis on NMT: The neural decoder
decodes the sequence from left to right, all other features are conceptually
more for guiding the neural network decoding. 
'''

import copy
from heapq import heappush, heappop
import heapq
import logging
from operator import add
import operator

from tensorflow.models.rnn.translate.utils import utils
from tensorflow.models.rnn.translate.utils import data_utils
from tensorflow.models.rnn.translate.decoding.core import Decoder
from tensorflow.models.rnn.translate.decoding import core
import numpy as np

NEG_INF = float("-inf")

class VanillaDecoder(Decoder):
    ''' The tensorflow vanilla decoder simply uses the probabilities given by the softmax of the neural model.
    '''
    def __init__(self, session, model):
        super(VanillaDecoder, self).__init__()
        self.session = session
        self.model = model
        self.name = "vanilla"

    def set_bucket_id(self, bucket_id):
        self.bucket_id = bucket_id

    def decode(self, token_ids):
        """ 
        Decode by passing the whole sequence to the model.
        """
        # Get batch of size 1 to feed the sentences to the model.
        encoder_inputs, decoder_inputs, target_weights = self.model.get_batch(
            {self.bucket_id: [ (token_ids, []) ]}, self.bucket_id)
        # Get output logits for the sentence.
        _, _, output_logits = self.model.step(self.session, encoder_inputs, decoder_inputs,
                                   target_weights, self.bucket_id, True)
        #print("Len logits={}x{}x{}".format(len(output_logits),len(output_logits[0]),len(output_logits[0][0])))

        # This is a greedy decoder - outputs are just argmaxes of output_logits.
        trgt_sentence = [int(np.argmax(logit, axis=1)) for logit in output_logits]

        # If there is an EOS symbol in outputs, cut them at that point.
        if data_utils.EOS_ID in trgt_sentence:
            trgt_sentence = trgt_sentence[:trgt_sentence.index(data_utils.EOS_ID)]
        score = 0
        return [Hypothesis(trgt_sentence, score, score)]

class GreedyDecoder(Decoder):
    ''' The greedy decoder does not revise decisions and therefore does not
    have to maintain predictor states. Therefore, this implementation is
    particularly simple and can be used as template for more complex
    decoders.
    '''
    #def __init__(self, closed_vocab_norm):
    def __init__(self, closed_vocab_norm=core.CLOSED_VOCAB_SCORE_NORM_NONE):
        super(GreedyDecoder, self).__init__(closed_vocab_norm)
        self.name = "greedy"

    def decode(self, src_sentence):
        self.initialize_predictors(src_sentence)
        trgt_sentence = []
        score_breakdown = []
        trgt_word = None
        score = 0.0
        print("Decoding loop")
        while trgt_word != utils.EOS_ID:
            posterior,breakdown = self.apply_predictors()
            trgt_word = utils.argmax(posterior)
            score += posterior[trgt_word]
            print("trgt_word={} score={}".format(trgt_word, score))
            trgt_sentence.append(trgt_word)
            score_breakdown.append(breakdown[trgt_word])
            self.consume(trgt_word)
        return [Hypothesis(trgt_sentence, score, score_breakdown)]

class Hypothesis:
    def __init__(self, trgt_sentence, total_score, score_breakdown = []):
        self.trgt_sentence = trgt_sentence
        self.total_score = total_score
        self.score_breakdown = score_breakdown

    def __repr__(self):
        return "%s (%f)" % (' '.join(str(w) for w in self.trgt_sentence),
                            self.total_score)

class PartialHypothesis:
    ''' Represents a partial hypothesis in various decoders. '''
    def __init__(self, initial_states = None):
        self.predictor_states = initial_states
        self.trgt_sentence = []
        self.score = 0.0
        self.score_breakdown = []
        self.word_to_consume = None
    
    def get_last_word(self):
        if not self.trgt_sentence:
            return None
        return self.trgt_sentence[-1]
    
    def generate_full_hypothesis(self):
        return Hypothesis(self.trgt_sentence, self.score, self.score_breakdown)
    
    def expand(self, word, new_states, score, score_breakdown):
        ''' Creates a new hypothesis by adding a new word with given
        probability and updates the state. '''
        hypo = PartialHypothesis(new_states)
        hypo.score = self.score + score
        hypo.score_breakdown = copy.copy(self.score_breakdown)
        hypo.trgt_sentence = self.trgt_sentence + [word]
        hypo.add_score_breakdown(score_breakdown)
        return hypo
    
    def cheap_expand(self, word, score, score_breakdown):
        ''' Creates a new hypothesis by adding a new word with given
        probability. Does NOT update the state but marks the hypothesis
        that the word has not been consumed yet. This can save memory
        because we can reuse the current state for many hypothesis, and
        consuming is normally cheap compared to predict_next '''
        hypo = PartialHypothesis(self.predictor_states)
        hypo.score = self.score + score
        hypo.score_breakdown = copy.copy(self.score_breakdown)
        hypo.trgt_sentence = self.trgt_sentence + [word]
        hypo.word_to_consume = word
        hypo.add_score_breakdown(score_breakdown)
        return hypo

    def add_score_breakdown(self, added_scores):
        ''' Adds scores to the breakdown '''
        self.score_breakdown.append(added_scores)
        self.score = core.breakdown2score_partial(self.score, self.score_breakdown)
        #if not self.score_breakdown:
        #    old_scores = [0.0] * len(added_scores)
        #else:
        #    old_scores = self.score_breakdown
        #self.score_breakdown = map(add, old_scores, [p for (p,_) in added_scores])

class BeamDecoder(Decoder):
    ''' This decoder implements beam search without heuristics.
    '''
    
    #def __init__(self, closed_vocab_norm, beam_size, early_stopping = True):
    def __init__(self, beam_size, early_stopping=True,
                 closed_vocab_norm=core.CLOSED_VOCAB_SCORE_NORM_NONE):
        super(BeamDecoder, self).__init__(closed_vocab_norm)
        self.beam_size = beam_size
        self.stop_criterion = self._best_eos if early_stopping else self._all_eos
        self.name = "beam"

    def _best_eos(self, hypos):
        return hypos[-1].get_last_word() != utils.EOS_ID

    def _all_eos(self, hypos):
        for hypo in hypos:
            if hypo.get_last_word() != utils.EOS_ID:
                return True
        return False
    
    def decode(self, src_sentence):
        self.initialize_predictors(src_sentence)
        hypos = [PartialHypothesis(self.get_predictor_states())]
        it = 0
        while self.stop_criterion(hypos):
            if it > 2*len(src_sentence): # prevent infinite loops
                break
            it = it + 1
            next_hypos = []
            next_scores = []
            #print("it={} num_hypos={}".format(it, len(hypos)))
            for i, hypo in enumerate(hypos):
                if hypo.get_last_word() == utils.EOS_ID:
                    next_hypos.append(hypo)
                    next_scores.append(hypo.score)
                    continue
                self.set_predictor_states(copy.deepcopy(hypo.predictor_states))
                if not hypo.word_to_consume is None: # Consume if cheap expand
                    self.consume(hypo.word_to_consume)
                posterior,score_breakdown = self.apply_predictors()
                hypo.predictor_states = self.get_predictor_states()
                top = utils.argmax_n(posterior, self.beam_size)

                # Expand hypo with top-scoring next words
                #print("Expand hypo={} into {} new hypos".format(i, len(top)))
                for trgt_word in top:
                    next_hypo = hypo.cheap_expand(trgt_word, posterior[trgt_word],
                                                  score_breakdown[trgt_word])
                    next_hypos.append(next_hypo)
                    next_scores.append(next_hypo.score)
            hypos = [next_hypos[idx] for idx in np.argsort(next_scores)[-self.beam_size:]]
            #bestHypo = hypos[-1]
            #print("best hypo:")
            #print(bestHypo.trgt_sentence)

        #self.set_predictor_states(bestHypo.predictor_states) # Leave predictors in final state
        #if not bestHypo.word_to_consume is None: # Consume if cheap expand
        #    self.consume(bestHypo.word_to_consume)
        ret = [hypos[-idx-1].generate_full_hypothesis() for idx in xrange(len(hypos)) 
                     if hypos[-idx-1].get_last_word() == utils.EOS_ID]
        if not ret:
            logging.warn("No complete hypotheses found for %s" % src_sentence)
            return [hypos[-idx-1].generate_full_hypothesis() for idx in xrange(len(hypos))]
        return ret


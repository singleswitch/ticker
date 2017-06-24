#!python
#cython: boundscheck=False

import time
import numpy as np
from math import exp

cimport numpy as np
DTYPE = np.float64

ctypedef np.float64_t DTYPE_t
 
#cython --directive boundscheck=False,wraparound=False,nonecheck=False  ticker_scores.pyx
#gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing -I/usr/include/python2.6 -o ticker_scores.so ticker_scores.c

class TickerScoresFast():
    
    def __init__(self, i_gauss_var, i_prob_thresh, i_n_samples):
        self.nletters = 28          #number of letters in the alphabet
        self.sigmoid_rate = 30.0     #Sigmoid rate for classification purposes
        self.dict_length = 112973    #Number of words in dictionary 
        self.nsamples = i_n_samples  #Number of samples used to approximte the integrals
        self.eps = 1E-20
        
        #Some precalculated constants
        self.exp_gauss_const = 1.0 / (2.0*i_gauss_var); #normaliser for exp in gaussian
        self.gauss_normaliser =  1.0 / (np.sqrt(2* np.pi * i_gauss_var ) )
        self.sigmoid_const = self.sigmoid_rate*np.log(i_prob_thresh/(1.0-i_prob_thresh))
        self.thresh= i_prob_thresh
 
    def updateWordPriors(self, np.ndarray[double, ndim=1] i_t1_x, 
            np.ndarray[double, ndim=1] m, 
            int target_letter_idx, 
            int target_word_idx, 
            np.ndarray[double, ndim=1] word_priors,
            np.ndarray[int, ndim=1] letter_list, 
            bool i_classify,
            np.ndarray[double, ndim=2] target_scores,
            np.ndarray[double, ndim=2] probs,
            np.ndarray[double, ndim=2] letter_scores,
            np.ndarray[double, ndim=1] short_tmp_letter_scores,
            np.ndarray[double, ndim=1] long_tmp_letter_scores,
            np.ndarray[double, ndim=1] normalisers, 
            np.ndarray[double, ndim=2] factor, 
            np.ndarray[double, ndim=1] classify_probs  ):
       
        cdef double delta_t = np.abs( i_t1_x[1] - i_t1_x[0] )
        cdef double const =  delta_t * self.gauss_normaliser
        cdef unsigned int n, i, j
        array_range = range(0, self.nsamples)
        
        for n in array_range:
            short_tmp_letter_scores   =  np.exp( -((i_t1_x[n] - m)**2) *  self.exp_gauss_const )
            letter_scores[n,:] =  short_tmp_letter_scores 
            long_tmp_letter_scores =  short_tmp_letter_scores.take(letter_list) 
            probs[n,:] = long_tmp_letter_scores*word_priors
        
        cdef double tmp_prob, tmp_prob2
          
        target_scores =  const*letter_scores.take( [target_letter_idx], axis=1)
        normalisers =  np.sum( probs , axis=1 ) 
        idx = np.nonzero( np.abs(normalisers) > 0.0 )[0]
        factor[idx,0] = 1.0 / (normalisers.take(idx))
        probs *= factor
        word_priors = np.sum(probs*target_scores, axis=0)
        tmp_prob = np.sum(word_priors)
        if tmp_prob > 1.0:
            tmp_prob = 1.0 / np.sum(word_priors)
            word_priors *= tmp_prob
        if not i_classify:
            return (word_priors, classify_probs)
        is_valid = False
        for n in array_range:
            long_tmp_letter_scores = probs.take([n], axis=0).flatten()
            idx = np.nonzero( long_tmp_letter_scores  >= (self.thresh-0.1) )[0]
            if len(idx) > 0:
                is_valid = True
                tmp_prob = target_scores[n]
                x1 =  long_tmp_letter_scores.take(idx)  
                x  = np.log(1.0 - x1) - np.log(x1) 
                tmp_probs = self.classify(x) * tmp_prob  
                classify_probs[idx] += tmp_probs
        if is_valid:
            tmp_prob = np.sum(classify_probs)
            if tmp_prob > 1.0:
                tmp_prob = 1.0 / tmp_prob
                classify_probs *= tmp_prob
        return (word_priors, classify_probs)
   
    def classify(self, i_x ):
        factor = 1.0 +np.exp(self.sigmoid_rate*i_x + self.sigmoid_const )
        return 1.0 / factor
        
 
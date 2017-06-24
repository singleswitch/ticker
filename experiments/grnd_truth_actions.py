

import sys
sys.path.append("../")
from utils import  WordDict
import numpy as np

class GrndTruthActions():
    #################################################### Init
    def __init__(self, i_disp_top_words=False, i_display=False, i_fox_example=False):
        self.dict = WordDict("../dictionaries/nomon_dict.txt") 
        self.max_time_word_scale = 2
        self.alphabet = list("abcdefghijklmnopqrstuvwxyz_.") 
        self.initLetterIndices()
        self.word_indices = self.wordIndices()
        #Display
        self.disp_top_words =i_disp_top_words
        self.display = i_display
        #Diagnostic example fox/for: showing that for has a significant prior compared to fox
        self.fox_example = i_fox_example
        
    def  initLetterIndices(self):
        """Store the alphabet positions in dictionary - optimisation when extracting word scores"""
        self.letter_indices = {}
        for n in range(0,  len(self.alphabet)):
            self.letter_indices[self.alphabet[n]] =  n
    
    ####################################################### Main
    
    def computeAll(self, i_word_thresh):
        #Go through the whole dictionary, and see for a given threshold which words are difficult to identify
        o_results = np.array([self.compute(word, i_word_thresh, False, n) for (n, word) in enumerate(self.dict.words[0:100])]) 
        (letter_idx, errors, selected_words) = (np.int32(o_results[:,0]),np.int32(o_results[:,1]),o_results[:,2]) 
        idx = np.nonzero( errors > 0 )[0] 
        print "idx = ", idx
        print "*****************************************************************"
        print "Number of words: ", len(self.dict.words)
        print "Number of words processed: ", len(errors)
        print "Number of errors:", len(idx)
        print "Percentage errors:", 100.0*len(idx) /float(len(errors)), "%"
        print "*****************************************************************"
        for n in idx:
            print "Input word = ", self.dict.words[n], " Output word = ", selected_words[n], " letter idx=", letter_idx[n] 
        
            
    def compute(self,i_word, i_word_thresh, i_store_probs=False, i_cur_word_idx=None):
        if self.display and (i_cur_word_idx is not None):
            print "n = ", i_cur_word_idx
        max_time_word = self.max_time_word_scale*len(i_word)
        log_priors =  np.array(self.dict.log_probs)
        (o_probs, o_letter_idx, is_error, o_word)  = ([],0,0, None) 
        for letter_idx in range(0, max_time_word):
            target_idx = self.wrapIndices(letter_idx, len(i_word)) 
            target_letter = i_word[target_idx]
            #Uncomment for fox/for example
            if self.fox_example and (letter_idx == 2):
                click_scores = np.log(self.letterProbs()[:,0])
            else:
                click_scores = -np.inf*np.ones(len(self.alphabet)) 
            click_scores[self.letter_indices[target_letter]] = 0.0
            log_priors = self.logWordPosteriors( log_priors,  letter_idx,  click_scores )
            #Find the word the maximum posterior probability, and extract the posterior prob of the desired word
            idx_max = np.argmax(log_priors)
            best_word = self.dict.words[idx_max]
            best_prob = np.exp(log_priors[idx_max])
            cur_prob =  np.exp(log_priors[self.word_indices[i_word]])
            #Update the display - diagnostic
            if self.fox_example:
                idx_for = np.array(self.dict.words == "for_") 
                idx_fox = np.array(self.dict.words == "fox_") 
                print "prob for = ",  np.exp(log_priors[idx_for]), " prob fox  = ", np.exp(log_priors[idx_fox])
            if self.display:
                self.updateDisplay(click_scores,letter_idx,target_letter, log_priors, i_word,cur_prob, best_word, best_prob)
            #Classification step
            if i_store_probs:
                o_probs.append(cur_prob)
            if np.abs(best_prob) >= np.abs(i_word_thresh-1E-10):
                if not (best_word == i_word):
                    is_error=1
                o_word  = str(best_word)
                o_letter_idx = letter_idx
                break
        if o_word is None:
            is_error = 1
        if i_store_probs:
            return (o_probs, o_letter_idx, is_error, o_word)
        else:
            return (o_letter_idx, is_error, o_word)

    ##################################### Word posteriors
    
    def wordIndices(self):
        #Store all the word positions in dict: diagnostic purposes
        word_indices = {}
        for n in range(0, len(self.dict.words)):
            word_indices[self.dict.words[n]] = n
        return word_indices
    
    def wrapIndices(self, i_letter_idx, i_word_lengths):
        """Wrap the index around in cases where letter_idx > input word lengths."""
        letter_idx = -i_letter_idx / i_word_lengths
        letter_idx *= i_word_lengths
        letter_idx += i_letter_idx
        return letter_idx
    
    def  curLetterList(self, i_letter_idx ):
        """Extract the current letter of the alphabet as an index"""
        letter_indices = self.wrapIndices( i_letter_idx, self.dict.word_lengths)
        letter_list =  np.array( [self.letter_indices[self.dict.words[n][idx]]  for n, idx in enumerate( letter_indices, start=0) ] )
        return letter_list
     
    def logWordPosteriors(self, i_log_priors, i_letter_idx, i_log_letter_scores ):
        letters_idx = self.curLetterList(i_letter_idx )
        log_scores = self.dict.normalise( i_log_letter_scores[letters_idx] + i_log_priors )
        return  log_scores
    
    ##################################### Diagnostic
 

    def updateDisplay(self, click_scores, letter_idx, target_letter, log_priors, i_target_word, i_target_prob, i_best_word, i_best_prob):
        """Print the result after some clicks were received: Find the maximum word prob etc."""
        print "************************************************************************************************"
        max_idx_letter = np.argmax(click_scores)
        print "letter_idx=%d, target_letter=%s" % (letter_idx, target_letter ) 
        print " best_letter=%s, score=%.2f," % ( self.alphabet[max_idx_letter], click_scores[max_idx_letter]),  
        print " target_word=%s, prob=%.3f," %(i_target_word, i_target_prob), 
        print " best_word=%s, best_prob=%.3f" % (i_best_word, i_best_prob)
        if self.disp_top_words:
            sort =np.flipud(  np.argsort(  log_priors )[-10:] )
            top_words = self.dict.words[sort]
            scores = np.exp(  log_priors[sort] )
            print "Top words : ",  top_words
            print "Their scores : ", scores.flatten()
            print ", sum (word_priors)= %.3f, sum(top words) =%.3f "   %( np.sum(np.exp(log_priors)) , np.sum(scores))
             

    def letterProbs(self):
        #Click probs for fox/for prior example
        letters = np.array(list("fqwaglrxbhmsycintzdjou_ekpv."))
        print "letters = ", letters
        p =np.array( [[0.0000,-11.3404],
        [0.0000, -9.5746],
        [0.0036, -2.2619],
        [0.0770, 0.7934],
        [0.0000, -7.5973],
        [0.0000, -6.9205],
        [0.0328, -0.0585],
        [0.8604, 3.2071],
        [0.0003, -4.8276],
        [0.0000, -6.9871],
        [0.0030, -2.4438],
        [0.0227, -0.4285],
        [0.0000, -7.9260],
        [0.0000, -14.4611],
        [0.0000, -11.4382],
        [0.0000, -7.0120],
        [0.0000, -9.4820],
        [0.0000, -15.7841],
        [0.0000, -13.3975],
        [0.0000, -7.8847],
        [0.0000, -7.8520],
        [0.0000, -15.7840],
        [0.0000, -15.7841],
        [0.0000, -8.6626],
        [0.0000, -6.8927],
        [0.0000, -15.7784],
        [0.0000, -15.7491],
        [0.0000, -15.1990]])
        print "Size p = ", p.shape
        new_probs = []
        for (n,letter) in enumerate(self.alphabet):
            idx = np.nonzero(letters == letter)[0]
            new_probs.append(p[idx,:].flatten())
            print "letter = ", letter, " prob = ", new_probs[-1]
        return np.array(new_probs)

if __name__ ==  "__main__":
    g = GrndTruthActions(i_disp_top_words=False, i_display=False, i_fox_example=False)
    if g.fox_example:
        g.compute("fox_", i_word_thresh=0.9)
    else:
        #g.compute("there_", i_word_thresh=0.9)
        g.computeAll(i_word_thresh=0.7)
    
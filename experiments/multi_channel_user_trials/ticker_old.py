
import numpy as np
from sys import stdin, exit, argv
from click_distr_old import ClickDistribution

class TickerOld():
    def __init__(self, i_disp):
        self.language_model = LanguageModel()
        self.click_distr = ClickDistribution()
        self.min_val = 1E-5
        self.disp=i_disp
        
    #################################################### Main
    
    def evalScore(self, i_observations):
        scores = self.click_distr.logLikelihood(i_observations)
        idx = np.nonzero(np.exp(scores) <= self.min_val)
        scores[idx] = np.log(self.min_val)
        return scores
    
    def newLetter(self, i_alphabet):
        self.letter_scores = np.zeros(len(i_alphabet))
    
    def updateLetterScores(self,i_scores, i_alphabet, i_long_alphabet): 
        """Extract the recorded log scores, to compare with new computations"""
        log_scores = np.array(i_scores)
        for (n, letter) in enumerate(i_alphabet):
            letter_idx = np.nonzero(np.array(i_long_alphabet) == letter)[0]
            letter_scores = np.array(log_scores[letter_idx])
            letter_score = self.__elnsum(letter_scores[0], letter_scores[1]) + np.log(2)
            self.letter_scores[n] = self.__elnprod(self.letter_scores[n], letter_score)
            #print "Letter = ", letter, " score = ", self.letter_scores[n] 
        
    def updateWordProbs(self,i_letters):
        #Store the scores and letters as a dictionary
        letter_scores = {}
        for (n, letter) in enumerate(i_letters):
            letter_scores[letter] = self.letter_scores[n]
        #Update word log probs
        for word in  self.word_log_probs.keys():
            letter_index = int(self.letter_index)
            while letter_index >= len(word):
                letter_index = letter_index - len(word)    
            self.word_log_probs[word] += letter_scores[word[letter_index]]
        self.letter_index += 1
        self.__normaliseWordLogProbs()
        (words, probs) = self.getBestWordProbs(i_n=1)
        if self.disp:
            print "********************************************************"
            print "TICKER OLD: Best words = ", words
            print "TICKER OLD: Their probs = ", probs
    
    def getBestWordProbs(self, i_n=-1):
        """Return the best i_n words, if i_n=-1 all will be returned"""
        words = np.array(self.word_log_probs.keys()) 
        log_probs = -np.array(self.word_log_probs.values()).flatten()
        best_idx = np.argsort(log_probs).flatten()
        if i_n > 0:
            best_idx = best_idx[0:i_n]
        o_words = list( words[best_idx] )
        o_probs = list( np.exp(-log_probs[best_idx]))
        return (o_words, o_probs)
        
        
    def newWord(self, i_alphabet):
        self.word_log_probs = dict(self.language_model.getDict())  
        self.letter_index = 0  
        self.newLetter(i_alphabet)
        
    ######################################### Private
    
    def __elnprod(self, elnx, elny):
        if np.isinf(elnx) or np.isinf(elny):
            return -np.inf
        return elnx + elny
    
    
    def __elnsum(self, elnx, elny):
        if np.isinf(elnx) and np.isinf(elny):
            return -np.inf
        if np.isinf(elnx):
            return elny
        if np.isinf(elny):
            return elnx
        if elnx > elny:
            return elnx + np.log( 1.0 + np.exp( elny - elnx ))
        return elny + np.log( 1.0 + np.exp( elnx - elny ))
    
    def __expTrick(self, i_log_probs):
        """Compute log(sum(exp(log(x)))
        Input:
        ====== 
               i_log_probs: * N x D matrix
                            * N is the number of examples
                            * D is the dimensions
                            * All computations are done over D (column space)
        ======
        Output:
        ====== 
               o_log_sum: * max(i_log_probs) + log(sum(exp(i_log_probs - log_max))
                          * This will be an N x 1 matrix"""
        o_log_sum = np.max( i_log_probs, axis=1).flatten()
        data = (np.exp( i_log_probs.transpose() - o_log_sum ) ).transpose()
        (rows,cols) = np.nonzero(np.isnan(data)  )
        data[rows,cols] = 0
        exp_sum = np.sum(data, axis = 1)
        idx = np.nonzero( exp_sum > 0.)[0]
        if len(idx) > 0:
            exp_sum[idx] = np.log(exp_sum[idx])
            o_log_sum[idx] += exp_sum[idx]
        return o_log_sum
      
    def __normaliseWordLogProbs(self):
        vals = np.array(self.word_log_probs.values())
        if len(vals) < 1:
            return
        vals = vals.reshape([1, len(vals)])
        log_sum = self.__expTrick(vals).flatten()[0]
        vals -= log_sum
        words = self.word_log_probs.keys()
        self.word_log_probs =  self.__wordListsToDict(words, vals.flatten())
    
    def __wordListsToDict(self, i_list_words, i_vals):
        wi = [ [i_list_words[n], i_vals[n]] for n in range(0, len(i_vals))]
        return dict(wi) 

class LanguageModel(object):
    
    ################################ Init Functions    
    def __init__(self, i_dict_name="../../dictionaries/nomon_dict.txt"):
        #t= time.time()
        #print "Time to load dictionary = ", (time.time() - t )*1000, "ms"
        self.loadDict(i_dict_name)
        self.__dict['.'] = self.__dict['the_']
         
    def loadDict(self, i_filename):
        print "Loading dictionary..." , i_filename
        file_name = file(i_filename)
        d = file_name.read()
        file_name.close()
        b = d.split('\n')
        wi = dict([[bb.split()[0]+"_",int(bb.split()[1])] for bb in b if (len(bb)>0) & (bb.find("_")==-1)])
        vals = np.array(wi.values())
        sum_freq = np.log(np.sum(vals))
        vals = np.log(vals) - sum_freq
        self.__dict  = self.__wordListsToDict( wi.keys(), vals)
        print "first item = ", self.__dict.keys()[0], " ", self.__dict.values()[0]
        print "last item = ", self.__dict.keys()[-1], " ", self.__dict.values()[-1]
    
    def __loadWordValsDict(self, i_file_name):
        f = file(i_file_name + "_vals.text",'r')
        vals = np.float64(np.array(f.read()[1:-1].split(", ")))
        f.close()
        f = file("nomon_dict" + "_words.text",'r')
        keys = f.read()[2:-2].split("', '")
        self.__dict = self.__wordListsToDict(keys, vals)
        print "LOADING separate"
        print "first item = ", self.__dict.keys()[0], " ", self.__dict.values()[0]
        print "last item = ", self.__dict.keys()[-1], " ", self.__dict.values()[-1]
        
    ######################### General private functions 
    
    def __wordListsToDict(self, i_list_words, i_vals):
        wi = [ [i_list_words[n], i_vals[n]] for n in range(0, len(i_vals))]
        return dict(wi) 

    ######################### Set functions    
    def setDict(self, i_file_name):
        self.__loadDict(i_dict_name)
            
    ######################### Get functions
    def getDict(self):
        return self.__dict
    
if __name__ ==  "__main__":
    model = LanguageModel()
    dict = model.getDict()
    
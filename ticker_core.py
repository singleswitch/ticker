 
import numpy as np 
from utils import Utils, WordDict
import time
from kernel_density import NonParametricPdf

class TickerCore():
    """  This is file contains the core algorithm of ticker.
        Input:
              * i_select_thresh: If p(word|clicks) >  word_select_thresh, we'll ask the user if he/she wants to select this word.
              * i_dict_name: The dictionary containing all the prior probabilities of words that can be selected."""
    
    ############################################# Initialisation functions    

    def __init__(self, i_select_thresh=0.9, 
                       i_dict_name="dictionaries/nomon_dict.txt" ):
        self.click_distr = None
        self.letter_idx = 0  
        self.click_times = []
        self.params = {}
        self.utils = Utils()
        self.setWordSelectThresh(i_select_thresh)
        self.setDict(i_dict_name)
        self.calibrate = False
        #Debug
        self.disp = False
        self.diagnostic = False
        
    ############################################# Main functions    
   
    def undoLastLetter(self): 
        self.click_times = []
       
    def newClick(self,  i_click_time):
        """Store the new click"""
        self.click_times.append( i_click_time )
        click_scores = self.click_distr.logLikelihood( self.click_times, i_log=True )
        return click_scores
         
    def newWord(self):  
        """Each time a new word has to be predicted this function has to be called."""
        self.dict.log_probs = np.array(self.prior_log_probs)
        if  self.click_distr is not None:
            while len(self.click_distr.train_obs) > len(self.click_distr.obs_letters):
                self.click_distr.train_obs.pop()
        self.letter_idx = 0  
        self.click_times = []
       
    def newLetter(self, i_process_word_selections=True, i_letter_scores=None):    
        """Process all the received clicks after the letters associated with them have been received"""  
        if not self.clicksReceived():
            return
        #Add the click times to the training data
        if self.is_train:
            self.click_distr.storeObservation(self.click_times)  
        #Update the  word posteriors: Priors for next iteration
        if i_letter_scores is not None:
            click_scores = np.array(i_letter_scores)
        else:
            click_scores =  self.click_distr.logLikelihood( self.click_times, i_log=True )
        self.updateWordPosteriors( click_scores )
        #Find the word the maximum posterior probability, and extract the posterior prob of the desired word
        best_idx = np.argmax(self.dict.log_probs)
        best_score = np.exp(self.dict.log_probs[best_idx])
        if self.diagnostic:
            print "TICKER CORE NEW: Best score = ", best_score, " best word = ", self.dict.words[best_idx]
        selected_word = self.selectWord(best_score, best_idx, i_process_word_selections)
        if (selected_word is None) or (not i_process_word_selections):
            self.letter_idx += 1
            self.click_times = []
        return selected_word

    def selectWord(self, i_best_score, i_best_idx, i_process_word_selections):
        if i_best_score < self.params['word_select_thresh']:
            return
        selected_word = self.dict.words[i_best_idx]
        if self.disp: 
            print "In ticker_core, selected_word = ", selected_word, " prob = ", i_best_score
        if not i_process_word_selections:
            if selected_word == '.':
                return '.'
            return selected_word[0:-1]
        self.train(selected_word)
        self.newWord()
        return selected_word 

    def clicksReceived(self):
        return  len(self.click_times) > 0
    
    def train(self, i_selected_word): 
        if not self.is_train: 
            print "NO TRAINING: return"
            return
        train_word = self.getTrainingWord(i_selected_word) 
        self.click_distr.train(train_word)
        
    def trainClickDistrAndInitialise(self, i_selected_word):
        """* Even if the click distr is not trainable it will be trained
           * All samples will be used to initialise a histogram (no online adaptation)
           * This is typically used only if there is no uncertainty to which word the 
            user was trying to write, i.e., d. """  
        (is_train, learn_rate) = (self.click_distr.is_train, self.click_distr.learning_rate)
        if not is_train:
            print "NO training"
            return
        (learn_delay, learn_std) = (self.click_distr.learn_delay, self.click_distr.learn_std)
        (learn_fp_rate, learn_fr) = (self.click_distr.learn_fp, self.click_distr.learn_fr)
        self.click_distr.learning_rate = 1.0
        print "LEARN DELAY = ", learn_delay, " std = ",  learn_std, " fp = ", learn_fp_rate, " fr = ", learn_fr
        self.click_distr.histogram.learning_rate = self.click_distr.learning_rate 
        #Correct the letter index (one more) because i_process_word_selections was False 
        self.letter_idx -= 1
        self.train(i_selected_word)
        self.click_distr.is_train = is_train
        self.click_distr.learning_rate = learn_rate 
        self.click_distr.histogram.learning_rate = learn_rate
        self.click_distr.is_train = is_train
        self.click_distr.learn_delay = learn_delay 
        self.click_distr.learn_std = learn_std
        self.click_distr.learn_fp = learn_fp_rate
        self.click_distr.learn_fr = learn_fr
        self.newWord()
            
    ############################################# Get Functions

    def getTrainingWord(self, i_selected_word):
        train_word = list(str(i_selected_word))
        if self.letter_idx < (len(i_selected_word)-1):
            train_word =  train_word[0:(self.letter_idx+1)]
            return  "".join(train_word)
        if i_selected_word == ".":
            if self.letter_idx == 0:
                return "".join(train_word)
            for n in range(0, self.letter_idx):
                train_word.append(".")
            return "".join(train_word)
        letter_idx = self.warpIndices(self.letter_idx+1, len(i_selected_word))
        if letter_idx == 0:
            word_multiple =  (self.letter_idx+1) / len(i_selected_word)
            for n in range(1, word_multiple):
                train_word.extend( list(str(i_selected_word)) )                
            return "".join(train_word)
        word_multiple = self.letter_idx / len(i_selected_word)
        for n in range(1, word_multiple):
            train_word.extend( list(str(i_selected_word)) )               
        end_idx = len(i_selected_word) + letter_idx
        for n in range(0, end_idx):
            train_word.append(i_selected_word[n])
        return "".join(train_word)
 
    def getBestWordProbs(self, i_n=-1):
        """Return the best i_n words, if i_n=-1 all will be returned"""
        best_idx = np.argsort(-self.dict.log_probs).flatten()
        if i_n > 0:
            best_idx = best_idx[0:i_n]
        return (self.dict.words[best_idx], np.exp(self.dict.log_probs[best_idx]))
    
    def getLetterIndex(self):
        return self.letter_idx
    
    def getNumberClicks(self):
        if not self.clicksReceived():
            return 0
        return len(self.click_times) 
    
    ############################################ Set Functions
    
    def setClickDistr(self, i_click_distr):
        self.is_train = i_click_distr.is_train
        self.click_distr = i_click_distr
        if i_click_distr is None:
            return
        self.letter_indices = self.letterIndices()
        self.newWord()
        
    def setChannelConfig(self, i_channel_config):
        self.click_distr.reset(i_channel_config)
 
    def setWordSelectThresh(self, i_value):
        self.params['word_select_thresh'] = i_value
    
    def setDict(self, i_file_name):
        self.params['dict_name'] =  i_file_name
        self.dict = WordDict(self.params['dict_name'])
        self.prior_log_probs = np.array(self.dict.log_probs)
        self.newWord()
        self.word_indices = self.wordIndices()
 
    ##################################### Word posteriors
    
    def letterIndices(self):
        """Store the alphabet positions in dictionary - optimisation when extracting word scores"""
        letter_indices = {}
        for n in range(0, self.click_distr.loc.shape[0] ):
            letter_indices[ self.click_distr.alphabet[n]] = n
        return  letter_indices

    def wordIndices(self):
        #Store all the word positions in dict: diagnostic purposes
        word_indices = {}
        for n in range(0, len(self.dict.words)):
            word_indices[self.dict.words[n]] = n
        return word_indices
    
    def warpIndices(self, i_letter_idx, i_word_lengths):
        """Wrap the index around in cases where letter_idx > input word lengths."""
        letter_idx = -i_letter_idx / i_word_lengths
        letter_idx *= i_word_lengths
        letter_idx += i_letter_idx
        return letter_idx
    
    def curLetterList(self, i_letter_idx ):
        """Extract the current letter of the alphabet as an index"""
        letter_indices = self.warpIndices( i_letter_idx, self.dict.word_lengths)
        letter_list =  np.array( [self.letter_indices[self.dict.words[n][idx]]  for n, idx in enumerate(letter_indices) ] )
        return letter_list
     
    def updateWordPosteriors(self, i_log_letter_scores ):
        letters_idx = self.curLetterList(self.letter_idx)
        self.dict.log_probs = self.dict.normalise( i_log_letter_scores[letters_idx] + self.dict.log_probs )
       
    #################################################### Display
    
    def dispClickTimes(self, i_grnd_truth_word=None, i_selected_word=None):
        if not self.disp:
            return 
        letter_scores = self.click_distr.logLikelihood(self.click_times, i_log=True)
        click_time_str = self.utils.stringVector(np.array(self.click_times)) 
        print "click_times = ", click_time_str, " letter index = ", self.letter_idx, 
        if i_grnd_truth_word is not None:
            warped_idx = self.wrapIndices(self.letter_idx, len(i_grnd_truth_word))
            print "Selected word = ",  i_selected_word, " grnd truth = ", i_grnd_truth_word,
            print " ", i_grnd_truth_word[warped_idx]
        else:
            print " "
        print "============================================================================="
        #print  " stored keys = ", cn.keys()
        for (m, letter) in enumerate( self.click_distr.alphabet):
            click_time_str = self.utils.stringVector(np.array(self.click_times)-self.click_distr.delay) 
            loc_str = self.utils.stringVector(self.click_distr.loc[m,:] , i_type="%.3f") 
            param_str = "delay=%.3f, std=%.3f, fr=%.3f, fp_rate=%.3f" % (self.click_distr.delay, 
                self.click_distr.std, self.click_distr.fr, self.click_distr.fp_rate)
            print "letter=%s, loc=%s, click_times-delay=%s, score=%.3f, %s" % (letter, loc_str, 
                click_time_str, letter_scores[m], param_str )  
      
    def dispBestWords(self):
        if not self.disp:
            return 
        print "Best words: "
        (best_words, best_word_scores) = self.getBestWordProbs(10)
        for n in range(0, len(best_words)):
            print best_words[n], " ", best_word_scores[n]
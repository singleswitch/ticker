
import os, cPickle, sys
sys.path.append("../../")
sys.path.append("../")
import numpy as np
import scipy.stats.distributions as sd
import pylab as p
from min_edit_distance import MinEditDistance
from click_distr import ClickDistribution
from utils import Utils, DispSettings
from ticker_core import TickerCore
from kernel_density import NonParametricPdf
from user_trial_setup import ExperimentSettingsStudents
 

#DIAGNOSTIC
from ticker_old import TickerOld

class TickerChannelResults(object):
    ############################################ Init
    def __init__(self):
        #Load pre-computed histogram
        self.load_click_distr = False
        self.click_distr_dir = "../../../user_trial_pictures/"
        self.root_dir = os.sep.join(['../../../user_trials/multi_channel_experiment/students/','user'])
        self.dist_measure = MinEditDistance()
        self.settings = ExperimentSettingsStudents()
        self.ticker = TickerCore(i_select_thresh=self.settings.thresh, i_dict_name="../../dictionaries/nomon_dict.txt")  
        #Debug
        self.disp = False
        self.ticker.disp = self.disp
        self.settings.click_distr.disp = self.disp
        self.settings.disp = self.disp
        self.disp_word_results = True
        self.dispExperimentHeading()
        #Plots 
        self.disp_plots = DispSettings()
        #Diagnostic all - compare against old results via print statements
        self.disp_all_diagnostic = False
        self.disp_some_diagnostic = False
        if self.disp_some_diagnostic:
            self.ticker_old = TickerOld(i_disp=self.disp_some_diagnostic)
        self.ticker.diagnostic=  self.disp_some_diagnostic
        
    def resetIfParamsChanged(self, i_user, i_c, i_n, i_delay, i_cur_overlap, i_cur_n_channels):
        cn = i_c['clicks'][i_n]
        delay_changed = np.abs( i_delay - cn['click_delay'] ) > 1E-6
        n_channels_changed = np.abs(i_cur_n_channels-i_c['nchannels'])>1E-6
        overlap_changed = np.abs( i_cur_overlap-i_c['overlap'])>1E-6
        (new_overlap, new_n_channels, new_delay) = (i_cur_overlap, i_cur_n_channels, i_delay)
        if  n_channels_changed or overlap_changed or delay_changed: 
            (new_delay, new_overlap) = (cn['click_delay'], i_c['overlap']) 
            (new_n_channels, letter_group) = (i_c['nchannels'],i_c['clicks'][0]['letter_group'])
            new_letter_loc = np.array(cn['observations'] + cn['click_delay']) + cn['click_time']
            if self.load_click_distr:
                self.ticker.train=False
                (channel, overlap) = (i_c['nchannels'],i_c['overlap'])
                file_name = "%shist_%.2d_%d_%.2f.cPickle" % (self.click_distr_dir , i_user, channel, overlap) 
                print "LOADING CLICK DISTR: file = ", file_name
                params = self.settings.utils.loadPickle(file_name)
                (hist, delay, std, fp_rate, fr) = params
                (self.settings.delay, self.settings.std, self.settings.fp_rate, self.settings.fr) = (delay, std, fp_rate, fr)
            distr_params = self.settings.getParams() 
            self.settings.resetClickDistr(new_n_channels, letter_group, new_overlap, distr_params, new_letter_loc)
            if self.load_click_distr:
                self.settings.click_distr.histogram = None
            else:
                self.settings.click_distr.initHistogram()
            self.ticker.setClickDistr(self.settings.click_distr)
            #Update the old ticker
            if self.disp_some_diagnostic:
                self.ticker_old.newWord(self.settings.click_distr.alphabet)   
        return (new_overlap, new_n_channels, new_delay) 
      
    def loadPhrases(self, i_user, i_channel):
        user_str = "%.2d" % i_user
        channel_str = "%d" % i_channel
        filename = file(os.sep.join([self.root_dir + user_str, 'channels' + channel_str, "phrases.txt"]))
        phrases = filename.read().split("\n")
        filename.close()
        return phrases[0:-1]
    
    def loadUserStats(self, i_user, i_channel, i_phrase_cnt, i_word_cnt):
        user_str = "%.2d" % i_user
        channel_str = "%d" % i_channel
        channel_dir = os.sep.join([self.root_dir + user_str, 'channels' + channel_str])
        file_name = "click_stats%.2d_%.2d.cPickle" % (i_phrase_cnt, i_word_cnt)
        file_name = os.sep.join([channel_dir , file_name])
        if not os.path.exists(file_name):
            return
        c = self.settings.utils.loadPickle( file_name)
        if not self.settings.speeds.has_key( str(c['overlap']) ):
            if self.disp:
                print "no speed settings for overlap of :" ,c['overlap']
            return 
        return c
   
    ################################################## Main 
    
    def computeUserListStats(self, i_users, i_channels):
        for user in i_users:
            for channel in i_channels:
                self.computeUserStats(user, channel)
        
    def computeUserStats(self,  i_user, i_channel):
        """Compute the errors etc per user and display on std, and return"""
        phrases = self.loadPhrases(i_user, i_channel) 
        for test_overlap in self.settings.speed_order:
            (overlap, n_channels, delay) = (0.0, 0, -np.inf) 
            for phrase_cnt in range(0,len(phrases)):
                words = phrases[phrase_cnt].split('_')
                for word_cnt in range(0, len(words)): 
                    c = self.loadUserStats(i_user, i_channel, phrase_cnt, word_cnt )
                    if c is None:
                        continue  
                    if not ( np.abs(c['overlap'] - float(test_overlap)) < 1E-6):
                        continue
                    word = self.getWord(words, word_cnt)
                    self.dispWordHeading(word, c)
                    (n_iter, letter_idx) = (0, 0) 
                    for n in range(0,len(c['clicks'])):
                        cn = c['clicks'][n]
                        (overlap,n_channels,delay) = self.resetIfParamsChanged(i_user, c,n,delay,overlap,n_channels)
                        if not (cn['letter_index'] == letter_idx):
                            (letter_idx, select_word, n_iter, is_error) = self.newLetter(n,word,cn,n_iter,c)
                            if select_word is not None:
                                break
                        #Store the click times for new ticker, and update old ticker
                        self.ticker.newClick(cn['click_time'])
                        if self.disp_some_diagnostic:
                            self.ticker_old.updateLetterScores(cn['scores'], self.settings.click_distr.alphabet, self.settings.long_alphabet)
                            if self.disp_all_diagnostic:
                                self.displayDiagnosticOldLetterOffsets( c, cn)
                    #Process the last few clicks if there are any left
                    if len(self.ticker.click_times) > 0: 
                        (letter_idx, select_word, n_iter, is_error) =  self.newLetter(n,word,cn,n_iter,c)
                    word_stats = self.newWord(overlap, select_word, word, is_error, n+1, n_iter, i_user, i_channel)
                    self.dispWordResults(word_stats, c)
                    #If all the words have been processed and no selection was make
                    if select_word is None:
                        self.ticker.undoLastLetter()
                    self.ticker.newWord()     
                    if self.disp_some_diagnostic:
                        self.ticker_old.newWord(self.settings.click_distr.alphabet)    
                    
    ####################################### New letter/word
    
    def newLetter(self, i_n, i_word, i_cn, i_n_iter, i_c):
        letter_idx  = self.ticker.warpIndices( i_cn['letter_index'] - 1,  len(i_word))
        if self.disp:
            (n_clicks_old, select_word_old, is_err_old) = (len(i_c['clicks']), i_c['selected_word'], i_c['word_error'])
            print "=============================================================================================="
            print "=============================================================================================="
            print "n = ", i_n,  " grnd_truth_word = ", i_word, " letter idx= ", letter_idx, " letter =", i_word[letter_idx] 
            print "Before: n_clicks = ",n_clicks_old, " select word = ",  select_word_old, " is error = ", is_err_old
        n_iter = i_n_iter + 1
        new_letter_idx = i_cn['letter_index']
        self.ticker.dispClickTimes(i_word,  i_c['selected_word'])
        letter_idx = new_letter_idx
        if self.disp_some_diagnostic:
            self.ticker_old.updateWordProbs(self.settings.click_distr.alphabet)
            if self.disp_all_diagnostic:
                self.displayDiagnosticOldLetterScores(np.array(self.ticker_old.letter_scores), i_c, i_cn)
        select_word = self.ticker.newLetter() #i_letter_scores=old_scores )
        self.ticker.dispBestWords()
        is_error = 1
        if select_word == i_word:
            is_error = 0
        if self.disp_some_diagnostic:
            self.ticker_old.newLetter(self.settings.click_distr.alphabet)
        return (letter_idx, select_word, n_iter,  is_error )
    
    def newWord(self, i_overlap, i_select_word, i_grnd_truth_word, i_is_error, i_n_clicks, i_n_iter, i_user, i_nchannels):
        if i_select_word is None:
            n_clicks = float(i_n_clicks) / len(i_grnd_truth_word)
        else:
            n_clicks = float(i_n_clicks) / len(i_select_word) 
        speed = self.settings.speeds[str(i_overlap)] 
        actual_speed = 60.0*len(i_grnd_truth_word) / (i_n_iter*speed*self.settings.word_length)
        est_speed = 60.0 / (speed*self.settings.word_length )
        min_edit_dist = 0.0 
        select_word = str(i_select_word)
        is_error = i_is_error
        if i_is_error:
            if i_select_word is None:
                """ See if we can't determine if this is real error or not 
                   *  if the user selected a wrong word in the prev experiment and a decsion  
                      in this experiment has not been made yet. """
                if i_n_iter < ( self.settings.n_repeat * len(i_grnd_truth_word)):
                    select_word = "*******"
                    min_edit_dist = None
                    is_error = None
                else:
                    select_word = ""
                    min_edit_dist = float(len(i_grnd_truth_word))
            else:
                #Wrong selection
                d =  MinEditDistance()
                min_edit_dist = d.compute(i_grnd_truth_word, select_word)
        params = (i_grnd_truth_word, select_word, n_clicks,est_speed,actual_speed, is_error, 
            min_edit_dist, i_overlap, i_n_clicks, i_n_iter, i_user, i_nchannels) 
        return params
    
    ################################################# Get
                       
    def getWord(self, i_words, i_word_cnt):
        word = i_words[i_word_cnt]
        if not (word == "."):
                word += "_"
        return word 
    
    ################################################# Disp
    
    def dispExperimentHeading(self):
        if not self.disp_word_results:
            return
        g0 = "%User | Grnd truth    |" 
        g1 = "{0:{1}}".format( "%s" % " Selected word", 29 ) + "|"
        g2 = g0 + g1 + " cpc  | Est wpm | wpm  |"
        g3 = "{0:{1}}".format( "%s" % " IsErr ", 12 ) + "|"
        g4 = g2 + g3 + " eDist | Overlap |"
        g5 =  "{0:{1}}".format( "%s" % " Clicks ", 11 ) + "|"
        g6 =  "{0:{1}}".format( "%s" % " NIter ", 11 ) + "|"
        g7 = g4 + g5 + g6 + " Ng | C | \n"
        disp_str = str(g7)
        g0 =  "{0:{1}}".format( "%s" % "%", 22 ) + "|"
        g1 =  "{0:{1}}".format( "%s" % " bef ", 14 ) + "|"
        g2 =  "{0:{1}}".format( "%s" % " aft ", 14 ) + "|"
        g3 =  "{0:{1}}".format( "%s" % "", 23 ) + "|"
        g4 =  "{0:{1}}".format( "%s" % " bef ", 6 ) + "|"
        g5 =  "{0:{1}}".format( "%s" % " aft ", 5 ) + "|"
        g6 =  "{0:{1}}".format( "%s" % "", 17 ) + "|"
        g7 =  "{0:{1}}".format( "%s" % " bef ", 4 ) + "|"
        g8 =  "{0:{1}}".format( "%s" % " aft ", 3 ) + "|"
        g9 =  "{0:{1}}".format( "%s" % " bef ", 4 ) + "|"
        g10 =  "{0:{1}}".format( "%s" % " aft ", 3 ) + "|"
        disp_str += (g0+g1+g2+g3+g4+g5+g6+g7+g8+g9+g10)
        print disp_str
                     
    def dispWordHeading(self, i_word, i_c):
        if not self.disp:
            return
        print "******************************************************"
        print "Experiment Old Results : "
        print "selected word = ",  i_c['selected_word'], " grnd truth = ", i_word
        print "n_channels = ", i_c['nchannels']
        print "n_clicks = ", len(i_c['clicks'])
        print "overlap = ", i_c['overlap']
        print "is_error = ", i_c['word_error']
    
    def dispWordResults(self, i_params, i_c):
        (grnd_truth_word, select_word, clicks_per_char,est_speed,  actual_speed,is_error, min_edit_dist, 
            overlap, n_clicks, n_iter, user, n_channels) = i_params 
        if not self.disp_word_results:
            return 
        g0 =  "{0:{1}}".format( "%d" % user, 6 ) + "|"
        g1 =  "{0:{1}}".format( "%s" % grnd_truth_word, 14 ) + "|"
        #Selected word and error
        (tmp_word, err_before) = (i_c['selected_word'], False)
        if not(i_c['selected_word'] == grnd_truth_word):
            (tmp_word, err_before) = (str(i_c['selected_word']), True)
        elif i_c['word_error']:
            (tmp_word, err_before) = ( "", True )
        g2_1 =  "{0:{1}}".format( "%s" % tmp_word, 13 ) + "|"
        g2_2 =  "{0:{1}}".format( "%s" % select_word, 13 ) + "|"
        g3 =  "{0:{1}}".format( "%.2f" % clicks_per_char, 5 ) + "|"
        g4 =  "{0:{1}}".format( "%.2f" % est_speed, 8 ) + "|"
        g5 =  "{0:{1}}".format( "%.2f" % actual_speed, 5 ) + "|"
        #Is error
        g6_1 =  "{0:{1}}".format( "%d" % err_before, 5 ) + "|"
        g6_2 =  "{0:{1}}".format( "%s" % self.dispVal(is_error, "%d"), 4 ) + "|"
        g7 =  "{0:{1}}".format( "%s" % self.dispVal(min_edit_dist, "%d"), 6) + "|"
        g8 =  "{0:{1}}".format( "%.2f" % overlap, 8) + "|"
        #Nclicks
        g9_1 =  "{0:{1}}".format( "%d" %  len(i_c['clicks']), 4) + "|"
        g9_2 =  "{0:{1}}".format( "%d" %  n_clicks, 4) + "|"
        #Niterations
        g10_1 =  "{0:{1}}".format( "%d" %  i_c['niterations'], 4) + "|"
        g10_2 =  "{0:{1}}".format( "%d" %  n_iter, 4) + "|"
        g11 =  "{0:{1}}".format( "%d" % len(grnd_truth_word), 3) + "|"
        g12 =  "{0:{1}}".format( "%d" % n_channels, 2) + "|"
        print "%s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s" % (g0,g1,g2_1,g2_2,g3,g4,g5,g6_1,g6_2,g7,g8,g9_1,g9_2,g10_1,g10_2,g11, g12)  
 
    def dispVal(self, i_val, i_type):
        if i_val is None:
            val_str = "-"
        else:
            val_str = i_type % i_val 
        return val_str
    
    
    ######################################################## Diagnostic
    #Relating current experiment to previous results
    def displayDiagnosticOldLetterOffsets(self, i_c, i_cn):
        click_delay_old = i_cn['click_delay']
        letter_offsets_old = i_cn['observations'] + click_delay_old +  i_cn['click_time']
        new_letter_offsets = np.array(self.settings.click_distr.loc)
        long_alphabet = np.array(self.settings.long_alphabet)
        obs = np.array( i_cn['observations'])
        hist_eval = self.ticker_old.evalScore(obs)
        old_scores = np.array(i_cn['scores'])
        print "TICKER OLD LETTER OFFSET COMPARISON"
        for (n, letter) in enumerate(self.settings.click_distr.alphabet):
            idx = np.nonzero(long_alphabet == letter)[0]
            diff = letter_offsets_old[idx] -  new_letter_offsets[n,:]
            old_offset = self.settings.utils.stringVector( letter_offsets_old[idx], i_type="%.3f")
            new_offset = self.settings.utils.stringVector( new_letter_offsets[n,:], i_type="%.3f")
            diff_str = self.settings.utils.stringVector( diff , i_type="%.3f")
            hist_score = self.settings.utils.stringVector( hist_eval[idx], i_type="%.3f")
            old_hist_score =  self.settings.utils.stringVector( old_scores[idx], i_type="%.3f")
            diff_scores =  self.settings.utils.stringVector( hist_eval[idx] - old_scores[idx], i_type="%.3f")
            obs_str = self.settings.utils.stringVector( obs[idx] , i_type="%.3f")
            print "letter=", letter, " old offset=", old_offset, " offset diff=", diff_str, 
            print " hist_score = ", hist_score, " diff_scores = ", diff_scores, " obs = ", obs_str
        
    def displayDiagnosticOldLetterScores(self, i_old_scores, i_c, i_cn):
        print "LETTER SCORE COMPARISON"
        new_scores = self.settings.click_distr.logLikelihood(self.ticker.click_times, i_log=True)
        for (n, letter) in enumerate(self.settings.click_distr.alphabet):
            diff =  i_old_scores[n] - new_scores[n]
            if diff > 0:
                exp_diff = np.exp(-diff)
            else:
                exp_diff = np.exp(diff)
            print "letter =  ", letter , " old score = ", i_old_scores[n], " new score = ", new_scores[n],
            print " diff = ", diff, " exp diff = ",  exp_diff
        
if __name__ ==  "__main__":
    results = TickerChannelResults()
    users = [7,8,9]  
    channels = [3,4,5] 
    results.computeUserListStats(users, channels)
    p.show()

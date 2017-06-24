

import sys,  cPickle, time, itertools
import numpy as np
import pylab as p
sys.path.append("../../")
sys.path.append("../")
from channel_config import  ChannelConfig, AlphabetTimes
from utils import Utils,  WordDict, DispSettings, PhraseUtils
from click_distr import ClickDistribution
from ticker_core import TickerCore
from ticker_sampling import TickerSampling
from min_edit_distance import MinEditDistance
 
class TickerSimulation():
    ################################################ Initialisation
    def __init__(self):
        self.ticker = TickerCore(i_select_thresh=0.9,  i_dict_name="../../dictionaries/nomon_dict.txt") 
        self.sampler = TickerSampling()   
        self.disp_settings = DispSettings()
        self.utils = Utils()
        self.phrase_utils = PhraseUtils()
        (channels, overlap, file_length, root_dir) = (5, 0.65, 0.21, "../../")
        self.channel_config  = ChannelConfig(channels, overlap, file_length, root_dir)
        self.click_distr = ClickDistribution()
        self.dist_measure = MinEditDistance()
        #Display parameters
        self.disp=True
        self.debug=False
        self.disp_settings.params = { 'axes.labelsize':20, 'xtick.labelsize': 20,'ytick.labelsize': 20, 
                'axes.titlesize':20, 'text.usetex': True,  'text_font': 15 }
        p.rcParams.update(self.disp_settings.params)
        #Some other parameters used everywhere:
        self.word_length = 5.0
        
    def initSimulation(self, i_sentence, i_params ):
        #The parameters
        (ticker_params, simulation_params) = i_params
        #Set the noise parameters 
        self.setNoise(ticker_params)
        #Compute word for word
        words = self.phrase_utils.wordsFromSentece( i_sentence )  
        #The simulation parameters
        (nrepeat, nsamples_sentence, nmax) = simulation_params
        #Total counts
        tot_error = np.zeros(len(words)) #Total number of word errors
        speed = np.zeros([nsamples_sentence,len(words)]) #Total scans to write a word
        nclicks = np.zeros([nsamples_sentence,len(words)]) #Total number of true clicks received
        error_rate = np.zeros([nsamples_sentence,len(words)]) #Character error rate
        word_select_lengths =  np.zeros([nsamples_sentence,len(words)]) #The length of the selected words
        #Diagnostic: all the generated click times involved in the simulation
        words = self.phrase_utils.wordsFromSentece(i_sentence)
        click_times = [  [ [ []  for letter in self.phrase_utils.getWord(w)]  for w in words  ] for n in range(0,nsamples_sentence)]
        return (tot_error, speed, nclicks, error_rate,  words, click_times, word_select_lengths )
    
    def initWordResults(self, i_grnd_truth):
        word_click_times = [[] for n in range(0,len(i_grnd_truth))] 
        self.ticker.newWord()
        return word_click_times
    
    def defaultParams(self):
        ticker_params = self.defaultTickerParams() 
        simulation_params = self.defaultSimulationParams()
        return (ticker_params, simulation_params)

    def defaultTickerParams(self):
        #Default parameters for experiments without waiting time at the end
        nchannels = 5
        #The length of the voice file
        file_length = 0.21
        #Click distr params
        (delay, std, fr, fa_rate) = (0.2, 0.05, 0.3, 0.01) 
        root_dir = "../../"
        #Waiting time at end
        end_delay = 0.3 
        #Overlap
        overlap = 0.65
        #Extra wait time at end
        extra_wait_time = 0.3
        params = (nchannels, file_length, delay, fa_rate , fr ,  overlap,  end_delay,  std, root_dir) 
        return params

    def defaultSimulationParams(self):
        #The number of times any word is allowed to repeat 
        nrepeat = 2                       
        #The number of samples per sentence - until an outcome is reached
        nsamples_sentence = 10   
        """The total number of times the user is allowed to repeat a sentence - this includes the 
        case where no true positives or negatives are sampled"""
        nmax = 5
        params = (nrepeat, nsamples_sentence, nmax)
        return params 
      
    def setNoise(self, i_params, i_adapt_end_delay=False):
        (nchannels, file_length, delay, fp_rate , fr ,  overlap,  end_delay,  std, root_dir) = i_params
        if std is None:
            std = self.overlapToStd(overlap)
        else: 
            std = std
        self.channel_config  = ChannelConfig(nchannels, overlap, file_length,  root_dir)
        self.click_distr.clear()
        self.click_distr.clearHistogram()
        (is_train,learning_rate) = (False, 1.0)
        (learn_delay, learn_std, learn_fp, learn_fr) = (False, False, False, False)
        self.click_distr.setParams(is_train, self.channel_config, delay, std, fp_rate, fr, learning_rate, end_delay)
        self.click_distr.setFixLearning(learn_delay, learn_std, learn_fp, learn_fr)
        self.click_distr.clearHistogram()
        self.ticker.setClickDistr(self.click_distr)
        disp_params =  (delay, std, fr, fp_rate, self.click_distr.T) 
        self.utils.printParams(disp_params, i_heading_str="New Simulation Parameters ")
 
    ##################################### Main 

    def compute(self, i_n_channels, i_file_length, i_sentence, i_nsamples,  i_nrepeat, i_nmax, i_overlap, i_delay, i_std, i_fr, i_fp_rate, i_display ):
        #Parameters
        end_delay = i_delay + 3.0*i_std
        ticker_params = (i_n_channels, i_file_length, i_delay, i_fp_rate, i_fr, i_overlap, end_delay, i_std, "../../") 
        simulation_params = (i_nrepeat, i_nsamples, i_nmax) #nsamples = samples per sentence
        params = (ticker_params, simulation_params)
        #Display for debugging purposes
        (disp_sentences, disp_top_words, disp_word_errors, save_click_times) = (True, False, False, False) 
        disp_settings = (disp_sentences, disp_top_words, disp_word_errors, save_click_times)
        #The results
        r = self.sentenceResults(i_sentence, params, disp_settings)
        (tot_error, speed, nclicks, error_rate,  click_times, word_lengths) = r
        r_normalised = self.normaliseResults(r)
        print "****************************************************************************************************************************"
        print "Final results Ticker"
        print "****************************************************************************************************************************"
        self.utils.dispResults(r_normalised)
        return r_normalised
     
    def sentenceResults(self, i_sentence, i_params, i_disp_settings ):
        """ * Return a simulated estimate of how long it will take on average to write 
                a sentence, what the error rate will be and the clicks per character
            * Results are returned as an array of nsamples x words.""" 
        #Diagnostic
        (ticker_params, simulation_params) = i_params
        (disp_sentence, disp_words, disp_top_words, save_click_times) = i_disp_settings
        #Initialise the simulation
        (tot_error, speed, nclicks, error_rate, words, click_times, word_select_lengths) = self.initSimulation(i_sentence, i_params)
        (nrepeat, nsamples_sentence, nmax) = simulation_params
        for n in range(0, nsamples_sentence):
            #Special treatment for punctuation marks
            for m in range(0, len(words)):
                grnd_truth = self.phrase_utils.getWord(words[m]) 
                word_select_lengths[n,m] = len(grnd_truth)
                word_results = self.wordResults(grnd_truth, nrepeat, nmax, i_disp_settings)
                (is_error, speed[n, m], click_times[n][m], selected_word,  nclicks[n, m])  = word_results
                if selected_word is not None:
                    selected_word =  self.phrase_utils.getWord(selected_word)
                    word_select_lengths[n,m] = len(selected_word)
                msg ="sentence sample=%.2d of %d, word num=%d, grnd truth word=%s, selected  word=%s, len=%d, err=%d   " % (n+1,nsamples_sentence, m+1,grnd_truth,selected_word, word_select_lengths[n,m], is_error)
                self.utils.dispMsg(disp_words, msg, disp_sentence)
                #If an error occurred the sentence could not be written - bail out
                if is_error:
                    tot_error[m] += 1
                    if selected_word is None:
                        error_rate[n, m] = 100.0
                        speed[n, m] = 0.0
                    else:
                        error_rate[n, m] = self.dist_measure.compute(grnd_truth, selected_word)
                        error_rate[n, m] = (100.0*float(error_rate[n, m]) / len(grnd_truth))
        return  (tot_error, speed, nclicks, error_rate,  click_times, word_select_lengths)
  
    def wordResults(self, i_grnd_truth, i_nrepeat, i_nmax, i_disp_settings):
        """Generate click times for each letter in the input word, update posteriors and then classify"""
        #The max time allowed to spend on a word, including false rejections
        max_time_word = i_nmax*(len( i_grnd_truth ))  
        #The max time allowed to spend on a word, excluding false rejections
        max_nrepeat = i_nrepeat*(len( i_grnd_truth ))  
        #Initialise the word simulation
        word_click_times= self.initWordResults(i_grnd_truth)
        nclicks_received = 0
        #Diagnostic
        (disp_sentence, disp_words, disp_top_words, save_click_times) = i_disp_settings
        #The selected word
        selected_word = None
        #The number of true clicks
        true_clicks = 0.0
        for sample_step in range(0, max_time_word):
            if disp_words:
                print "===================================================================="
            #Diagnostic
            ts  = time.time()
            #Total time it took
            total_time = sample_step + 1
            #The target letter 
            target_idx = self.ticker.warpIndices(self.ticker.letter_idx, len(i_grnd_truth))
            target_letter = i_grnd_truth[target_idx]
            #Sample some click times, and compute their scores
            (click_times, N, C) = self.sampler.sample(self.click_distr, target_letter, i_return_labels=False,  
                i_n_rejection_samples=1E3, i_display=disp_words)
            #If not clicks have been received go to the next opportunity
            if (N + C) < 1:
                continue
            nclicks_received += 1
            #A click was received in this iteration, but the a system error is generated
            if nclicks_received > max_nrepeat:
                msg = "-- is_error = 1,  nrepeat iterations,  word =%s, max_time=%d "% (i_grnd_truth, max_nrepeat)
                self.utils.dispMsg(False, msg, disp_words)
                return (True, total_time, word_click_times, selected_word, true_clicks)
            true_clicks += C
            #Plot word samples - if 2 clicks were received
            if save_click_times:
                word_click_times[target_idx].append(np.array(click_times))
            (selected_word, is_error, cur_prob) = self.updateTicker(click_times, i_disp_settings, sample_step,  target_letter, ts, i_grnd_truth)
            #Classification step
            if selected_word is not None:
                msg = "-- is_error=%d, time=%d, cur_prob=%.3f, word=%s, max_time=%d"  % (is_error, sample_step + 1, cur_prob, i_grnd_truth, max_time_word) 
                self.utils.dispMsg( False, msg, disp_words)
                return (is_error, total_time, word_click_times, selected_word, true_clicks)
        #An error state if the user has iterated through the word more than nmax times
        total_time = max_time_word
        msg = "-- is_error = 1,  not processed in time,  word =%s, max_time=%d "% (i_grnd_truth, total_time)
        self.utils.dispMsg(False, msg, disp_words)
        return (True, total_time, word_click_times, selected_word, true_clicks)
    
    def updateTicker(self, i_click_times, i_disp_settings, i_sample_step, i_target_letter, i_ts, i_grnd_truth):
        #Diagnostic
        (disp_sentence, disp_words, disp_top_words, save_click_times) = i_disp_settings
        #Initialisation
        #Add the clicks to ticker
        for click_time in i_click_times:
            click_scores = self.ticker.newClick(click_time)
        #Not going to train or reset anything
        selected_word = self.ticker.newLetter(i_process_word_selections=False)
        (best_word, best_prob) = self.ticker.getBestWordProbs(1)
        cur_prob = np.exp(self.ticker.dict.log_probs[self.ticker.word_indices[i_grnd_truth]])
        #Update the display - diagnostic
        if disp_words:
            letter_idx = self.ticker.letter_idx - 1
            self.updateSimulationDisplay( disp_top_words, click_scores, i_sample_step, letter_idx, 
                i_target_letter, i_click_times, self.ticker.dict.log_probs, i_ts, i_grnd_truth, cur_prob, best_word, best_prob)
        is_error = False
        if selected_word is not None:
            cmpr_word = self.phrase_utils.getWord(selected_word)
            is_error =  not(cmpr_word == i_grnd_truth)
        return (selected_word, is_error, cur_prob)
 
    def normaliseResults(self, i_results):
        (tot_error, speed, nclicks, error_rate,  click_times,  word_lengths) = i_results
        avg_cpc = np.mean(np.mean(nclicks / word_lengths, axis=0))
        std_cpc = np.mean(np.std(nclicks / word_lengths, axis=0))
        scan_time = self.click_distr.T
        print "Normalising the results: scan time = ", scan_time, " s"
        [row, col] = np.nonzero(np.abs(speed) > 1E-3)
        wpm = np.array(speed)  
        wpm[row,col] = 60.0/(scan_time * wpm[row,col]) * ( word_lengths[row,col] / self.word_length)
        avg_wpm = np.mean(np.mean(wpm, axis=0))
        std_wpm = np.mean(np.std(wpm,axis=0))
        avg_err_rate = np.mean(np.mean(error_rate, axis=0))
        std_err_rate = np.mean(np.std(error_rate,axis=0))
        return (avg_cpc, std_cpc, avg_wpm, std_wpm, avg_err_rate, std_err_rate)
    
    ##################################### Display
 
       
    def updateSimulationDisplay(self, i_disp_top_words, click_scores, sample_step, letter_idx, 
        target_letter, click_times, log_priors, ts, i_target_word, i_target_prob, 
        i_best_word, i_best_prob):
        """Print the result after some clicks were received: Find the maximum word prob etc."""
        max_idx_letter = np.argmax(click_scores)
        print "step=%d, letter_idx=%d," % (sample_step,letter_idx),
        print " target_letter=%s, M=%d," % (target_letter, len(click_times)), 
        best_letter = self.click_distr.alphabet[max_idx_letter]
        print " best_letter=%s, score=%.2f," % (best_letter, click_scores[max_idx_letter]),  
        print " target_word=%s, prob=%.3f," %(i_target_word, i_target_prob), 
        print " best_word=%s, best_prob=%.3f, time=%.0fms" % (i_best_word, i_best_prob, 1000.0*(time.time()-ts))
        if i_disp_top_words:
            (top_words,  scores) = self.ticker.getBestWordProbs(10)
            sum_all = np.sum(np.exp(self.ticker.dict.log_probs))
            print "Top words : ",  top_words, 
            print ", sum (word_priors)= %.3f, ,  sum(top words) =%.3f "   %(sum_all, np.sum(scores))
             
    def dispSentenceResultsRaw(self, i_r):
        (tot_error, speed, nclicks, error_rate,  click_times, word_lengths) = i_r
        self.utils.dispMsg(True, "Final unnormalised results:")
        self.utils.printMatrix(np.atleast_2d(tot_error), "total error", i_precision="%.3f" )
        self.utils.printMatrix(speed, "speed", i_precision="%d" )
        self.utils.printMatrix(nclicks, "nclicks", i_precision="%d" )
        self.utils.printMatrix(error_rate, "error_rate", i_precision="%.2f" )
        self.utils.printMatrix(word_lengths, "word lengths", i_precision="%d" )
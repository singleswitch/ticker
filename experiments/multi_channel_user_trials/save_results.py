
import sys
sys.path.append("../../")
sys.path.append("../")
from utils import Utils
import numpy as np
import pylab, os
from min_edit_distance import MinEditDistance
from click_distr import ClickDistribution
from numpy.core import multiarray as multiarray
import cPickle

class TickerChannelResults(object):
    def __init__(self, i_is_students):
        #Settings are different for the case stud
        if i_is_students:
            self.root_dir = "../../../user_trials/multi_channel_experiment/students/"
            self.speeds = {'0.12' : 12.5, '0.10': 11.2, '0.08':  9.0 }
            self.speeds_ids = {'0.12' : "S", '0.10': "M", '0.08': "F" }
            self.speed_order = ['0.12','0.10','0.08']
            self.file_out = "results/student_audio_results.cPickle"
            self.users = [7,8,9,10,11,12,13,15,16,17,18,19]
            self.channels = [3,4,5]
        else:
            self.root_dir = "../../../user_trials/multi_channel_experiment/case_study/results-0.0.0.0/"
            self.speeds = {'0.12' : 9.0 }
            self.speeds_ids = {'0.12' : "F"}
            self.speed_order = ['0.12']
            self.file_out = "results/case_study_audio_results.cPickle"
            self.users = [10] #7,10]
            self.channels = [5]
        self.dist_measure = MinEditDistance()
        self.wpm = {}
        self.word_length = 5.0
        for key in self.speed_order:
            self.wpm[key] = 1.0 / (self.speeds[key]*self.word_length / 60.0)
        self.utils = Utils()
        
    ####################################### Load Functions
    def loadPhrases(self, i_user, i_channel):
        output_dir =  "%suser%.2d/channels%d/" %(self.root_dir, i_user, i_channel)
        filename =file(output_dir + "phrases.txt")
        phrases = filename.read().split("\n")
        filename.close()
        return phrases[0:-1]
    
    def loadUserClickStats(self, i_user, i_channel, i_phrase_cnt, i_word_cnt):
        """Values needed for word error rate computation
        * click_stats['overlap']: audio speed (overlap between letters)
        * click_stats['nchannels']: number of channels
        * click_stats['selected_word']: word selection
        * click_stats['word_error']: was word selection an error? I.e. program could not converged
        * click_stats['top_ten_words']: top ten words for the selection: Tuple of (best_words, best_probs)
        * click_stats['niterations']: The number of alphabet iterations to make the word selection. 
        * click_stats['time']: "Wall clock time it took to make the word selection"""
         
        """Values needed for click distribution analysis: 
        *Clicks is a list: Each item in this list contains a dictionary entry:
        * clicks_stats['letter_group']: letters configuration used 
        * clicks_stats['scores']: PDF scores of all letters
        * clicks_stats['observations']: x - the scores are P(x|model for each letter in lettergroup)
        * clicks_stats['click_time']: Time of click
        * clicks_stats['click_delay']: Mean value of distribution (same for all letter)   
        * clicks_stats['letter_index']: click number
        * clicks_stats['letter_state']: clicking on letters or words?
        * click_stats['min_val']: minimum allowed probablity for a letter"""
        output_dir =  "%suser%.2d/channels%d/" % (self.root_dir, i_user, i_channel)
        filename =  "%sclick_stats%.2d_%.2d.cPickle" % (output_dir, i_phrase_cnt, i_word_cnt) 
        if not os.path.exists(filename):
            return None
        click_stats = self.utils.loadPickle(filename)
        if click_stats.has_key("clicks"):
            clicks = list(click_stats['clicks'])
            del click_stats['clicks']
        else:
            clicks = None
        return (click_stats, clicks)

    def phraseStats(self, i_phrases, i_display=False ):
        #Return the number of phrases
        letters = "".join(i_phrases)
        letters = np.array([letters[n] for n in range(0, len(letters))])
        alphabet=np.array(['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','_','.']) 
       
        for letter in alphabet:
            n = len(np.nonzero(letters == letter)[0])
            if i_display:
                print "Number of letter ", letter, " :", n
        
        if i_display:
            print "Number of Phrases: ", len(i_phrases)
            print "Number of letters : ", len(letters)
            print "Estimated time (13s per letter) = ", len(letters) * 13.0 / 60.
        return len(i_phrases)
    
    ################################ Get
    

    def getGroundTruthWord(self, i_phrases, i_phrase_cnt, i_word_cnt):
        cur_phrase = i_phrases[i_phrase_cnt]
        phrase_words = cur_phrase.split("_")[0:-1]
        phrase_words.append('.')
        if i_word_cnt >= len(phrase_words):
            return None
        o_word = phrase_words[i_word_cnt]
        if (not (o_word == ".")) and (not (o_word == "")): 
            o_word += "_"
        return o_word
    
    ################################# User results
    
    def avgUserStats(self, i_display=False):
        """Compute the average user stats"""
        results = {}  
        for user in self.users:
            for channel in self.channels:
                user_results = self.avgUserChannelResults(user, channel, i_display=False )
                for overlap in self.speed_order:
                    (speed_id, speed_bound, alphabet_time, n_correct, n_total, perc_correct, n_correct_top_three, 
                    perc_correct_top_three, n_errors, perc_errors, n_error_select, n_error_top_three,
                    n_error_time_out, perc_error_select, perc_error_top_three, perc_time_out, 
                    min_edit_dist ) = self.userChannelOverlapStats(user_results, overlap)
                    if not results.has_key(channel):
                        results[channel] = {}
                    if not results[channel].has_key(overlap):
                        results[channel][overlap] = {}
                        results[channel][overlap]['speed_bound'] = 0.0
                        results[channel][overlap]['wpm' ] = 0.0
                        results[channel][overlap]['n_correct'] = 0.0
                        results[channel][overlap]['n_total'] = 0.0
                        results[channel][overlap]['n_top_three'] = 0.0
                        results[channel][overlap]['n_select_errors'] = 0.0
                        results[channel][overlap]['n_time_out_errors'] = 0.0
                        results[channel][overlap]['min_edit_dist'] = 0.0
                        results[channel][overlap]['clicks_per_char'] = 0.0
                        
                    results[channel][overlap]['speed_bound'] += speed_bound
                    results[channel][overlap]['wpm' ] += alphabet_time
                    results[channel][overlap]['n_correct'] += n_correct
                    results[channel][overlap]['n_total'] += n_total
                    results[channel][overlap]['n_top_three'] += n_correct_top_three
                    results[channel][overlap]['n_select_errors'] += n_error_select
                    results[channel][overlap]['n_time_out_errors'] += n_error_time_out
                    
                    results[channel][overlap]['min_edit_dist'] = min_edit_dist
                    clicks_per_char = user_results[overlap]['clicks_per_char'] / n_correct
                    results[channel][overlap]['clicks_per_char'] += clicks_per_char
                    if i_display:
                        print "%.2d & %.2d & %s & %.2f & %.2f & %.2d & %.2d & %.2d & %.2d & %.2d & %.2f & %.2f \\\\ \hline" %(user,
                        channel,  self.speeds_ids[overlap], speed_bound, alphabet_time, n_total, n_correct,
                        n_correct_top_three, n_error_select, n_error_time_out, clicks_per_char , min_edit_dist) 
        return results
    
    def userStats(self, i_display=False, i_display_min_iter=False):
        self.dispHeading()
        #Initialise all the output results - results contain per word results
        (results, wpm, error_rate, clicks) = ({},{},{},{})  
        for user in self.users:
            results[str(user)] = {}
            for channel in self.channels:
                results[str(user)][str(channel)] = {}
                (wpm[str(channel)], error_rate[str(channel)], clicks[str(channel)]) = ({},{},{})
                for overlap in self.speed_order:
                    results[str(user)][str(channel)][overlap] = {}
                    wpm[str(channel)][overlap] = []
                    error_rate[str(channel)][overlap] = []
                    clicks[str(channel)][overlap] = []
        #Extract the results
        for user in self.users:
            for channel in self.channels:
                self.n_min = 0   #The number of times written in minimum number of iterations
                self.n_total = 0 #The total number of words written
                for overlap in self.speed_order:
                    results[str(user)][str(channel)][overlap] = self.userChannelResults( user,  channel, overlap, i_display )
                    wpm_data = results[str(user)][str(channel)][overlap]['wpm']
                    error_rate_data = results[str(user)][str(channel)][overlap]['error rate']
                    click_data = results[str(user)][str(channel)][overlap]['clicks per char'] 
                    if len(self.users) > 1:
                        wpm_data = np.mean(wpm_data)
                        error_rate_data = np.mean(error_rate_data)
                        click_data = np.mean(click_data)
                    #print "USER = ", user, " channel = ", channel, " overlap = ", overlap,
                    #print " DATA  = ", " data mean = ", np.mean(wpm_data)
                    wpm[str(channel)][overlap].extend([wpm_data])
                    error_rate[str(channel)][overlap].extend([error_rate_data])
                    clicks[str(channel)][overlap].extend([click_data])
                #Display the percentage of words written in minimum number of iterations.
                if i_display_min_iter:
                    print "user=%d channel=%d speed=%s: " % (user, channel, overlap),
                    print "total words=%d, total words <= min_iterations=%d, %2.2f" % (self.n_total,self.n_min,100.0*self.n_min/self.n_total)
        self.utils.savePickle( (wpm, error_rate,clicks), self.file_out)        
  

    ################################## User-channel stats
   
    def userChannelResults(self, i_user, i_channel, i_overlap, i_display=False ):
        """Accummulate all the stats"""
        phrases = results.loadPhrases(i_user, i_channel)
        n_phrases = self.phraseStats(phrases)
        user_results = {}
        for phrase_cnt in range(0, n_phrases):
            for word_cnt in range(0, len(phrases[phrase_cnt])):
                user_stats = self.loadUserClickStats(i_user, i_channel, phrase_cnt, word_cnt)
                if user_stats is not None:
                    (click_stats, clicks) =  user_stats
                    grnd_truth_word = results.getGroundTruthWord(phrases, phrase_cnt, word_cnt)
                    if grnd_truth_word is None:
                        #This continue will happen in some cases where a space was inserted after the sentence
                        continue
                    overlap =  "%.2f"  %(click_stats['overlap'])
                    if not(i_overlap == overlap):
                        continue
                    if not self.speeds_ids.has_key( overlap ):
                        #This can happen if the user tried a speed not meant for the experiment
                        continue
                    selected_word = click_stats['selected_word']
                    if selected_word == "_":
                        selected_word = "" 
                    is_error = True
                    if not user_results.has_key('wpm'):
                        user_results['wpm']  = []
                        user_results['error rate']  = []
                        user_results['clicks per char'] = [] 
                    n_chars = float(len(grnd_truth_word))
                    alphabet_time = click_stats['niterations']*self.speeds[overlap]
                    n_alphabet_words = float(len(selected_word)) / self.word_length
                    if len(selected_word) < 1:
                        wpm = 0
                    else:
                        wpm = 60.0/(self.word_length*alphabet_time / float(len(selected_word)))
                    #Store the number of iterations where clicks were received and the number of clicks
                    if clicks is not None:
                        click_iterations = 1
                        cur_idx = 0 
                        for m in range(0, len(clicks)):
                            letter_idx =  clicks[m]['letter_index']
                            if not(letter_idx == cur_idx):
                                cur_idx = letter_idx
                                click_iterations += 1
                        nclicks = float(len(clicks))
                    else:
                        click_iterations = None
                        nclicks = float(click_stats['nclicks'])
                    if click_stats['niterations'] <= n_chars:
                        self.n_min += 1
                    self.n_total += 1 
                    #See if timeout error occurs - if so wpm=0, and the number of clicks per character is irrelevant
                    cpc = nclicks / float(len(grnd_truth_word))
                    if (selected_word == grnd_truth_word) or (selected_word == ""):
                        is_error = click_stats['word_error']
                        if is_error:
                            word_length = float(len(grnd_truth_word))
                            user_results['error rate'].append(100.0)
                            #user_results['clicks per char'].append(None)
                            #user_results['wpm'].append(None)
                            user_results['clicks per char'].append(cpc)
                            user_results['wpm'].append(0.0)
                        else:
                            user_results['clicks per char'].append(cpc)
                            user_results['error rate'].append(0.0)
                            user_results['wpm'].append(wpm) #In wpm
                    else:
                        user_results['clicks per char'].append(nclicks /float(len(grnd_truth_word)))
                        user_results['wpm'].append(wpm) #In wpm
                        is_error = True
                        edit_dist = self.dist_measure.compute(grnd_truth_word, selected_word)
                        if np.abs(edit_dist) > 0.0:
                            edit_dist /= float(len(grnd_truth_word))
                            edit_dist *= 100 
                        user_results['error rate'].append(edit_dist)
                        
                    #Display
                    if i_display or (not click_iterations == click_stats['niterations']):
                        tot_time = None
                        if click_stats.has_key("time"):
                            tot_time = click_stats['time']
                        vals = (i_user, grnd_truth_word, selected_word, is_error, click_stats['niterations'], 
                            click_iterations, self.speeds[overlap], alphabet_time,  tot_time, wpm, 
                            user_results['error rate'][-1], user_results['clicks per char'][-1])
                        self.dispVals(vals)
        return user_results
        
    def avgUserChannelResults(self, i_user, i_channel, i_display=False ):
        phrases = results.loadPhrases(i_user, i_channel)
        n_phrases = self.phraseStats(phrases)
        user_results = {}
        for phrase_cnt in range(0, n_phrases):
            for word_cnt in range(0, len(phrases[phrase_cnt])):
                user_stats = self.loadUserClickStats(i_user, i_channel, phrase_cnt, word_cnt)
                if user_stats is not None:
                    (click_stats, clicks) =  user_stats
                    grnd_truth_word = results.getGroundTruthWord(phrases, phrase_cnt, word_cnt)
                    if grnd_truth_word is None:
                        #This continue will happen in some cases where a space was inserted after the sentence
                        continue
                    overlap =  "%.2f"  %(click_stats['overlap'])
                    if not self.speeds_ids.has_key( overlap ):
                        #This can happen if the user tried a speed not meant for the experiment
                        continue
                    selected_word = click_stats['selected_word']
                    is_error = True
                    if not user_results.has_key(overlap):
                        user_results[overlap] = {}
                        user_results[overlap]['n_correct'] = 0
                        user_results[overlap]['n_total']  = 0.0
                        user_results[overlap]['n_error_select']  = 0
                        user_results[overlap]['n_error_timeout']  = 0
                        user_results[overlap]['n_error_top_three'] = 0
                        user_results[overlap]['alphabet_time']  = 0.0
                        user_results[overlap]['min_edit_dist']  = 0.0
                        user_results[overlap]['clicks_per_char'] = 0.0 
                        user_results[overlap]['nchars'] = 0.0 
                    user_results[overlap]['nchars'] += float(len(grnd_truth_word))
                    if selected_word == grnd_truth_word:
                        user_results[overlap]['n_correct'] += int(not(click_stats['word_error']))
                        user_results[overlap]['n_error_timeout']  += int(click_stats['word_error'])
                        is_error =   click_stats['word_error']
                        alphabet_time = click_stats['niterations']*self.speeds[overlap] / 60.0 #Minutes it took on alphabet
                        n_alphabet_words = float(len(selected_word)) / self.word_length
                        alphabet_time =n_alphabet_words / alphabet_time #In wpm
                        user_results[overlap]['alphabet_time'] += alphabet_time
                        user_results[overlap]['clicks_per_char'] += (float(len(clicks)) /float(len(grnd_truth_word))) 
                        if is_error:
                            word_length = float(len(grnd_truth_word))
                            user_results[overlap]['min_edit_dist'] += word_length
                    else:
                        user_results[overlap]['n_error_select']  += 1
                        is_error = True
                        user_results[overlap]['min_edit_dist'] += self.dist_measure.compute(grnd_truth_word, selected_word)
                    user_results[overlap]['n_total']  += 1.0
                    if is_error:
                        top_ten_words =  click_stats['top_ten_words'][0]
                        top_five_words = np.array(top_ten_words[0:4]) 
                        word = selected_word
                        if word == ".":
                            word = "._"
                        idx = np.nonzero( top_five_words == word)[0]
                        user_results[overlap]['n_error_top_three']  += len(idx)
                        #print "User %.2d, nchannels=%d, error_type=%d, speed=%s, grnd_truth_word= [%s] , selection = [%s]"  % (i_user,  i_channel,  int(click_stats['word_error']), self.speeds_ids [overlap], grnd_truth_word, selected_word)
        if i_display:
            self.dispUserChannelStats(i_user, i_channel, user_results)
        return user_results
    
    def userChannelOverlapStats(self, user_results, overlap):
        speed_id =  self.speeds_ids[overlap]
        speed_bound = self.wpm[overlap]
        alphabet_time = user_results[overlap]['alphabet_time'] / float(user_results[overlap]['n_correct']) 
        n_correct = float(user_results[overlap]['n_correct'])
        n_total =   float(user_results[overlap]['n_total'])
        perc_correct = 100.0*n_correct / n_total
        n_correct_top_three = float( n_correct + user_results[overlap]['n_error_top_three'])
        perc_correct_top_three = 100.0 * n_correct_top_three / n_total
        n_errors = float(user_results[overlap]['n_error_timeout']  + user_results[overlap]['n_error_select'])
        perc_errors = 100.0*n_errors / n_total
        n_error_select =  float(user_results[overlap]['n_error_select'])
        n_error_top_three = float(user_results[overlap]['n_error_top_three'])
        n_error_time_out = float(user_results[overlap]['n_error_timeout'])
        perc_error_select= 100.0*n_error_select
        perc_error_top_three = 100.0*n_error_top_three
        perc_time_out =  100.0*n_error_time_out
        if n_errors > 0:
            perc_error_select /= n_errors
            perc_error_top_three /= n_errors
            perc_time_out  /= n_errors
        min_edit_dist =  float( user_results[overlap]['min_edit_dist'] )
        nchars = user_results[overlap]['nchars']
        perc_min_edit_dist  = 100.0  - 100.0* min_edit_dist/ nchars 
          
        return (speed_id, speed_bound, alphabet_time, n_correct, n_total, perc_correct, n_correct_top_three, 
                perc_correct_top_three, n_errors, perc_errors, n_error_select, n_error_top_three,
                n_error_time_out, perc_error_select, perc_error_top_three, perc_time_out, perc_min_edit_dist )
    
    ######################################## Display
     
    def dispAvgUserStats(self, i_display=False):
        results = self.avgUserStats(i_display=i_display)
        n_users = float(len(self.users))
        for channel in self.channels:
                for overlap in self.speed_order:
                    results[channel][overlap]['speed_bound'] /= n_users
                    results[channel][overlap]['wpm' ] /= n_users
                    results[channel][overlap]['n_total'] /= n_users
                    results[channel][overlap]['n_correct'] = 100.0*( results[channel][overlap]['n_correct']/n_users) / results[channel][overlap]['n_total'] 
                    results[channel][overlap]['n_top_three'] = 100.0*( results[channel][overlap]['n_top_three']/n_users) / results[channel][overlap]['n_total'] 
                    results[channel][overlap]['n_select_errors'] = 100.0*( results[channel][overlap]['n_select_errors']/n_users) / results[channel][overlap]['n_total'] 
                    results[channel][overlap]['n_time_out_errors'] = 100.0*( results[channel][overlap]['n_time_out_errors']/n_users) / results[channel][overlap]['n_total'] 
                    results[channel][overlap]['min_edit_dist'] /= n_users
                    results[channel][overlap]['clicks_per_char']  /= n_users
                    print "%d & %s & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f \\\\ \hline" %(channel,  
                    self.speeds_ids[overlap], results[channel][overlap]['speed_bound'],
                    results[channel][overlap]['wpm' ], results[channel][overlap]['n_total'], 
                    results[channel][overlap]['n_correct'],results[channel][overlap]['n_top_three'],
                    results[channel][overlap]['n_select_errors'], results[channel][overlap]['n_time_out_errors'],
                    results[channel][overlap]['min_edit_dist'], results[channel][overlap]['clicks_per_char'] ) 
    
    def dispAvgUserChannelStats(self, i_user, i_channel, user_results):
        print "****************************************************************************************************"
        print "Participant %d, channels %d:" % (i_user, i_channel)
        print "****************************************************************************************************"
        print "===================================================================================================="
        for overlap in self.speed_order:
            (speed_id, speed_bound, alphabet_time, n_correct, n_total, perc_correct, n_correct_top_three, 
            perc_correct_top_three, n_errors, perc_errors, n_error_select, n_error_top_three,
            n_error_time_out, perc_error_select, perc_error_top_three, perc_time_out, 
            min_edit_dist) = self.userChannelOverlapStats(user_results, overlap)
            print "Speed: %s" % speed_id 
            print "(Theoretical approximation): %0.2f wpm" % speed_bound
            print "(Average time spent on alphabet (including missed clicks), correct words): %.2f wpm" %alphabet_time
            print "===================================================================================================="
            print "Average number of correct selections: %d out of %d = %.2f" % (n_correct, n_total, perc_correct ) + "%"
            print "Average number of correct in top three words after any selection/timeout: %d out of %d = %.2f" % (n_correct_top_three, n_total,perc_correct_top_three) + "%"
            print "----------------------------------------------------------------------------------------------------"
            print "Error Analysis:"
            print "----------------------------------------------------------------------------------------------------"
            print "Average number of word errors (total): %d out of %d = %.2f" % (n_errors , n_total, perc_errors ) + "%"
            print "Average number of erroneous word selections: %d out of %d = %.2f" % (n_error_select, n_errors, perc_error_select ) + "% of all errors"
            print "Average number of time out word errors: %d out of %d = %.2f" % (n_error_time_out , n_errors , perc_time_out  ) + "% of all errors"
            print "Average number of errors in top three words: %d out of %d = %.2f" % (n_error_top_three , n_errors, perc_error_top_three ) + "% of all errors."
            print "Average min edit distance for eronous word selections = %0.2f" % min_edit_dist
            print "===================================================================================================="
    
    def dispHeading(self):
        h  = ( "{0:{1}}".format( "user", 4 ) + "|")
        h += ( "{0:{1}}".format( "grnd_truth", 15 ) + "|")
        h += ( "{0:{1}}".format( "select", 15 ) + "|")
        h += ( "{0:{1}}".format( "err", 3 ) + "|")
        h += ( "{0:{1}}".format( "iter", 4 ) + "|")
        h += ( "{0:{1}}".format( "citer", 5 ) + "|")
        h += ( "{0:{1}}".format( "speed", 5 ) + "|") 
        h += ( "{0:{1}}".format( "char_time", 10 ) + "|")
        h += ( "{0:{1}}".format( "clock_time", 10 ) + "|")
        h += ( "{0:{1}}".format( "wpm", 4 ) + "|")
        h += ( "{0:{1}}".format( "err dist", 8 ) + "|")
        h += ( "{0:{1}}".format( "cpc", 4 ) + "|")
        print h
               
    def dispVals(self, i_vals):
        (user, grnd_truth, selection, is_err, iter, c_iter, speed, alphabet_time, clock_time, wpm, err_dist,cpc)  = i_vals
        g  =  ( "{0:{1}}".format( "%d" % user, 4 ) + "|")
        g +=  ( "{0:{1}}".format( "%s" % grnd_truth, 15 ) + "|")
        g +=  ( "{0:{1}}".format( "%s" % selection, 15 ) + "|")
        g +=  ( "{0:{1}}".format( "%d" % is_err, 3 ) + "|")
        g +=  ( "{0:{1}}".format( "%.2d" % iter, 4 ) + "|")
        g +=  self.getDispVal(c_iter, "%.2d", 5)
        g +=  ( "{0:{1}}".format( "%1.2f" % speed, 5 ) + "|")
        g +=  ( "{0:{1}}".format( "%2.2f" % alphabet_time, 10 ) + "|")
        g +=  self.getDispVal(clock_time, "%2.2f", 10)
        g +=  self.getDispVal(wpm, "%1.2f", 4)
        g +=  ( "{0:{1}}".format( "%d" % err_dist, 8 ) + "|")
        g +=  self.getDispVal(cpc, "%2.2f", 4)
        print g
        
    def getDispVal(self, i_val, i_format, i_spaces):
        if i_val is None:
            return "{0:{1}}".format( "-",  i_spaces ) + "|"
        return "{0:{1}}".format( i_format % i_val,  i_spaces ) + "|"
            

if __name__ ==  "__main__":
    results = TickerChannelResults(i_is_students=False) #False)
    #results.dispAvgUserStats(i_display=False) 
    results.userStats(i_display=True, i_display_min_iter=False)
    
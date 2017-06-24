
import sys, copy, os
sys.path.append("../../")
sys.path.append("../")
from utils import Utils, PhraseUtils
from min_edit_distance import MinEditDistance
from click_distr import ClickDistribution 
from channel_config import ChannelConfig
 
class TickerAudioResults(object):
    
    def __init__(self):
        self.root_dir = "../../../user_trials/audio_experiment/ticker/"
        self.file_out = "./results/graphs/results_ticker.cPickle"
        self.phrase_file = "phrases.txt"
        self.users = [3] #[3]
        self.sessions = [2,3,4,5]
        self.sub_sessions = [1,2,3,4,5,6]
        self.sub_session_ids = {1:[1,2,3,4,5,6],2:[2,3,4],3:[2,3,4],4:[2,3,4],5:[1,2]}
        self.dist_measure = MinEditDistance()
        self.word_length = 5.0
        #NB change this variable according to experiment settings (was not saved)
        #Extra waiting time at the end to give the user a chance to click, this value was set to 300ms during all 
        #"no-noise" experiments.
        self.extra_wait_time = 0.0 #0.3
        self.utils = Utils()
        #Diagnostic: count the total number of clicks
        self.nclicks_total = 0
        self.phrase_utils = PhraseUtils()
    
    def initDataStructures(self, i_display):
        (users, wpm_min,wpm_theory,wpm,cpc,char_err,speeds,phrases)=({},{},{},{},{},{},{},[])
        for user in self.users:
            user_dir= "%suser_%.2d/" % (self.root_dir, user)
            if not os.path.exists(user_dir): 
                continue
            for session in self.sessions:
                s = "%d" % session
                session_dir = "%ssession_%.2d/" % (user_dir, session)
                if not os.path.exists(session_dir): 
                    continue
                for sub_session in self.sub_session_ids[session]:
                    ss = "%d" % sub_session
                    sub_session_dir = "%ssub_session_%.2d/" % (session_dir,sub_session)
                    if not os.path.exists(sub_session_dir):
                        continue
                    if not users.has_key(ss):
                        (users[ss], wpm_min[ss],wpm_theory[ss],wpm[ss],cpc[ss],char_err[ss],
                            speeds[ss])=({},{},{},{},{},{},{})
                    (users[ss][s], wpm_min[ss][s],wpm_theory[ss][s],wpm[ss][s],cpc[ss][s],char_err[ss][s],
                        speeds[ss][s])=([],[],[],[],[],[],[])
        if i_display:
            self.dispHeading()
        return (users,wpm_min,wpm_theory,wpm,cpc,char_err,speeds)
       
    ####################################### Load Functions

    def compute(self, i_display):
        (users,wpm_min,wpm_theory,wpm,cpc,char_err,speeds) = self.initDataStructures(i_display)
        for user in self.users:
            user_dir= "%suser_%.2d/" % (self.root_dir, user)
            if not os.path.exists(user_dir): 
                continue
            for sub_session in self.sub_sessions:
                ss = "%d" % sub_session
                if not users.has_key(ss):
                    continue
                for session in self.sessions:
                    s = "%d" % session
                    if not users[ss].has_key(s):
                        continue
                    sub_session_dir = "%ssession_%.2d/sub_session_%.2d/" % (user_dir,int(s),int(ss))
                    if not os.path.exists(sub_session_dir):
                        continue       
                    phrases = self.utils.loadText(sub_session_dir + self.phrase_file).split("\n")[0:-1]
                    for phrase_cnt in range(0,len(phrases)):
                        words = phrases[phrase_cnt].split('_')
                        for word_cnt in range(0, len(words)): 
                            filename =  "%sclick_stats_%.2d_%.2d.cPickle" % (sub_session_dir, phrase_cnt, word_cnt) 
                            if not os.path.exists(filename):
                                continue
                            #print "file_name = ", filename
                            click_stats = self.utils.loadPickle(filename)
                            if not click_stats['is_calibrated']:
                                continue 
                            results = self.getResults(user, s, ss, click_stats, words[word_cnt], i_display)   
                            saved_results = (users,wpm_min,wpm_theory,wpm,cpc,char_err,speeds)
                            saved_results = self.updateResults(results, saved_results, s, ss)
        #Save the results
        r = {}
        (r['users'], r['wpm_min'],r['wpm_theory'],r['wpm'],r['cpc'],r['char_err'],r['speeds']) = saved_results 
        print "Saving to file ", self.file_out
        self.utils.savePickle(r, self.file_out)
        if i_display:
            print "Total clicks received = ", self.nclicks_total
    
    def updateResults(self, i_results, i_saved_results, i_session, i_sub_session):
        #The final output results
        (users,wpm_min,wpm_theory,wpm,cpc,char_err,speeds) = i_saved_results 
        #All the results to display
        (user, grnd_truth, selection, iterations, n_click_iter, n_undo, is_word_err,
            char_read_time, char_read_theory,end_delay,iter_time, cur_wpm_min,cur_wpm_theory,cur_wpm, 
            click_distr, n_clicks, cur_cpc, min_edit_dist, overlap, speed) = i_results 
        #Update the final results to save
        users[i_sub_session][i_session].append(user)
        wpm_min[i_sub_session][i_session].append(cur_wpm_min)
        wpm_theory[i_sub_session][i_session].append(cur_wpm_theory)
        wpm[i_sub_session][i_session].append(cur_wpm)
        cpc[i_sub_session][i_session].append(cur_cpc)
        char_err[i_sub_session][i_session].append(min_edit_dist)
        speeds[i_sub_session][i_session].append(speed)
        return (users,wpm_min,wpm_theory,wpm,cpc,char_err,speeds)
                            
    ################################################# Get 
    
    def getResults(self, i_user, i_session, i_subsession, i_click_stats, i_cur_word, i_display):
        c = dict(i_click_stats)
        grnd_truth = c['grnd_truth']
        cur_word = self.phrase_utils.getWord(i_cur_word)
        if not ( grnd_truth == cur_word):
            print "grnd_truth = ", grnd_truth, " should be " , cur_word
            raise ValueError("Grnd truth incorrect")
        selection = self.phrase_utils.getWord(c['selected_word'])
        #The number of times the user used the undo function
        n_undo = c['nundo']
        #Total number of alphabet sequence iteration
        iterations = c['niterations'] 
        #Total number of iterations where clicks were received
        n_click_iter = c['nclick_iterations']
        #Is there a timeout or word-selection error? 
        is_word_err = c['word_error'] or (not (grnd_truth == selection))
        #Normalise the time it took to read only the alphbaet sequence correctlt
        #The delay at the end of the sequence (after reading the character)
        #Initially this was not recorded
        if c['settings'].has_key('end_delay'):
            end_delay = c['settings']['end_delay']
        else:
            end_delay = self.extra_wait_time
        #The measured time taken to read a character
        char_read_time = c['alphabet_read_time']*float(c['nclick_iterations']) / float(iterations - n_undo)
        #The theoretical time it should take to read a character
        (file_length, nchannels, overlap) = (c['settings']['file_length'],5,c['settings']['overlap'])
        root_dir = "../../"
        channel_config = ChannelConfig(nchannels,overlap ,file_length, root_dir)
        char_read_theory = channel_config.getSoundTimes()[-1,-1]
        #Compute all the wpm from the char reading times and the number of iterations
        #The min possible wpm for the number of iterations
        speed =  60.0/(self.word_length*char_read_theory)
        wpm_min = 60.0/(char_read_theory * iterations)* (float(len(selection)) / self.word_length)
        #The theoretical wpm 
        wpm_theory = 60.0/((char_read_theory+end_delay)*iterations)* (float(len(selection))/ self.word_length)
        #The measured wpm
        #Iter time = total time used to compute the measured wpm
        iter_time = (char_read_time+end_delay) * iterations
        wpm = 60.0/iter_time* (float(len(selection))/ self.word_length)
        #The number of clicks used to write the word
        n_clicks = c['nclicks'] 
        self.nclicks_total += n_clicks
        #The speed (how much the sounds overlapped)
        overlap = c['settings']['overlap'] 
        #The click distribution after the word was selected
        click_distr = copy.deepcopy(c['click_distr_after'])
        #The clicks per character
        cpc = float(n_clicks) / float(len(grnd_truth))
        #The error rate
        min_edit_dist = 0.0
        if is_word_err: 
            if not (selection == ""):
                min_edit_dist = self.dist_measure.compute(grnd_truth,selection)
                min_edit_dist = (100.0*float(min_edit_dist) / len(grnd_truth))
                cpc = float(n_clicks) / float(len(selection))
            else:
                min_edit_dist = 100.0
                (wpm_theory, wpm) = (0.0, 0.0)
        r = (i_user, grnd_truth, selection, iterations, n_click_iter, n_undo, is_word_err,
            char_read_time, char_read_theory,end_delay,iter_time, wpm_min,wpm_theory,wpm, 
            click_distr, n_clicks, cpc, min_edit_dist, overlap, speed )
        if i_display:
            self.dispUserResults(r, i_session, i_subsession )
        #Some diagnostic tests
        if (not i_user == 3) or ((i_user==3)  and (int(i_session)==5)):
            dist = abs(char_read_theory-char_read_time)
            if dist > 1E-1:
                raise ValueError("Character reading times should be equal")
        return r
                            
    ################################################### Display
    
    def dispUserResults(self, i_results, i_session, i_subsession):
            
        (user, grnd_truth, selection, iterations, n_click_iter, n_undo, is_word_err,
            char_read_time, char_read_theory,end_delay,iter_time, wpm_min,wpm_theory,wpm, 
            click_distr, n_clicks, cpc, min_edit_dist, overlap, speed) = i_results 
        g  =  ( "{0:{1}}".format( "%d" % user, 4 ) + "|")
        g +=  ( "{0:{1}}".format( "%s" % i_session, 8 ) + "|")
        g +=  ( "{0:{1}}".format( "%s" % i_subsession, 8 ) + "|")
        g +=  ( "{0:{1}}".format( "%s" % grnd_truth, 12 ) + "|")
        g +=  ( "{0:{1}}".format( "%s" % selection, 12 ) + "|")
        g +=  ( "{0:{1}}".format( "%d" % is_word_err, 3 ) + "|")
        g +=  ( "{0:{1}}".format( "%.2d" % iterations, 4 ) + "|")
        g +=  ( "{0:{1}}".format( "%.2d" % n_click_iter, 5 ) + "|")
        g +=  ( "{0:{1}}".format( "%.2d" % n_undo, 5 ) + "|")
        g +=  ( "{0:{1}}".format( "%2.2f" % char_read_theory, 7 ) + "|")
        g +=  ( "{0:{1}}".format( "%2.2f" % char_read_time, 8 ) + "|")
        g +=  ( "{0:{1}}".format( "%2.2f" % end_delay, 5 ) + "|")
        g +=  ( "{0:{1}}".format( "%1.2f" % wpm_min, 7 ) + "|")
        g += self.phrase_utils.getDispVal(wpm_theory, "%1.2f", 7)
        g += self.phrase_utils.getDispVal(wpm, "%1.2f", 4)
        g +=  ( "{0:{1}}".format( "%3.2f" % iter_time, 8 ) + "|")
        g += self.phrase_utils.getDispVal(cpc, "%1.2f", 4)
        g +=  ( "{0:{1}}".format( "%3.2f" % min_edit_dist, 6 ) + "|")
        g +=  ( "{0:{1}}".format( "%1.2f" % overlap, 4 ) + "|")
        g +=  ( "{0:{1}}".format( "%1.2f" % speed, 5 ) + "|")
        g +=  ( "{0:{1}}".format( "%.2d" % n_clicks, 6 ) + "|")
        print g
        
    def dispHeading(self):
        self.dispHeadingDescriptions()
        h  = ( "{0:{1}}".format( "user", 4 ) + "|")
        h += ( "{0:{1}}".format( "session", 8 ) + "|")
        h += ( "{0:{1}}".format( "subsess", 8 ) + "|")
        h += ( "{0:{1}}".format( "grnd_truth", 12 ) + "|")
        h += ( "{0:{1}}".format( "select", 12 ) + "|")
        h += ( "{0:{1}}".format( "err", 3 ) + "|")
        h += ( "{0:{1}}".format( "iter", 4 ) + "|")
        h += ( "{0:{1}}".format( "citer", 5 ) + "|")
        h += ( "{0:{1}}".format( "nundo", 5 ) + "|")
        h += ( "{0:{1}}".format( "chr_thr", 7 ) + "|")
        h += ( "{0:{1}}".format( "chr_time", 8 ) + "|")
        h += ( "{0:{1}}".format( "delay", 5 ) + "|")
        h += ( "{0:{1}}".format( "wpm_min", 7 ) + "|")
        h += ( "{0:{1}}".format( "wpm_thr", 7 ) + "|")
        h += ( "{0:{1}}".format( "wpm", 4 ) + "|")
        h += ( "{0:{1}}".format( "tot_time", 8 ) + "|")
        h += ( "{0:{1}}".format( "cpc", 4 ) + "|")
        h += ( "{0:{1}}".format( "%error", 6 ) + "|") 
        h += ( "{0:{1}}".format( "over", 4 ) + "|")
        h += ( "{0:{1}}".format( "speed", 5 ) + "|") 
        h += ( "{0:{1}}".format( "nclicks", 6 ) + "|")
        print h
                       
                    
    def dispHeadingDescriptions(self):
        print "grnd_truth: The word the user is supposed to write"
        print "select: The word selected by the user"
        print "err (1/0): 1 if an error occured (time-out error or if select is not equal to grnd_truth"
        print "iter: The total number of alphabet-sequence repetitions (scans) to select the word"
        print "citer: The number of alphabet-sequence repetitions (scans) where the user clicked"
        print "nundo: The number of times the user clicked 4 or more times to restart a letter selection"
        print "chr_thr (seconds): The time it should take to read one alphabet sequence (scan)"
        print "chr_time (seconds): The measured time it took to read one alphabet sequence (scan)"
        print "delay (seconds): The delay at the end of an alphabet sequence to wait for last clicks"
        print "wpm_min (words/min): The min time it could to write the word take with an efficient implementation"
        print "wpm_theory(words/min): The min time it could to write the word take with an efficient implementation, including the end delay"
        print "wpm (words/min): The measured time it took write the word"
        print "tot_time (seconds): The total time it took to write the word (corresponding to wpm)"
        print "cpc: The average number of clicks per character used to  write the word, word lenghth=5"
        print "%error: %of word length that was wrong after the word selection"
        print "over (a number between 0 and 1): Sound overlap between adjacent letters (speed setting)"
        print "speed (words/min): The sound overlap (speed setting) expressed as wpm" 
        print "nclicks: The total number of clicks to select the word." 
    
    
if __name__ ==  "__main__":
    app = TickerAudioResults() 
    app.compute(i_display=True)

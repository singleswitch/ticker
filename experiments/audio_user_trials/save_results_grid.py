
import sys, copy, os
sys.path.append("../../")
sys.path.append("../")
from utils import Utils, PhraseUtils
from min_edit_distance import MinEditDistance
import numpy as np
 
class GridAudioResults(object):
    
    def __init__(self):
        self.root_dir = "../../../user_trials/audio_experiment/grid/"
        self.file_out = "./results/graphs/results_grid.cPickle"
        self.phrase_file = "phrases.txt"
        self.users = [3]
        self.sessions = [1,2,3,4]
        self.sub_sessions = [1,2,3,4,5,6]
        self.sub_session_ids = {1:[2,3,4], 2:[2,3,4], 3:[1,2,3],4:[1,2]}
        self.dist_measure = MinEditDistance()
        self.word_length = 5.0
        self.phrase_utils = PhraseUtils()
        self.utils = Utils()
        #Diagnostic: count the total number of clicks
        self.nclicks_total = 0
    
    def initDataStructures(self, i_display):
        (users, wpm,cpc,char_err,speeds,phrases)=({},{},{},{},{},[])
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
                        (users[ss], wpm[ss],cpc[ss],char_err[ss],speeds[ss])=({},{},{},{},{})
                    (users[ss][s], wpm[ss][s],cpc[ss][s],char_err[ss][s],speeds[ss][s])=([],[],[],[],[])
        if i_display:
            self.dispHeading()
        return (users,wpm,cpc,char_err,speeds)
       
    ####################################### Load Functions

    def compute(self, i_display):
        (users, wpm,cpc,char_err,speeds) = self.initDataStructures(i_display)
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
                            results = self.getResults(user, s, ss, click_stats, words[word_cnt], i_display)   
                            saved_results = (users, wpm,cpc,char_err,speeds)
                            saved_results = self.updateResults(results, saved_results, s, ss)
        #Save the results
        r = {}
        (r['users'], r['wpm'],r['cpc'],r['char_err'],r['speeds']) = saved_results 
        print "Saving to file ", self.file_out
        self.utils.savePickle(r, self.file_out)
        if i_display:
            print "Total clicks received = ", self.nclicks_total
    
    def updateResults(self, i_results, i_saved_results, i_session, i_sub_session):
        (users, wpm,cpc,char_err,speeds) = i_saved_results
        (user, grnd_truth, selection, n_scans, n_undo, n_delete, is_word_err, scan_delay, total_time, 
            n_clicks, cur_wpm, cur_cpc, min_edit_dist) = i_results
        users[i_sub_session][i_session].append(user)
        wpm[i_sub_session][i_session].append(cur_wpm)
        cpc[i_sub_session][i_session].append(cur_cpc)
        char_err[i_sub_session][i_session].append(min_edit_dist)
        speeds[i_sub_session][i_session].append(scan_delay)
        return (users,wpm,cpc,char_err,speeds)
                            
    ################################################# Get 

    def getResults(self, i_user, i_session, i_subsession, i_click_stats, i_cur_word, i_display):
        c = dict(i_click_stats)
        grnd_truth = c['grnd_truth']
        cur_word = self.phrase_utils.getWord(i_cur_word)
        if not ( grnd_truth == cur_word):
            print "grnd_truth = ", grnd_truth, " should be " , cur_word
            raise ValueError("Grnd truth incorrect")
        selection = str(c['selected_word'])
        #Total number of scans
        n_scans = np.sum(c['nscans'])
        n_undo = c['undo_last_action_cnt']
        n_delete = c['delete_cnt']
        is_word_err = c['word_error'] or (not (grnd_truth == selection))
        #Normalise the time it took to read only the alphbaet sequence correctlt
        scan_delay = c['settings']['scan_delay']
        total_time = scan_delay*n_scans
        n_clicks = c['nclicks']
        self.nclicks_total += n_clicks
        wpm = 60.0/(self.word_length*total_time / float(len(selection)))
        cpc = float(n_clicks) / float(len(grnd_truth))
        min_edit_dist = 0.0
        if is_word_err: 
            if not (selection == ""):
                min_edit_dist = self.dist_measure.compute(grnd_truth,selection)
                min_edit_dist = (100.0*float(min_edit_dist) / len(grnd_truth))
                cpc = float(n_clicks) / float(len(selection))
            else:
                min_edit_dist = 100.0
                (wpm_theory, wpm) = (0.0, 0.0)
        else:
            cpc = float(n_clicks) / float(len(selection))
        r = (i_user, grnd_truth, selection, n_scans, n_undo, n_delete, is_word_err, scan_delay, total_time, 
            n_clicks, wpm, cpc, min_edit_dist)
        if i_display:
            self.dispUserResults(r, i_session, i_subsession )
        return r
                            
    ################################################### Display
    
    def dispUserResults(self, i_results, i_session, i_subsession):
        (user, grnd_truth, selection, n_scans, n_undo, n_delete, is_word_err, scan_delay, total_time, 
            n_clicks, wpm, cpc, min_edit_dist) = i_results
        g  =  ( "{0:{1}}".format( "%d" % user, 4 ) + "|")
        g +=  ( "{0:{1}}".format( "%s" % i_session, 8 ) + "|")
        g +=  ( "{0:{1}}".format( "%s" % i_subsession, 8 ) + "|")
        g +=  ( "{0:{1}}".format( "%s" % grnd_truth, 14 ) + "|")
        g +=  ( "{0:{1}}".format( "%s" % selection, 14 ) + "|")
        g +=  ( "{0:{1}}".format( "%d" % is_word_err, 6 ) + "|")
        g +=  ( "{0:{1}}".format( "%.2d" % n_scans, 6) + "|")
        g +=  ( "{0:{1}}".format( "%.2d" % n_undo, 5 ) + "|")
        g +=  ( "{0:{1}}".format( "%.2d" % n_delete, 8) + "|")
        g +=  ( "{0:{1}}".format( "%1.2f" % n_clicks, 8 ) + "|")
        g +=  ( "{0:{1}}".format( "%2.2f" % total_time, 11 ) + "|")
        g += self.phrase_utils.getDispVal(wpm, "%1.2f", 6)
        g += self.phrase_utils.getDispVal(cpc, "%1.2f", 6)
        g +=  ( "{0:{1}}".format( "%1.2f" % min_edit_dist, 6 ) + "|")
        g +=  ( "{0:{1}}".format( "%2.2f" % scan_delay, 10 ) + "|") 
        print g
        
    def dispHeading(self):
        h  = ( "{0:{1}}".format( "user", 4 ) + "|")
        h += ( "{0:{1}}".format( "session", 8 ) + "|")
        h += ( "{0:{1}}".format( "subsess", 8 ) + "|")
        h += ( "{0:{1}}".format( "grnd_truth", 14 ) + "|")
        h += ( "{0:{1}}".format( "select", 14 ) + "|")
        h += ( "{0:{1}}".format( "is_err", 6 ) + "|")
        h += ( "{0:{1}}".format( "nscans", 6 ) + "|")
        h += ( "{0:{1}}".format( "nundo", 5 ) + "|")
        h += ( "{0:{1}}".format( "n_delete", 8 ) + "|")
        h += ( "{0:{1}}".format( "n_clicks", 8 ) + "|") 
        h += ( "{0:{1}}".format( "total_time", 11 ) + "|")   
        h += ( "{0:{1}}".format( "wpm", 6 ) + "|")
        h += ( "{0:{1}}".format( "cpc", 6 ) + "|")
        h += ( "{0:{1}}".format( "%error", 6 ) + "|")
        h += ( "{0:{1}}".format( "scan_delay", 10 ) + "|")
        print h
     
if __name__ ==  "__main__":
    app = GridAudioResults() 
    app.compute(i_display=True)

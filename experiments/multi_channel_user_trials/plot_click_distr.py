
import os, cPickle, sys
sys.path.append("../../")
sys.path.append("../")
from utils import Utils, DispSettings
import numpy as np
import pylab as p
from kernel_density import NonParametricPdf
from model_comparison import TickerChannelResults

class TickerChannelResultsClickDistrDisplay(object):
    
    def __init__(self):
        #Params
        self.disp = False
        self.save = True 
        #Settings are different for the case stud
        self.results = TickerChannelResults()
        self.results.load_click_distr = False
        self.utils = Utils()
        self.plots = DispSettings()
        self.plots.setClickDataDisp()
        self.R = 2.0
        self.fp_thresh = 0.6 #False positive threshold
        self.dir_out = "../../../user_trials/user_trial_pictures/"
       
    ####################################### Main
    
    def saveHistograms(self, i_users):
        if not self.save: 
            return
        (data_tp,  fr , fp) = ([],[],[])
        for user in i_users:
            file_name = "%shist_%.2d" % (self.dir_out, user)
            (tmp_data_tp, tmp_fr, tmp_fp) = self.displayUserClickData(user, i_display_avg=False)
            self.plots.saveFig(file_name, i_save_eps=False)
            p.close()
            data_tp.extend(list(tmp_data_tp))
            fr.append(tmp_fr)
            fp.append(tmp_fp)
        (data_tp, fr, fp) = (np.array(data_tp), np.array(fr), np.array(fp))
        self.newFigure()
        self.drawPdf(data_tp, np.mean(fr), np.mean(fp), i_title="")
        self.plots.saveFig(self.dir_out + "all_user_data",i_save_eps=False) 
    
    def newFigure(self):
        p.figure(facecolor='w',figsize=(15,14))
        self.plots.newSubFigure()

    def displayUserClickData(self, i_user, i_display_avg=True):
        channels = [3,4,5]
        overlaps = list(self.results.settings.speed_order)
        self.newFigure()
        n_fig = 1
        (data_tp,  fr , fp) = ([],[],[])
        for (n_ch, channel) in enumerate(channels):
            for (n_over, overlap_str) in enumerate(overlaps):
                overlap = float(overlap_str)
                p.subplot(len(channels), len(overlaps), n_fig);
                (tmp_data_tp, tmp_fr, tmp_fp) = self.displayUserClickDataSetting(i_user, channel, overlap)
                n_fig += 1
                data_tp.extend(list(tmp_data_tp))
                fr.append(tmp_fr)
                fp.append(tmp_fp)
        (data_tp, fr, fp) = (np.array(data_tp), np.array(fr), np.array(fp))
        if i_display_avg:
            self.newFigure()
            disp_str = "user=%d " % i_user
            self.drawPdf(data_tp, np.mean(fr), np.mean(fp), i_title=disp_str)
        return (data_tp, fr, fp)
            
    def displayUserClickDataSetting(self, i_user, i_channel, i_overlap):
        #Get the click data for a specific user and settings
        (click_times, selected_letters) = self.getClickData(i_user, i_channel, i_overlap)
        #Display all the clicks
        (deltas, false_reject) = self.normaliseClickData(click_times, selected_letters)
        #Identify the false positive
        idx_fp = np.nonzero( np.abs(deltas) > self.fp_thresh )[0]
        data_fp = deltas[idx_fp]
        idx_tp = np.nonzero( np.abs(deltas) <= self.fp_thresh )[0]
        data_tp = deltas[idx_tp]
        fp_rate = float(len(data_fp)) / (len(selected_letters)*self.results.settings.click_distr.T)
        speed_str = self.results.settings.speeds_ids["%.2f" % i_overlap]
        title ="u=%d, ch=%d, sp=%s, "% (i_user, i_channel, speed_str)
        (data_mean, data_std) = self.drawPdf(data_tp, false_reject, fp_rate, title)
        p.plot( data_tp, np.zeros(len(data_tp)), 'rx', linewidth=2)
        p.plot( data_fp, np.zeros(len(data_fp)), 'gx', linewidth=5)
        #self.displayClickDistrOut(click_times, deltas, selected_letters)
        #Save the histogram
        if self.save:
            file_name = "%shist_%.2d_%d_%.2f.cPickle" % (self.dir_out, i_user, i_channel, i_overlap) 
            pdf = self.results.settings.click_distr.histogram
            params = (pdf, data_mean, data_std, fp_rate, false_reject)
            self.utils.savePickle( params, file_name)
        return (data_tp, false_reject, fp_rate)
        
        
    def drawPdf(self, i_data, i_fr, i_fp_rate, i_title=""):
        #Setup the pdf
        self.results.settings.click_distr.histogram = NonParametricPdf()
        self.results.settings.click_distr.histogram.setStdFromData(i_data)
        self.results.settings.click_distr.histogram.saveDataPoints(i_data)
        #pdf.setKernelBandWidth(0.05)
        #Data stats
        kernel_std = self.results.settings.click_distr.histogram.kernel_std
        (data_mean, data_std) = (np.mean(i_data), np.std(i_data))
        #Make drawing
        disp_str =  "%sk_std=%.2f, "% (i_title, kernel_std)
        disp_str2 = "u=%.2f, std=%.2f, fr=%.2f, fp=%.4f" % (data_mean, data_std, i_fr, i_fp_rate)
        print disp_str + disp_str2
        self.results.settings.click_distr.histogram.draw(i_color="k", i_histogram=True)
        p.title(disp_str+"\n"+disp_str2)
        return (data_mean, data_std)
    
    
    ######################################################## Click data
    
    def normaliseClickData(self, i_click_times, i_selected_letters):
        if self.disp:
            print "*************************************************************"
        o_data = []
        o_fr = 0.0
        for n in range(0, len(i_click_times)):
            letter = i_selected_letters[n] 
            letter_idx = np.nonzero(np.array(self.results.settings.click_distr.alphabet) == letter )[0][0]
            letter_times = self.results.settings.click_distr.loc[letter_idx,:]
            deltas = np.atleast_2d(i_click_times[n]).transpose() - letter_times
            dist = np.abs(deltas)
            min_vals_idx = np.argmin(abs(dist),axis=1)
            min_vals = deltas[range(0,dist.shape[0]), min_vals_idx].flatten()
            o_data.extend(list(min_vals))
            #Also store the percentage of clicks missed
            o_fr += len(i_click_times[n])
            if self.disp:
                letter_times_str = self.utils.stringVector(letter_times, i_type="%.3f")
                click_times_str = self.utils.stringVector(i_click_times[n], i_type="%.3f")
                delta_str = self.utils.stringVector(min_vals, i_type="%.3f")
                disp_str =  "n=%.2d, letter=%s, letter_idx=%.2d " % (n, letter, letter_idx)
                disp_str += ("click_times=%s, letter_times_str=%s " % (click_times_str,letter_times_str))
                disp_str += ("deltas=%s" % delta_str)  
                print disp_str
        o_fr /= (2.0*len(i_click_times))
        o_fr = 1.0 - o_fr
        o_fr = np.clip(o_fr, 0.0, 1.0)
        return (np.array(o_data), o_fr)
  
    def getClickData(self, i_user, i_channel, i_overlap):
        (overlap, n_channels, delay) = (0.0, 0, -np.inf)  
        phrases = self.results.loadPhrases(i_user, i_channel) 
        selected_letters = []
        click_times = []
        for phrase_cnt in range(0, len(phrases)):
            words = phrases[phrase_cnt].split('_')
            for word_cnt in range(0, len(words)):
                c = self.results.loadUserStats(i_user, i_channel, phrase_cnt, word_cnt )
                if c is None:
                    continue  
                if np.abs( i_overlap-float(c['overlap']))>1E-6:
                    continue
                word = self.results.getWord(words, word_cnt)
                (n_iter, letter_idx) = (0, 0) 
                if len(c['clicks']) < 1:
                    continue
                selected_letters.append(word[letter_idx])
                click_times.append([])
                for n in range(0,len(c['clicks'])):
                    cn = c['clicks'][n]
                    (overlap,n_channels, delay) = self.results.resetIfParamsChanged(i_user, c,n,delay,overlap,n_channels)
                    if not (n_channels == i_channel):
                        raise ValueError("Settings have changed")
                    if not (cn['letter_index'] == letter_idx):
                        click_times.append([])
                        letter_idx += 1
                        selected_letters.append(word[self.results.ticker.warpIndices(letter_idx, len(word))])
                    click_times[-1].append(cn['click_time'])
        return (click_times, selected_letters)
            
    ############################################### Diagnostic
    
    def displayClickDistrOut(self, i_click_times, i_deltas, i_selected_letters):
        delta_index = 0
        for n in range(0,len(i_click_times)):
            letter_scores = self.results.settings.click_distr.logLikelihood(i_click_times[n], i_log=False)
            letter = i_selected_letters[n] 
            letter_idx = np.nonzero(np.array(self.results.settings.click_distr.alphabet) == letter )[0][0]
            y = letter_scores[letter_idx]
            delta_n = i_deltas[range(delta_index, delta_index+len(i_click_times[n]))]
            delta_index += len(i_click_times[n])
            print "n = ", n, " letter = ", letter, " deltas= ", delta_n, " score = ", y
            
if __name__ ==  "__main__":
    results = TickerChannelResultsClickDistrDisplay()
    #results.displayUserClickData(i_user=9)
    results.saveHistograms(range(9,10)) #7,20))
    p.show()

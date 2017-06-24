
import pylab as p
import numpy as np
import sys, copy
sys.path.append('../../')
sys.path.append('../')
from utils import DispSettings
from click_distr_disp import  ClickDistributionDisplay
from channel_config import ChannelConfig
from ticker_sampling import TickerSampling
from ticker_click_noise_numerical import TickerClickNoise2D
from PyQt4 import QtCore, QtGui

class SimulationPlots():

    def __init__(self):
        self.disp = DispSettings()
        #Output directory for final plots
        self.output_dir = "results"
        self.save = True
        #The channel colours
        self.channel_colours = [ QtGui.QColor("red"), 
                                QtGui.QColor("green"),
                                QtGui.QColor("blue"), 
                                QtGui.QColor("Brown"),
                                QtGui.QColor("Purple")]
        for n in range(0, len(self.channel_colours)):
            c = self.channel_colours[n]
            rgb = (c.red()/255.0, c.green()/255.0, c.blue()/255.0)
            self.channel_colours[n] = rgb
        
    def clickDistributionPlots(self):
        #Initialise the channel configuration and all parameters
        channel_config = ChannelConfig(i_nchannels=5, i_sound_overlap=0.65, i_file_length=0.21, i_root_dir="../../")
        distr_disp  = ClickDistributionDisplay()
        distr_disp.disp = self.disp
        distr_disp.disp.setClickDistributionSeqDisplay1D()
        distr_disp.setChannel(channel_config)
        distr_disp.disp.params['text.usetex'] = False
        print "T = ", distr_disp.distr.T
        #Click time model params - can test variety of fp and fr
        fp_rates = [0.0, 0.05, 0.1, 0.15, 0.3, 0.5, 0.75, 1.0]
        fr_rates = [0.0, 0.1, 0.2,  0.3, 0.5, 0.8, 1.0]
        gauss_std = 0.1
        gauss_delay = 0.0
        target_idx = 8  #Which letter is intended
        self.letter = self.getLetter(channel_config, target_idx) #The current letter
        aspect_1D =  0.6 #Aspect ratio for 1D click-time plots
        max_1D = None    #Max for 1D click-time plot
        distr_disp.distr.setGaussParams(channel_config, gauss_delay, gauss_std)
        distr_disp.distr.clearHistogram()
        #The overly simple noise model (GMM)
        max_1D = self.dispSimpleClickDistrPlots(channel_config, target_idx,fp_rates, fr_rates, distr_disp, aspect_1D)
        #The complicated model - see how the figure changes
        self.dispComplicatedClickDistrPlots(channel_config, target_idx, fp_rates, fr_rates, distr_disp, max_1D, aspect_1D)
        
        """The 1D GMM plots (1 click received, click distribution)
        * The old simple noise model (without labelling all clicks), and not fp and fr"""
        """The 1D and 2D letter configuarations. 1D is with some samples"""
        self.dispLetterConfigs( distr_disp, channel_config, self.save)
        
    ################################## Letter configurations
                
    def dispLetterConfigs( self, i_distr_disp, channel_config, i_save ):
        self.disp.setAlphabetConfigDisplay()
        channels = range(1,6)
        for channel in channels:
            print "**************************************************************"
            print "channels = ", channel
            print "**************************************************************"
            distr_disp = copy.deepcopy(i_distr_disp)
            channel_config = ChannelConfig(i_nchannels=channel, i_sound_overlap=0.0, i_file_length=1.0, i_root_dir="../../")
            distr_disp.setChannel(channel_config) 
            self.__dispLetterConfig(distr_disp, channel_config)
            #Get and display the nearest neighbours
            k = 8
            for n in range(0, distr_disp.distr.loc.shape[0]):
                point = distr_disp.distr.loc[n,:]
                dist = np.abs(distr_disp.distr.loc - point) 
                dist = np.sort(np.max(dist,axis=1))[1:k]
                print "n = ", n, " dist = ", dist
            
    def __dispLetterConfig(self, i_distr_disp, i_channel_config):
        self.disp.newFigure()
        (nticks, nchannels) = self.plotAlphabet(i_channel_config, i_distr_disp)
        p.xlabel('$\mu_{1}$')
        p.ylabel('$\mu_{2}$')
        xmin = np.min(i_distr_disp.distr.loc[:,0])
        xmax = np.max(i_distr_disp.distr.loc[:,0])+2.4
        ymin = np.min(i_distr_disp.distr.loc[:,1])-1.0 
        ymax = np.max(i_distr_disp.distr.loc[:,1])+2.0
        p.axis([xmin,xmax,ymin,ymax])
        ax = p.gca()
        aspect_ratio = 1.0
        ax.set_aspect(aspect_ratio)
        file_name =  "%s/letter_config_%.2d" % (self.output_dir, nchannels)
        self.saveFig("Letter configuration ", file_name )
        
    def plotAlphabet(self, i_channel_config, i_distr_disp):
        alphabet = np.array(i_channel_config.getAlphabetLoader().getUniqueAlphabet( i_with_spaces=True, i_group=True))
        nchannels = alphabet.shape[0]
        nticks = i_channel_config.getNumberOfPreTicks()         
        delta_x = 1.7
        p.plot(i_distr_disp.distr.loc[:,0]+1, i_distr_disp.distr.loc[:,1]+1, 'ko', linewidth=1)
        for n in range(0, i_distr_disp.distr.loc.shape[0]):
            letter =  i_distr_disp.distr.alphabet[n]
            if (i_distr_disp.distr.alphabet[n] == '_') and self.disp.params['text.usetex'] :
                letter_str = "\_"
            else:
                letter_str = letter
            (x,y) = (i_distr_disp.distr.loc[n,0]+1, i_distr_disp.distr.loc[n,1]+1)
            font = self.disp.params['text_font']
            a = p.text(x+delta_x, y,  letter_str, fontsize=font,ha='center', va='center' )
            color = self.getColour(alphabet, letter)
            a.set_color(color)
        return  (nticks, nchannels)
    
    def getColour(self, i_alphabet, i_letter):
        nchannels = i_alphabet.shape[0]
        if nchannels == 1:
            return 'k'
        for ch in range(0, nchannels):
            idx  = np.nonzero(i_alphabet[ch,:] == i_letter)[0]
            if len(idx) > 0:
                break
        return self.channel_colours[ch]
         
        
    ################################# Simple Noise Model Plots
    
    def dispSimpleClickDistrPlots(self, i_channel_config, i_target_idx, fp_rates, fr_rates, i_distr_disp, i_aspect):
        start_num = 1
        #One click received
        max_1D = self.dispSimpleSingle1DTargetClickDistrPlots(i_target_idx,fp_rates,fr_rates,i_distr_disp, start_num, i_aspect)
        #Two clicks received
        self.dispSimpleSingle2DTargetClickDistrPlots(i_channel_config, i_target_idx, i_distr_disp)
        return max_1D 

    def dispSimpleSingle1DTargetClickDistrPlots( self, i_target_idx, fp_rates, fr_rates, i_distr_disp,  start_num, i_aspect=None):
        i_distr_disp.disp.setClickDistributionDisplay1D()
        fp_idx = 0
        fr_idx = 4
        noise_simple = True
        is_many = False
        n = fp_idx*len(fr_rates) + fr_idx
        (fp, fr)  = (fp_rates[fp_idx], fr_rates[fr_idx])
        file_name = self.output_dir + "/simple1D_" + str(start_num)  
        print "***************************************************"
        print "Simple noise 1D file_name = ", file_name
        print "***************************************************"
        max_val = self.seqSingleTarget1DClick([fp], [fr], i_distr_disp, i_target_idx, 300, noise_simple, is_many)
        ylabel = "P(t_{1}, M=1 \mid  \\boldsymbol{\\theta}, \ell=\mathrm{b})"
        if i_aspect is not None:
            aspect = i_aspect
        else:
            aspect = None
        self.finalise1DPlot(ylabel, aspect, file_name, self.save)
        return max_val
    
    def dispSimpleSingle2DTargetClickDistrPlots(self, i_channel_config, i_target_idx, i_distr_disp):
        display = True
        is_plot = True #Plot contours
        prob_thresh = 0.9
        priors = None                  # Priors probs for letters in i_alphabet
        score_mask = False             # Compute the region of interest where the posterior is bigger than all the rest
        (gauss_delay, gauss_std, fr, fp_rate)= i_distr_disp.distr.getParams()
        (T, n_std) = (i_distr_disp.distr.T, 0.0)
        #Load the alphabet and click times
        alphabet_loader =  i_channel_config.getAlphabetLoader()
        alphabet = np.array(alphabet_loader.getAlphabet( i_with_spaces=False))
        alphabet_unique = alphabet_loader.getUniqueAlphabet( i_with_spaces=False)
        click_times =  i_channel_config.getClickTimes()
        idx = alphabet_loader.getLetterPositions()
        click_times_2D  = click_times[idx]
        noise_model =  TickerClickNoise2D()
        #Plot the 2D Click distribution and decision boundaries for one letter
        self.disp.setClickDistributionDisplay2DTarget()
        noise_model.disp = self.disp
        noise_model.letterDecisionBoundaries(alphabet, click_times_2D, i_target_idx, gauss_std, 
            prob_thresh, priors, is_plot, display, score_mask, T, n_std)
        p.xlabel("$t_{1}$ $\\mathrm{(seconds)}$"); p.ylabel("$t_{2}$ $\\mathrm{(seconds)}$")
        p.axis('on'); p.grid('on')
        self.saveFig( "Simple noise 2D click times", self.output_dir + "/simple1")
        #Plot the decision boundaries all letters
        noise_model.allLetterDecisionBoundaries( alphabet,  click_times_2D,  gauss_std, priors, prob_thresh, display, is_plot, T, n_std  )
        self.saveFig( "Simple noise 2D boundaries", self.output_dir + "/simple2")
        
    ################################## Complicated Noise Model Plots        
    
    def dispComplicatedClickDistrPlots( self, i_channel_config, i_target_idx, i_fp_rates, i_fr_rates, i_distr_disp, 
        i_max_val=None, i_aspect=None):
        start_num = 1
        fp_idx = [0, 4]
        fr_idx = [4, 4]
        #Click-time models for 2 clicks
        self.dispComplicated2DClicks( i_channel_config, i_target_idx, i_fp_rates, i_fr_rates, i_distr_disp, 
            fp_idx, fr_idx, start_num, i_max_val, i_aspect)
        #Click-time models for 2 clicks
        self.dispComplicated1DClicks( i_channel_config, i_target_idx, i_fp_rates, i_fr_rates, i_distr_disp, 
            fp_idx, fr_idx, start_num, i_max_val, i_aspect)
            
    def dispComplicated2DClicks(self, i_channel_config, i_target_idx, i_fp_rates, i_fr_rates, i_distr_disp, 
        i_fp_idx, i_fr_idx, i_start_num, i_max_val=None, i_aspect=None):
        """First run with save = True (to save scores) and then 
           save=False (to display/save image)""" 
        self.disp.setClickDistributionDisplay2DTarget()
        save_scores = False
        #The 2D Target Gaussian plots
        self.single2DTargetClickDistrPlots( i_target_idx, i_fp_idx, i_fr_idx, i_fp_rates, i_fr_rates, i_distr_disp, i_start_num, i_save=save_scores)
        #The 2D Decision boundary plots (2 clicks received)
        self.single2DClickDistrPlots( i_fp_idx, i_fr_idx, i_fp_rates, i_fr_rates, i_distr_disp, i_start_num, i_save=save_scores)
     
    def dispComplicated1DClicks(self, i_channel_config, i_target_idx, i_fp_rates, i_fr_rates, i_distr_disp, 
        i_fp_idx, i_fr_idx, i_start_num, i_max_val=None, i_aspect=None):
        #Parameters
        plot_samples = False
        #1 click received
        i_distr_disp.disp.setClickDistributionDisplay1D()
        filename = self.single1DTargetClickDistrPlots(i_target_idx, i_fp_idx, i_fr_idx, i_fp_rates, i_fr_rates, 
            i_distr_disp, i_start_num, self.save, i_max_val, i_aspect )
        #Add samples to the last 1 click received
        #Plot some samples 
        if plot_samples:
            sampler = TickerSampling()
            (samples, gauss_mu, tp_samples, fp_samples) = sampler.sample(i_distr_disp.distr, self.letter, True,  1E6)
            p.plot( fp_samples, np.zeros(len(fp_samples)), 'ko', alpha=0.5)
            p.plot( tp_samples, np.zeros(len(tp_samples)), 'ro')
            if self.save:
                self.disp.saveFig(filename) 
       
    def single1DTargetClickDistrPlots( self, i_target_idx, fp_idxs, fr_idxs, fp_rates, fr_rates, 
            i_distr_disp,  start_num, i_save, i_max_val=None, i_aspect=None ):
        i_distr_disp.disp.setClickDistributionDisplay1D()
        noise_simple = False
        nsamples = 300
        is_many = False
        for k in range(0, len(fp_idxs)):
            (fp_idx, fr_idx) = (fp_idxs[k], fr_idxs[k])
            n = fp_idx*len(fr_rates) + fr_idx
            (fp, fr)  = (fp_rates[fp_idx], fr_rates[fr_idx])
            file_name = self.output_dir + "/complicated1D_" + str(start_num + k)  
            print "***************************************************"
            print "k = ", k, " fp = ", fp, " fr = ", fr, " file_name = ", file_name
            print "***************************************************"
            max_val = self.seqSingleTarget1DClick([fp], [fr], i_distr_disp, i_target_idx, nsamples, noise_simple,is_many)
            ylabel = "P(t_{1}, M=1 \mid  \\boldsymbol{\\theta}, \ell=\mathrm{%s})" % self.letter 
            if (i_max_val is not None) and (i_aspect is not None):
                scale_fac = i_max_val / max_val
                aspect = i_aspect * scale_fac
            else:
                aspect = None
            self.finalise1DPlot(ylabel, aspect, file_name, i_save)
        return file_name 
    
    def single2DTargetClickDistrPlots( self, i_target_idx, fp_idxs, fr_idxs, fp_rates, fr_rates, i_distr_disp,  start_num, i_save ):
        i_distr_disp.disp.setClickDistributionDisplay2DTarget()
        for k in range(0, len(fp_idxs)):
            (fp_idx, fr_idx) = (fp_idxs[k], fr_idxs[k])
            n = fp_idx*len(fr_rates) + fr_idx
            (fp, fr)  = (fp_rates[fp_idx], fr_rates[fr_idx])
            file_name = self.output_dir + "/complicated2D_" + str(start_num + k)  
            print "***************************************************"
            print "k = ", k, " fp = ", fp, " fr = ", fr, " file_name = ", file_name,
            print " save scores = ", i_save, " save file = ", self.save
            print "***************************************************"
            is_many = False
            self.seqSingleTarget2DClick([fp], [fr], i_distr_disp, i_target_idx, 300, i_save, n, is_many=is_many)
            if self.save:
                self.disp.saveFig(file_name)
                    
    def single2DClickDistrPlots( self, fp_idxs, fr_idxs, fp_rates, fr_rates, i_distr_disp, start_num, i_save ):
        i_distr_disp.disp.setClickDistributionDisplay2DTarget()
        for k in range(0, len(fp_idxs)):
            (fp_idx, fr_idx) = (fp_idxs[k], fr_idxs[k])
            n = fp_idx*len(fr_rates) + fr_idx
            (fp, fr)  = (fp_rates[fp_idx], fr_rates[fr_idx])
            file_name = self.output_dir + "/complicatedPosterior2D_" + str(start_num + k)  
            print "***************************************************"
            print "k = ", k, " fp = ", fp, " fr = ", fr, " file_name = ", file_name,
            print " save scores = ", i_save, " save file = ", self.save
            print "***************************************************"
            is_many = True
            self.seqOf2DClickDistrPlots([fp], [fr], i_distr_disp, 300, i_save, n, is_many)
            p.title("")
            if self.save:
                self.disp.saveFig(file_name)
        
    
    #################################### View Plots        
    def seqSingleTarget1DClick(self, fp_rates, fr_rates, i_distr_disp, i_target_idx, i_nsamples, i_noise_simple, is_many ):
        self.disp.newFigure()
        o_max_val = 0
        for (row, fp_rate) in enumerate(fp_rates):
            for (col, fr) in enumerate(fr_rates):
                (i_distr_disp.distr.fp_rate,  i_distr_disp.distr.fr, n ) = (fp_rate, fr, row*len(fr_rates) + col)
                p.subplot(len(fp_rates), len(fr_rates), n+1)
                self.disp.newSubFigure()
                max_val = i_distr_disp.draw1DClick(i_target_idx, i_nsamples, i_noise_simple, is_many)
                if row < (len(fp_rates)-1):
                    p.xlabel("")
                    p.xticks([])
                if col >= 1:
                    p.ylabel("")
                #p.title( "FP=%.2f, FR=%.2f" % (fp_rate, fr) )
                o_max_val = max(o_max_val, max_val)
        return max_val
            
    def seqSingleTarget2DClick(self, fp_rates, fr_rates, i_distr_disp, i_target_idx, i_nsamples, i_save, i_plot_num=None, is_many=True):
        self.disp.newFigure(); p.grid('off')
        for (row, fp_rate) in enumerate(fp_rates):
            for (col, fr) in enumerate(fr_rates):
                (i_distr_disp.distr.fp_rate,  i_distr_disp.distr.fr, n ) = (fp_rate, fr, row*len(fr_rates) + col)
                if not i_save:
                    p.subplot(len(fp_rates), len(fr_rates), n+1)
                    self.disp.newSubFigure()
                else:
                    self.dispCurrentRates(i_distr_disp)
                if i_plot_num is None:
                    plot_num=n
                else:
                    plot_num=i_plot_num 
                cbar = i_distr_disp.draw2DClick( i_target_idx, self.letter, i_nsamples, False, i_save,  plot_num, is_many)
                if not i_save:
                    #p.title( "FP=%.2f, FR=%.2f" % (fp_rate, fr) )
                    if row < (len(fp_rates)-1):
                        p.xlabel("")
                        p.xticks([])
                    if col >= 1:
                        p.ylabel("")
                    elif col < (len(fr_rates)-1):
                        if cbar is not None:
                            cbar.set_label("")
        return cbar

    def seqOf2DClickDistrPlots(self, fp_rates, fr_rates, i_distr_disp,i_nsamples, i_save, i_plot_num, is_many):
        self.disp.newFigure()
        for (row, fp_rate) in enumerate(fp_rates):
            for (col, fr) in enumerate(fr_rates):
                (i_distr_disp.distr.fp_rate,  i_distr_disp.distr.fr, n ) = (fp_rate, fr, row*len(fr_rates) + col)
                if not i_save:
                    p.subplot(len(fp_rates), len(fr_rates), n+1)
                    self.disp.newSubFigure()
                else:
                    self.dispCurrentRates(i_distr_disp)
                if i_plot_num is None:
                    plot_num=n
                else:
                    plot_num=i_plot_num 
                letter_ids_file = "./click_distr_plots/scores2D_%.2d.npy" % plot_num
                scores_file = "./click_distr_plots/letter_ids2D_%.2d.npy" % plot_num
                cbar = i_distr_disp.drawAll2DClicks( letter_ids_file, scores_file, i_nsamples, False, i_save)
                if not i_save:
                    #p.title( "FP=%.2f, FR=%.2f" % (fp_rate, fr) )
                    if row < (len(fp_rates)-1):
                        p.xlabel("")
                        p.xticks([])
                    if col >= 1:
                        p.ylabel("")
                    elif col < (len(fr_rates)-1):
                        cbar.set_label("")
                        
    ############################################## Generic
    
    def saveFig(self, i_title, i_filename):
        print "***************************************************"
        print i_title, "  file_name  = ", i_filename
        print "***************************************************"
        if not self.save:
            return
        self.disp.saveFig(i_filename)
        
    def finalise1DPlot(self, i_ylabel, i_aspect, i_filename, i_save):
        p.ylabel(self.disp.getString(i_ylabel))
        if i_aspect is not None:
            ax = p.gca()
            ax.set_aspect(i_aspect)
        else:
            p.axis("tight")
        p.title("")
        if self.save:
            self.disp.saveFig(i_filename) 
        
    def dispCurrentRates(self ,  i_distr_disp ):
        print "*********************************************************************"
        print "fp_rate = ",   i_distr_disp.distr.fp_rate, "  fr = ",  i_distr_disp.distr.fr
        print "*********************************************************************"             
              
    def getLetter(self, i_channel_config, i_idx):
        return i_channel_config.getAlphabetLoader().getUniqueAlphabet(False)[i_idx]
    
if __name__ ==  "__main__":
    s = SimulationPlots()
    s.clickDistributionPlots()
    if not s.save:
        p.show()

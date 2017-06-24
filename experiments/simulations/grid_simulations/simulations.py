
import sys,cPickle, shutil
sys.path.append('../')
sys.path.append('../../')
from grid_simulation import GridSimulation
from ticker_simulation import TickerSimulation
import numpy as np
import pylab as p
from utils import DispSettings, Utils, PhraseUtils
from scipy.special import erfinv, erf
from channel_config import ChannelConfig 
 
"""Experiments
Trial 1: 
    * A single sentence grid2: Plots for a setting of false pos, false neg, and gauss overlap.
    * A single sentence comparison between Grid2 and Ticker.
Trial 2: Same as trial 2 but with lots of sentences, 
    * A selection of noise settings, averaged over multiple sentences. """

class Simulations():
    ##################################### Init
    def __init__(self, i_trial):
        self.grid = GridSimulation()
        self.ticker = TickerSimulation()
        self.trial_num = i_trial                    #Assign a number to the experiment
        self.root_dir = './results/simulations'     #The root directory to save the results in
        self.grid_file =  "%s/grid2/prob_%.2d.cPickle" %(self.root_dir, self.trial_num)
        self.ticker_file =  "%s/ticker/prob_%.2d.cPickle" %(self.root_dir, self.trial_num)
        self.input_file = "%s/inputs/trial_%.2d.cPickle" %(self.root_dir, self.trial_num)
        self.utils = Utils()
        self.phrase_utils = PhraseUtils()
        #Setup the display parameters
        #Output directory for final plots
        self.output_dir = "./results/simulations/"
        self.save = False
        #Debug
        self.debug = False
        #Display
        self.disp = DispSettings()
        self.disp.setSimulationPlotsDisp()
 
    ##################################### Experiment parameters
    
    def paramsTrialOne(self):
        """
        *Low noise settings (almost no noise)
        *Show effect of latence increase on Grid, not adapting scan delay
        """
        #Parameters applicable to both Ticker and the Grid2
        self.sentence = 'the_quick_brown_fox_jumps_over_the_lazy_dog_.'
        #User parameters
        self.std_range  = [0.05]       #Std deviation range of click delay: 50ms
        self.delay_range = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 1.0, 2.0, 3.0, 4.5]#Reaction delay
        self.fr_range = [0.05]        #False rejection rate: 1/20 is ignored 
        self.fp_range = [0.001]       #False acceptance: 1 every 2 mintues, max 0.4, number of false positives per second
        """* The max total number of scans to try and select something in Ticker & Grid 
           * The user can try for nmax X min number of scans it should take. 
           * Grid: Simulate for T=n_max*(n_rows+n_cols)*len(input_word) scans            
           * Ticker: Simulate for n_max*(len(input_word)) scans. In each scan the whole alphabet is read n_repeat times.  
        """ 
        self.nmax = 5  
        #Ticker params
        self.ticker_samples=1000        #The number of samples to take per sentence in Ticker.
        self.ticker_n_repeat = 2        #The number of times the user is allowed to go through a word
        self.ticker_speed_range = [0.65]#The overlap of voice files
        self.ticker_file_length = 0.21  #Ticker file length
        self.nchannels = 5              #Number of audio channels to use (determines alphabet configuration)
        #Grid 2 Params
        self.grid_speed_range = [0.5] #The length of a scan (in seconds)
        self.grid_delay_range = list(np.array(self.delay_range) + 0.5*self.grid_speed_range[0])
        self.n_errors = 2             #Number of errors allowed before system failure
        self.n_undo = 2               #Number of undo iterations before row selection is undone
       
    def paramsTrialTwo(self):
        """
        * Increase latency on low noise setting.
        * Adapting both models accordingly: 
           - Ticker: end-time delay is increased.
           - Grid: scanning delay for each cell is increased.
        """
        #Parameters applicable to both Ticker and the Grid2
        self.sentence = 'the_quick_brown_fox_jumps_over_the_lazy_dog_.'
        #User parameters
        self.std_range  = [0.05]         #Std deviation range of click delay: 50ms
        self.delay_range = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]#Reaction delay
        self.fr_range = [0.05]          #False rejection rate: 1/20 is ignored 
        self.fp_range = [0.001]         #False acceptance: 1 every 2 mintues, max 0.4, number of false positives per second
        """* The max total number of scans to try and select something in Ticker & Grid 
           * The user can try for nmax X min number of scans it should take. 
           * Grid: Simulate for T=n_max*(n_rows+n_cols)*len(input_word) scans            
           * Ticker: Simulate for n_max*(len(input_word)) scans. In each scan the whole alphabet is read n_repeat times.  
        """ 
        self.nmax = 5  
        #Ticker params
        self.ticker_samples=1000    #The number of samples to take per sentence in Ticker.
        self.ticker_n_repeat = 2    #The number of times the user is allowed to go through a word
        self.ticker_speed_range = [0.65]  # The overlap of voice files
        self.ticker_file_length = 0.21
        self.nchannels = 5 
        #Grid 2 Params
        self.grid_delay_range = list(self.delay_range)
        self.__gridSpeedFromDelayRange(i_min_delay=0.5)
        self.n_errors = 2             #Number of errors allowed before system failure
        self.n_undo = 2               #Number of undo iterations before row selection is undone
       
    def __gridSpeedFromDelayRange(self, i_min_delay=0.4):
        #i_min_delay: The minimum length of a scan (per letter), the click_time delay and std can only make it longer
        self.grid_speed_range = []
        std = self.std_range[0]
        k = 3.0 #Make the delay + k*std longer
        eps = 0.001 #How far the click-time delay from the boundary
        for (n, click_time_delay) in enumerate(self.delay_range): 
            scan_delay = max(i_min_delay, click_time_delay + eps + k*std)
            self.grid_speed_range.append(scan_delay)   
  
    def paramsTrialThree(self):
        """
        * Testing the speed increase.
        """
        #Parameters applicable to both Ticker and the Grid2
        self.sentence = 'the_quick_brown_fox_jumps_over_the_lazy_dog_.'
        #User parameters
        self.std_range  = [0.1]         #Std deviation range of click delay: 50ms
        self.delay_range = [0.0]        #Reaction delay
        self.fr_range = [0.05]          #False rejection rate: 1/20 is ignored 
        self.fp_range = [0.001]         #False positives per second 0.001: 1 every 16 minutes
        self.nmax = 5  
        #Ticker params
        self.ticker_samples=1000    #The number of samples to take per sentence in Ticker.
        self.ticker_n_repeat = 2    #The number of times the user is allowed to go through a word
        self.ticker_speed_range = list(np.arange(0.0,0.85,0.05))  # The overlap of voice files
        self.ticker_speed_range.extend(list(np.arange(0.85, 1.0, 0.005)))
        self.ticker_speed_range = np.array(self.ticker_speed_range)
        self.ticker_file_length = 0.6
        self.nchannels = 5 
        #Grid 2 Params         
        self.__gridSpeedFromTickerSpeed()
        self.n_errors = 2             #Number of errors allowed before system failure
        self.n_undo = 2               #Number of undo iterations before row selection is undone
        print "Grid speed range = ", self.grid_speed_range
    
    def __gridSpeedFromTickerSpeed(self):
        self.grid_speed_range = []
        (delay, std) = (self.delay_range[0], self.std_range[0])
        for (n, overlap) in enumerate(self.ticker_speed_range):
            channel_config = ChannelConfig(i_nchannels=5, i_sound_overlap=overlap, 
                i_file_length=self.ticker_file_length, i_root_dir="../../", i_display=False)
            letter_times = channel_config.getSoundTimes()
            deltas = letter_times[:,1] - letter_times[:,0]
            min_delta = np.min(deltas)
            speed_ratio = min_delta / std
            self.grid_speed_range.append(min_delta)
            #grid_scan_delay = speed_ratio * std
            print "overlap = ", overlap, " min delta = " , min_delta, 
            print " speed ratio = ", speed_ratio
        self.grid_delay_range = delay + 0.5*np.array(self.grid_speed_range)
    
    def paramsTrialFour(self):
        """
        * Testing the effect of the false positive rate.
        """
        #Parameters applicable to both Ticker and the Grid2
        self.sentence = 'the_quick_brown_fox_jumps_over_the_lazy_dog_.'
        #User parameters
        self.std_range  = [0.05]         #Std deviation range of click delay: 50ms
        self.delay_range = [0.15]        #Reaction delay
        self.fr_range = [0.05]           #False rejection rate: 1/20 is ignored 
        self.fp_range =  np.arange(0.0, 0.3, 0.005)   #False positives per second 0.001: 1 every 16 minutes
        self.nmax = 5  
        #Ticker params
        self.ticker_samples=100         #The number of samples to take per sentence in Ticker.
        self.ticker_n_repeat = 2        #The number of times the user is allowed to go through a word
        self.ticker_speed_range = [0.8] #The overlap of voice files
        self.ticker_file_length = 0.21  #Ticker file length
        self.nchannels = 5              #Number of audio channels to use (determines alphabet configuration)
        #Grid 2 Params
        self.grid_delay_range = list(self.delay_range)
        self.__gridSpeedFromDelayRange(i_min_delay=0.4)
        self.n_errors = 2             #Number of errors allowed before system failure
        self.n_undo = 2   
    
    def paramsTrialFive(self):
        """
        * Testing the speed increase, same as trial 3, but with 1 one channel (just Ticker).
        * Speed is a function of the standard deviation.
        """
        self.paramsTrialThree()
        self.nchannels = 1 
          
    def paramsTrialSix(self):
        """
        * Testing the speed increase, same as trial 4, but with 1 one channel (just Ticker).
        * Speed is a function of the standard deviation.
        """
        self.paramsTrialFour()
        self.nchannels = 1 
        
    ##################################### Load 

    def loadParams(self):
        inputs = self.utils.loadPickle(self.input_file)
        self.sentence = inputs['sentence'] 
        #User
        self.delay_range = inputs['delay_range']
        self.std_range = inputs['std_range']
        self.fr_range = inputs['fr_range']
        self.fp_range =  inputs['fp_range']
        self.nmax = inputs['nmax']
        #Grid2
        self.n_errors = inputs['n_errors']
        self.n_undo = inputs['n_undo']
        self.grid_speed_range = inputs['grid_speed_range'] 
        self.grid_delay_range = inputs['grid_delay_range']
        #Ticker
        self.ticker_speed_range = inputs['ticker_speed_range']
        self.ticker_samples = inputs['ticker_samples']
        self.ticker_n_repeat = inputs['ticker_n_repeat']
        self.ticker_file_length = inputs['ticker_file_length']
        self.nchannels = inputs['nchannels'] 
    
    def loadResults(self):
        self.loadParams()
        ticker_results = self.utils.loadPickle(self.ticker_file)
        grid_results = self.utils.loadPickle(self.grid_file)
        return (ticker_results, grid_results)
    
    ##################################### Save 
     
    def saveInputs(self):
        inputs = {'sentence':self.sentence,'delay_range':self.delay_range, 'std_range':self.std_range,
                   'fr_range':self.fr_range, 'fp_range':self.fp_range, 
                   'grid_speed_range': self.grid_speed_range, 'grid_delay_range':self.grid_delay_range,
                   'n_errors': self.n_errors, 'n_undo':self.n_undo, 
                   'ticker_speed_range': self.ticker_speed_range, 'ticker_samples': self.ticker_samples,
                   'ticker_n_repeat': self.ticker_n_repeat, 'nmax': self.nmax, 
                   'ticker_file_length':self.ticker_file_length, 'nchannels':self.nchannels}
        self.utils.savePickle(inputs, self.input_file)
    
    def saveResults(self, i_grid_results, i_ticker_results):
        print "Saving Grid results to " , self.grid_file
        self.utils.savePickle(i_grid_results, self.grid_file )
        #print "Saving Ticker results to " , self.ticker_file
        #self.utils.savePickle(i_ticker_results, self.ticker_file)
  
    ##################################### Main
    
    def compute(self):
        #Load the parameters  
        if self.trial_num == 1:
            self.paramsTrialOne()
        elif self.trial_num == 2:
            self.paramsTrialTwo()
        elif self.trial_num == 3:
            self.paramsTrialThree()
        elif self.trial_num ==4:
            self.paramsTrialFour()
        elif self.trial_num == 5:
            self.paramsTrialFive()
        elif self.trial_num == 6:
            self.paramsTrialSix()
        #Save the parameters
        self.saveInputs()        
        #Compute the stats for a the noise setting
        print "*********************************************************************************"
        print "*********************************************************************************"
        print "sentence = ", self.sentence, " len = ", len(self.sentence) 
        print "grid file = ", self.grid_file 
        print "ticker_file = ", self.ticker_file
        print "*********************************************************************************"
        print "*********************************************************************************"
        if (self.trial_num == 1) or (self.trial_num == 2):
            self.latencyIncrease()
        elif self.trial_num == 3:
            self.speedIncrease()
        elif self.trial_num == 4:
            self.falsePositiveIncrease(i_compute_grid=True)
        elif self.trial_num == 5:
            self.speedIncrease(i_compute_grid = False)
        elif self.trial_num == 6:
            self.falsePositiveIncrease(i_compute_grid=False)
        
    def latencyIncrease(self, i_increase_grid_scan_delay=False): 
        """  * Increasing the scan delay to compare speed and accuracy changes between the Grid and Ticker.  """ 
        #Ticker & Grid parameters
        (draw_tikz, debug_grid, disp_grid, disp_ticker) = (False, False, False, False)
        (scan_delay, n_max) = (self.grid_speed_range[0], self.nmax)
        (fp_rate, fr, std) = (self.fp_range[0], self.fr_range[0], self.std_range[0])
        #Grid-specific parameters
        (n_errors, n_undo) = (self.n_errors, self.n_undo)
        #Ticker-specfic parameters
        (nsamples, nrepeat) = (self.ticker_samples, self.ticker_n_repeat)
        overlap = self.ticker_speed_range[0] #Ticker speed
        #The final results
        (grid_results, ticker_results) = ([], [])
        for (n, click_time_delay) in enumerate(self.delay_range): 
            grid_click_time = self.grid_delay_range[n]
            if self.trial_num == 2:
                #Increase the grid scan delay according to the click time delay
                scan_delay = self.grid_speed_range[n]
            disp_str=  "n=%d, ticker click delay=%1.3f,  grid scan_delay=%1.3f" % (n, click_time_delay, scan_delay)
            disp_str+=(" grid click delay=%1.3f, sentence=%s" % (grid_click_time, self.sentence)) 
            self.utils.dispMsg(True, i_msg=disp_str, i_disp_msg=True) 
            grid_params = (n_errors, n_undo, fr, fp_rate, scan_delay, grid_click_time, std, n_max, draw_tikz, disp_grid)
            self.grid.init(i_scan_delay=scan_delay, i_click_time_delay=grid_click_time) 
            grid_results.append(self.grid.compute(self.sentence, grid_params))
            #ticker_results.append(self.ticker.compute( self.nchannels, self.ticker_file_length, self.sentence, nsamples, nrepeat, 
            #    n_max, overlap, click_time_delay, std, fr, fp_rate, disp_ticker))
        self.saveResults(grid_results, ticker_results)

    def speedIncrease(self, i_compute_grid=True): 
        """  * Do the computations to be able to compare the speed between Grid2 and Ticker for a specific noise setting.""" 
        #Ticker & Grid parameter50, scan_delay=0.090, grid_s
        (draw_tikz, debug_grid, disp_grid, disp_ticker) = (False, False, False, False) 
        (fp_rate, fr, std) = (self.fp_range[0], self.fr_range[0], self.std_range[0])
        n_max = self.nmax 
        #Grid-specific parameters
        (n_errors, n_undo) = (self.n_errors, self.n_undo)
        #Ticker-specfic parameters
        click_time_delay = self.delay_range[0]
        (nsamples, nrepeat) = (self.ticker_samples, self.ticker_n_repeat)
        (grid_results, ticker_results) = ([], [])
        for (n, scan_delay) in enumerate(self.grid_speed_range):
            overlap = self.ticker_speed_range[n]
            disp_str = "n=%d, overlap=%1.3f, scan_delay=%1.3f, sentence=%s" % (n, overlap, scan_delay, self.sentence) 
            self.utils.dispMsg(True, i_msg=disp_str, i_disp_msg=True) 
            if i_compute_grid:
                grid_click_time = self.grid_delay_range[n]
                grid_params = (n_errors, n_undo, fr, fp_rate, scan_delay, grid_click_time, std, n_max, draw_tikz, disp_grid)
                self.grid.init(i_scan_delay=scan_delay, i_click_time_delay=grid_click_time) 
                grid_results.append(self.grid.compute(self.sentence, grid_params))
            #ticker_results.append(self.ticker.compute(self.nchannels, self.ticker_file_length, self.sentence, nsamples, nrepeat, 
            #    n_max, overlap, click_time_delay, std, fr, fp_rate, disp_ticker))
        self.saveResults(grid_results, ticker_results)

    def falsePositiveIncrease(self, i_compute_grid=True): 
        """  * Compare the false-positive rate tolerance between Grid2 and Ticker""" 
        #Ticker & Grid parameter50, scan_delay=0.090, grid_s
        (draw_tikz, debug_grid, disp_grid, disp_ticker) = (False, False, False, False) 
        (fr, std) = (self.fr_range[0], self.std_range[0])
        n_max = self.nmax 
        #Grid-specific parameters
        (n_errors, n_undo) = (self.n_errors, self.n_undo)
        scan_delay = self.grid_speed_range[0]
        grid_click_time = self.grid_delay_range[0]
        #Ticker-specfic parameters
        overlap = self.ticker_speed_range[0]
        click_time_delay = self.delay_range[0]
        (nsamples, nrepeat) = (self.ticker_samples, self.ticker_n_repeat)
        (grid_results, ticker_results) = ([], [])
        for (n, fp_rate) in enumerate(self.fp_range):
            disp_str = "n=%d, fp_rate=%1.3f, sentence=%s" % (n, fp_rate, self.sentence) 
            self.utils.dispMsg(True, i_msg=disp_str, i_disp_msg=True) 
            if i_compute_grid:
                grid_params = (n_errors, n_undo, fr, fp_rate, scan_delay, grid_click_time, std, n_max, draw_tikz, disp_grid)
                self.grid.init(i_scan_delay=scan_delay, i_click_time_delay=grid_click_time) 
                grid_results.append(self.grid.compute(self.sentence, grid_params))
            #ticker_results.append(self.ticker.compute(self.nchannels, self.ticker_file_length, self.sentence, nsamples, nrepeat, 
            #    n_max, overlap, click_time_delay, std, fr, fp_rate, disp_ticker))
        self.saveResults(grid_results, ticker_results)

    #################################### Get

    def getGridResults(self, i_r):
        r = np.array(i_r)
        (min_scans, avg_scans, std_scans) = (r[:,0],r[:,1], r[:,2])
        (min_wpm, avg_wpm, std_wpm) = (r[:,3], r[:,4], r[:,5])
        (avg_err_rate, std_err_rate) = (r[:,6], r[:,7])
        (avg_cpc, std_cpc) = (r[:,8], r[:,9])
        return (avg_cpc, std_cpc, avg_wpm, std_wpm, avg_err_rate, std_err_rate) 
        
    def getTickerResults(self, i_r):
        r = np.array(i_r)
        (avg_cpc, std_cpc) = (r[:,0], r[:,1])
        (avg_wpm, std_wpm) = (r[:,2], r[:,3])
        (avg_err_rate, std_err_rate) = (r[:,4], r[:,5])
        return (avg_cpc, std_cpc, avg_wpm, std_wpm, avg_err_rate, std_err_rate) 
        
    #################################### Plot trials
    
    def plot(self):
        if self.trial_num == 1:
            self.plotTrialOne()
        elif self.trial_num == 2:
            self.plotTrialTwo()
        elif self.trial_num == 3:
            self.plotTrialThree()
        elif self.trial_num == 4:
            self.plotTrialFour()
        elif self.trial_num == 5:
            self.plotTrialFive()
        elif self.trial_num == 6:
            self.plotTrialSix()
        p.show()
    
    def plotBestWpmTicker(self, i_figs, i_fig_num):
        #Plot the theoretical optimum
        p.figure(i_figs[i_fig_num].number)
        #Compute the theoretical best result
        n_channels = self.nchannels
        overlap = self.ticker_speed_range[0] 
        self.ticker_file_length = self.ticker_file_length
        channel_config = ChannelConfig(i_nchannels=n_channels, i_sound_overlap=overlap, i_file_length=0.21, i_root_dir="../../", i_display=False)
        letter_times = channel_config.getSoundTimes()
        total_time = letter_times[-1,-1]
        deltas = letter_times[:,1] - letter_times[:,0]
        avg_delta= np.mean(deltas)
        min_delta = np.min(deltas)
        wpm = 60.0 / (letter_times.shape[0] * avg_delta * 5.0)
        print "*****************************************************************"
        print "deltas = "
        print deltas
        print "avg delta = ", avg_delta, " min delta = ", min_delta, "wpm = ", wpm
        wpm = wpm * np.ones(len(self.delay_range))
        p.plot(self.delay_range, wpm, 'r--', linewidth=3)
    
    def plotTrialOneAndTwo(self, i_figs=None, i_save=True, i_ignore_ticker=False):
        #Latency increase, Ticker (5 channels) and Grid
        self.loadParams()
        (ticker_results, grid_results) = self.loadResults() 
        scan_delay = self.grid_speed_range[0]
        std = self.std_range[0]
        for (n, click_time_delay) in enumerate(self.delay_range):
            grid_click_time = self.grid_delay_range[n]
            if self.trial_num == 2:
                #Increase the grid scan delay according to the click time delay
                scan_delay = self.grid_speed_range[n]
            disp_str=  "n=%d, std=%1.3f, ticker click delay=%1.3f,  grid scan_delay=%1.3f" % (n, std, click_time_delay, scan_delay)
            disp_str+=(" grid click delay=%1.3f, sentence=%s" % (grid_click_time, self.sentence)) 
            self.utils.dispMsg(True, i_msg=disp_str, i_disp_msg=True)
            self.dispResult(ticker_results, grid_results, n)
        xlabels = ['$\\Delta$ $\\mathrm{(seconds)}$', '$\\Delta$ $\\mathrm{(seconds)}$',  '$\\Delta$ $\\mathrm{(seconds)}$']
        o_figs = self.plotResults(grid_results,ticker_results, self.delay_range, xlabels, i_save=i_save, 
            i_figs=i_figs, i_plot_params=True, i_plot_2D=False, i_ignore_ticker=i_ignore_ticker)
        return o_figs

    def plotTrialOne(self):
        self.loadParams()
        figs = self.plotTrialOneAndTwo(i_save=False)
        return figs
    
    def plotTrialTwo(self):
        #Plot Ticker and Grid together for long paper
        self.__init__(1)
        figs = self.plotTrialOneAndTwo(i_save=False)
        self.plotBestWpmTicker(figs, i_fig_num=0)
        x_max = np.max(self.delay_range)
        print "**************************************"
        self.__init__(2)
        self.plotTrialOneAndTwo(i_figs=figs, i_save=True)
        x_max = min(x_max, np.max(self.delay_range))
        x_tick_labels = []
        for n in p.arange(x_max):
            x_tick_labels.append("$%d$" % n)
        for (n, fig) in enumerate(figs):
            p.figure(fig.number)
            p.xlim(xmax=x_max)
            p.xticks( p.arange(x_max), tuple(x_tick_labels)) 
            self.disp.saveFig("%strial%d_%d" % (self.output_dir, 2, n) )
        #Plot only the Grid results for the CHI paper
        self.__init__(1)
        figs2 = self.plotTrialOneAndTwo(i_save=False,  i_ignore_ticker=True)
        self.__init__(2)
        self.plotTrialOneAndTwo(i_figs=figs2, i_save=True, i_ignore_ticker=True)
        for (n, fig) in enumerate(figs2):
            p.figure(fig.number)
            p.xlim(xmax=x_max)
            p.xticks( p.arange(x_max), tuple(x_tick_labels)) 
            self.disp.saveFig("%strial_grid%d_%d" % (self.output_dir, 2, n) )
       
    def plotTrialThree(self, i_save=True, i_ignore_ticker=False):
        #Speed increase, Ticker (5 channels) and Grid
        self.loadParams()
        (ticker_results, grid_results) = self.loadResults() 
        std = self.std_range[0]
        std_ratios = []
        for (n, scan_delay) in enumerate(self.grid_speed_range):
            overlap = self.ticker_speed_range[n]
            grid_click_time = self.grid_delay_range[n]
            ratio = scan_delay / std
            std_ratios.append(ratio)
            disp_str = "overlap=%1.3f, scan_delay=%1.3f, " % (overlap, scan_delay)
            disp_str += ("grid_click_time=%1.3f, std_ratio=%1.3f" % (grid_click_time, ratio))
            self.utils.dispMsg(True, i_msg=disp_str, i_disp_msg=True)
            self.dispResult(ticker_results, grid_results, n)
        xlabels = ['$T_{\mathrm{S}}/\sigma$', '$T_{\mathrm{S}}/\sigma$',  '$T_{\mathrm{S}}/\sigma$']
        return self.plotResults(grid_results,ticker_results, np.array(self.grid_speed_range)/std, xlabels, i_save=i_save, i_ignore_ticker=i_ignore_ticker)
    
    def plotTrialFour(self, i_save=True, i_ignore_ticker=False):
        #False positive rate increase, Ticker (5 channels) and  Grid
        self.loadParams()
        (ticker_results, grid_results) = self.loadResults() 
        scan_delay = self.grid_speed_range[0]
        for (n, fp_rate) in enumerate(self.fp_range):
            disp_str = "n=%d, fp_rate=%1.3f, sentence=%s, scan_delay=%1.4f" % (n, fp_rate, self.sentence, scan_delay) 
            self.utils.dispMsg(True, i_msg=disp_str, i_disp_msg=True) 
            self.dispResult(ticker_results, grid_results, n)
        xlabels = ['$\lambda$', '$\lambda$',  '$\lambda$']
        return self.plotResults(grid_results,ticker_results, self.fp_range, xlabels, i_save=i_save, i_ignore_ticker=i_ignore_ticker)
        
    def plotTrialFive(self):
        #Speed increase, Ticker (1 channel)
        self.__init__(3)
        figs = self.plotTrialThree(i_save=False)
        self.__init__(5)
        (ticker_results, grid_results) = self.loadResults()
        ticker_results = self.getTickerResults(ticker_results)
        std = self.std_range[0]
        ticker_line_handles = self.__plotResults(figs, np.array(self.grid_speed_range)/std, ticker_results, 'g', 'Ticker_1', i_save=True)
        self.plotResultsAgainstWpm(figs, ticker_results, 'g', 'Ticker_1', i_save=True, i_fig_offset=3)
        #Plot without Ticker
        self.__init__(3)
        figs2 = self.plotTrialThree(i_save=False, i_ignore_ticker=True)
        #Do without Grid
        for (n, fig2) in enumerate(figs2):
            p.figure(fig2.number)
            #p.xlim(xmax=4)
            #p.xticks( p.arange(x_max), tuple(x_tick_labels)) 
            self.disp.saveFig("%strial_grid%d_%d" % (self.output_dir, 5, n) )
    
    def plotTrialSix(self):
        #False positive rate increase, Ticker (1 channel)
        self.__init__(4)
        figs = self.plotTrialFour(i_save=False)
        self.__init__(6)
        (ticker_results, grid_results) = self.loadResults()
        ticker_results = self.getTickerResults(ticker_results)
        ticker_line_handles = self.__plotResults(figs, self.fp_range, ticker_results, 'g', 'Ticker_1', i_save=True)
        self.plotResultsAgainstWpm(figs, ticker_results, 'g', 'Ticker_1', i_save=True, i_fig_offset=3)
        for (n, fig) in enumerate(figs):
            p.figure(fig.number)
            locs, labels = p.xticks()
            locs = locs[range(0,len(locs),2)]
            labels=[]
            for loc in locs: 
                labels.append("$%.1f$" % loc)
            p.xticks(locs, tuple(labels))
            self.disp.saveFig("%strial%d_%d" % (self.output_dir, 6, n) )
    
        #Plot without Ticker
        self.__init__(4)
        figs2 = self.plotTrialFour(i_save=False, i_ignore_ticker=True)
        #Do without Grid
        for (n, fig2) in enumerate(figs2):
            p.figure(fig2.number)
            #p.xlim(xmax=4)
            #p.xticks( p.arange(x_max), tuple(x_tick_labels)) 
            locs, labels = p.xticks()
            locs = locs[range(0,len(locs),2)]
            labels=[]
            for loc in locs: 
                labels.append("$%.1f$" % loc)
            p.xticks(locs, tuple(labels))
            self.disp.saveFig("%strial_grid%d_%d" % (self.output_dir, 6, n) )
            
    
    ################################################# Plot general
    
    def plotResults(self, i_grid_results, i_ticker_results, i_x, i_xlabels, i_save=True, i_figs=None, i_plot_params=True, i_plot_2D=True, i_ignore_ticker=False):
        """i_plot_params=True: Plot wpm, cpc and char errors agains the parameter that was varied. 
           i_plot_2D=Plot wpm against cpc and wpm agains char errors."""
        (grid_results, ticker_results) = (self.getGridResults(i_grid_results), self.getTickerResults(i_ticker_results))
        o_figs = []
        fig_offset = 0
        self.disp.setSimulationPlotsDisp()
        if self.trial_num==2:
            grid_color = 'g'
        else:
            grid_color = 'k'
        if i_plot_params:
            figs = self.plotParams(i_figs, i_x, grid_results, ticker_results, i_xlabels, grid_color, i_save, i_ignore_ticker)
            o_figs.extend(figs)
            fig_offset = 3
        if i_plot_2D:
            figs_2D = self.plot2D(i_figs, grid_results, ticker_results, grid_color, i_save, fig_offset, i_ignore_ticker) 
            o_figs.extend(figs_2D)
        return o_figs
    
    def plotParams(self, i_figs, i_x, i_grid_results, i_ticker_results, i_xlabels, i_grid_color, i_save, i_ignore_ticker=False):
        """Plot wpm, cpc and char errors agains the parameter that was varied."""
        if i_figs is not None:
            figs = list(i_figs[0:3])
        else:
            figs= [ self.disp.newFigure() for n in range(0,3)]
        ylabels = ['$wpm$', '$\# clicks$ $\\mathrm{(cpc)}$', '$\# errors$ $(\%)$']
        grid_line_handles = self.__plotResults(figs, i_x,  i_grid_results, i_grid_color, 'Grid', i_save, i_ignore_ticker)
        if not i_ignore_ticker:
            ticker_line_handles = self.__plotResults(figs, i_x, i_ticker_results, 'r', 'Ticker', i_save, i_ignore_ticker)
        for (n, fig) in enumerate(figs):
            p.figure(fig.number)
            p.xlabel(i_xlabels[n])
            p.ylabel(ylabels[n])
            #p.legend([grid_line_handles[n], ticker_line_handles[n]], ['Grid', 'Ticker'])
        return figs
    
    def plot2D(self, i_figs, i_grid_results, i_ticker_results, i_grid_color, i_save, i_fig_offset, i_ignore_ticker=False):
        """Plot wpm against cpc and wpm agains char errors."""
        if i_figs is not None:
            figs_2D = list(i_figs[fig_offset:])
        else:
            figs_2D = [self.disp.newFigure() for n in range(0,2)] 
        self.plotResultsAgainstWpm(figs_2D, i_grid_results, i_grid_color, 'Grid', i_save, i_fig_offset)
        if not i_ignore_ticker:
            self.plotResultsAgainstWpm(figs_2D, i_ticker_results, 'r', 'Ticker', i_save, i_fig_offset)
        return figs_2D
        
    def plotResultsAgainstWpm(self, i_figs,  i_results, i_color, i_label, i_save,  i_fig_offset, i_ignore_ticker=False):
        (avg_cpc, std_cpc, avg_wpm, std_wpm, avg_err_rate, std_err_rate) = i_results
        if i_fig_offset > len(i_figs):
            fig_offset = 0
        else:
            fig_offset = i_fig_offset
        #Wpm vs cpc
        p.figure(i_figs[0+fig_offset].number)
        p.plot(avg_wpm, avg_cpc, i_color, linewidth=4, label=i_label)
        p.plot(avg_wpm, avg_cpc, i_color+'o', linewidth=4, markersize=7)
        p.xlabel('$wpm$')
        p.ylabel('$\# clicks$ $\\mathrm{(cpc)}$')
        if i_save:
            fig_name = "%strial" % self.output_dir
            if i_ignore_ticker:
                fig_name += "_grid"
            self.disp.saveFig("%s%d_%d" % (fig_name, self.trial_num, i_fig_offset) )
        #Wpm vs error rate
        p.figure(i_figs[1+fig_offset].number)
        p.plot(avg_wpm, avg_err_rate, i_color, linewidth=4, label=i_label)
        p.plot(avg_wpm, avg_err_rate, i_color+'o', linewidth=4, markersize=7)
        p.xlabel('$wpm$')
        p.ylabel('$\# errors$ $(\%)$')
        if i_save:
            fig_name = "%strial" % self.output_dir
            if i_ignore_ticker:
                fig_name += "_grid"
            self.disp.saveFig("%s%d_%d" % (fig_name, self.trial_num, i_fig_offset+1) )
        
    def __plotResults(self, i_figs, i_x, i_r, i_color, i_label, i_save, i_ignore_ticker=False):
        (avg_cpc, std_cpc, avg_wpm, std_wpm, avg_err_rate, std_err_rate) = i_r
        line_handles = []
        #wpm
        p.figure(i_figs[0].number)
        line_handles.append(self.__plotErrorBar(i_x, avg_wpm, std_wpm, i_color, i_label))
        #cpc
        p.figure(i_figs[1].number)
        line_handles.append(self.__plotErrorBar(i_x, avg_cpc, std_cpc, i_color, i_label))
        #character error rate
        p.figure(i_figs[2].number)
        line_handles.append(self.__plotErrorBar(i_x, avg_err_rate, std_err_rate, i_color, i_label))
        #save the figures
        if i_save:
            for n in range(0,3):
                p.figure(i_figs[n].number)
                fig_name = "%strial" % self.output_dir
                if i_ignore_ticker:
                    fig_name += "_grid"
                self.disp.saveFig("%s%d_%d" % (fig_name, self.trial_num, n) )
        return line_handles
       
    def __plotErrorBar(sef, i_x, i_avg, i_std, i_color, i_label):
        line_handle, = p.plot(i_x, i_avg, i_color, linewidth=4, label=i_label)
        p.plot(i_x, i_avg, i_color+'o', linewidth=4, markersize=7)
        for (n, std) in enumerate(i_std):
            p.plot([i_x[n], i_x[n]], [max(0,i_avg[n]-std), i_avg[n]+std], i_color,  linewidth=2,alpha=0.6)
        return line_handle 
    
    ######################################### Display
    
    def dispResult(self, i_ticker_results, i_grid_results, i_index):
        print "Ticker: ",
        self.utils.dispResults(i_ticker_results[i_index])
        print "Grid  : ",
        self.grid.displayResults(np.array(i_grid_results[i_index]), True, False)
    
def runApps():
    #trials to plot for paper: [2,5,6]
    trials = [2] #1,2,3,4,5,6]
    for (n, trial_num) in enumerate(trials):
        app = Simulations(i_trial=trial_num)
        #app.compute()
        app.plot()
        
if __name__ ==  "__main__":
    runApps()
     
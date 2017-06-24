
import sys,cPickle, shutil
sys.path.append('../')
sys.path.append('../../')
from grid_simulation import GridSimulation 
from grid_to_ticker_simulation import GridToTickerSimulation
import numpy as np
import pylab as p
from utils import DispSettings, Utils, PhraseUtils
from scipy.special import erfinv, erf
   
class Simulations():
    ##################################### Init
    def __init__(self, i_trial):
        self.grid = GridSimulation() 
        self.grid_to_ticker = GridToTickerSimulation()
        self.trial_num = i_trial                    #Assign a number to the experiment
        self.root_dir = './results/grid_to_ticker_trials'     #The root directory to save the results in
        self.grid_file =  "%s/grid2/prob_%.2d.cPickle" %(self.root_dir, self.trial_num)
        self.grid_to_ticker_file =  "%s/grid_to_ticker/prob_%.2d.cPickle" %(self.root_dir, self.trial_num)  
        self.grid_best_file =  "%s/grid_best/prob_%.2d.cPickle" %(self.root_dir, self.trial_num)  
        self.grid_to_ticker_best_file =  "%s/grid_to_ticker_best/prob_%.2d.cPickle" %(self.root_dir, self.trial_num)  
        
        self.input_file = "%s/inputs/trial_%.2d.cPickle" %(self.root_dir, self.trial_num)
        self.utils = Utils()
        self.phrase_utils = PhraseUtils()
        #Setup the display parameters
        #Output directory for final plots
        self.output_dir = "./results/simulations/grid_to_ticker_trials"
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
        self.std_range  = [0.000001]  #Std deviation range of click delay: 50ms
        self.scan_delay_range = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0] ) #Standard grid2 range
        self.fr_range = [0.0]         #False rejection rate: 1/20 is ignored 
        self.fp_range = [0.0]         #False acceptance: 1 every 2 mintues, max 0.4, number of false positives per second
        """* The max total number of scans to try and select something in Ticker & Grid 
           * The user can try for nmax X min number of scans it should take. 
           * Grid: Simulate for T=n_max*(n_rows+n_cols)*len(input_word) scans            
           * Ticker: Simulate for n_max*(len(input_word)) scans. In each scan the whole alphabet is read n_repeat times.  
        """ 
        #Grid 2 Params
        self.min_delay = 0.5
        self.n_errors = 2             #Number of errors allowed before system failure
        self.n_undo = 2               #Number of undo iterations before row selection is undone
        self.n_max = 5
        self.click_time_range = self.scan_delay_range - 2.0*self.std_range[0]
        
    def paramsTrialTwo(self):
        self.paramsTrialOne() 
        self.std_range = [0.000001]
        self.fr_range = [0.1]
        self.click_time_range = self.scan_delay_range - 2.0*self.std_range[0]
        
    def paramsTrialThree(self):
        self.paramsTrialOne() 
        self.min_delay = 0.05
    
    def paramsTrialFour(self):
        self.paramsTrialOne() 
        self.std_range = [0.000001]
        self.fr_range = [0.5]
        self.click_time_range = self.scan_delay_range - 2.0*self.std_range[0]
        
        
    ##################################### Load 

    def loadParams(self):
        inputs = self.utils.loadPickle(self.input_file)
        self.sentence = inputs['sentence'] 
        #User
        self.scan_delay_range = inputs['scan_delay_range']
        self.std_range = inputs['std_range']
        self.fr_range = inputs['fr_range']
        self.fp_range =  inputs['fp_range']
        self.n_max = inputs['n_max']
        #Grid2
        self.min_delay = inputs['min_delay']
        self.n_errors = inputs['n_errors']
        self.n_undo = inputs['n_undo']
        self.click_time_range = inputs['click_time_range'] 
    
    def loadResults(self):
        self.loadParams()
        grid_results = self.utils.loadPickle(self.grid_file)
        grid_to_ticker_results = self.utils.loadPickle(self.grid_to_ticker_file)
        
        grid_best_results = self.utils.loadPickle(self.grid_best_file)
        grid_to_ticker_best_results = self.utils.loadPickle(self.grid_to_ticker_best_file)
        return grid_to_ticker_results, grid_results, grid_to_ticker_best_results, grid_best_results
    
    ##################################### Save 
     
    def saveInputs(self ):
        inputs = {'sentence':self.sentence,'scan_delay_range':self.scan_delay_range, 'std_range':self.std_range,
                   'fr_range':self.fr_range, 'fp_range':self.fp_range, 
                   'n_errors': self.n_errors, 'n_undo':self.n_undo, 'min_delay' : self.min_delay,
                    'click_time_range' : self.click_time_range, 'n_max' : self.n_max}
        self.utils.savePickle(inputs, self.input_file)
    
    def saveResults(self, i_grid_results, i_grid_to_ticker_results, i_grid_best_results, i_grid_to_ticker_best_results):
        print "Saving Grid results to " , self.grid_file
        self.utils.savePickle(i_grid_results, self.grid_file )
        print "Saving Grid to ticker results to " , self.grid_to_ticker_file
        self.utils.savePickle(i_grid_to_ticker_results, self.grid_to_ticker_file )
        print "Saving Grid best results to " , self.grid_best_file  
        self.utils.savePickle(i_grid_best_results, self.grid_best_file )
        print "Saving Grid to ticker best results to " , self.grid_to_ticker_file
        self.utils.savePickle(i_grid_to_ticker_best_results, self.grid_to_ticker_best_file)
      
    ##################################### Main
    
    def compute(self):
        #Load the parameters  
        if self.trial_num == 1:
            self.paramsTrialOne()
        elif self.trial_num == 2:
            self.paramsTrialTwo()
        elif self.trial_num == 3:
            self.paramsTrialThree()
        elif self.trial_num == 4:
            self.paramsTrialFour()
        #Save the parameters
        self.saveInputs()        
        #Compute the stats for a the noise setting
        print "*********************************************************************************"
        print "*********************************************************************************"
        print "sentence = ", self.sentence, " len = ", len(self.sentence) 
        print "grid file = ", self.grid_file  
        print "grid to ticker file = ", self.grid_to_ticker_file
        print "*********************************************************************************"
        print "*********************************************************************************"
        self.latencyIncrease()
     
        
    def latencyIncrease(self): 
        """  * Increasing the scan delay to compare speed and accuracy changes between the Grid and Ticker.  """ 
        #Ticker & Grid parameters
        (draw_tikz, debug_grid, disp_grid, disp_ticker) = (False, False, False, False)
        (fp_rate, fr, std) = (self.fp_range[0], self.fr_range[0], self.std_range[0])
        #Grid-specific parameters
        (n_errors, n_undo) = (self.n_errors, self.n_undo)
        #The final results
        (grid_results, grid_to_ticker_results, grid_best_results, grid_to_ticker_best_results)  = ([], [], [], [])
        grid_to_ticker_scan_delay = self.min_delay
        
        for (n, click_time) in enumerate(self.click_time_range):  
            grid_scan_delay = self.scan_delay_range[n]
            disp_str=  "n=%d, click_time=%1.3f,  grid scan_delay=%1.3f" % (n, click_time, grid_scan_delay)
            disp_str+=(" grid  to_ticker scan delay=%1.3f, sentence=%s" % (grid_to_ticker_scan_delay, self.sentence)) 
            self.utils.dispMsg(True, i_msg=disp_str, i_disp_msg=True) 
            #Set the grid  parameters
            grid_params = (n_errors, n_undo, fr, fp_rate, grid_scan_delay, click_time, std, self.n_max, draw_tikz, disp_grid)
            self.grid.init(i_scan_delay=grid_scan_delay, i_click_time_delay=click_time, i_std=std, i_config=None) 
            print "------------------------------------------------------------------------------------------"
            print "Testing Grid 2 with scan time ", grid_scan_delay, " and click time of " , click_time
            grid_results.append(self.grid.compute(self.sentence, grid_params))
            min_scans = self.grid.getMinScans(self.sentence)
            min_wpm =  self.grid.scansToWpm(min_scans, self.sentence)
            grid_best_results.append( (min_scans, min_wpm) )
            #Set the grid2ticker parameters
            grid_to_ticker_params = (n_errors, n_undo, fr, fp_rate, grid_to_ticker_scan_delay,  click_time, std, self.n_max, draw_tikz, disp_grid)
            self.grid_to_ticker.init(i_scan_delay=grid_to_ticker_scan_delay, i_click_time_delay=click_time, i_std=std, i_config=None) 
            self.grid_to_ticker.setGroupDelta(grid_scan_delay)
            grid_to_ticker_results.append(self.grid_to_ticker.compute(self.sentence, grid_to_ticker_params))
            min_scans = self.grid_to_ticker.getMinScans(self.sentence, i_last_scan_time=self.grid_to_ticker.last_scan_time)
            min_wpm =  self.grid_to_ticker.scansToWpm(min_scans, self.sentence)
            print "------------------------------------------------------------------------------------------"
            print "Testing Grid2Ticker with scan time ", grid_to_ticker_scan_delay, " and click time of " , click_time
            grid_to_ticker_best_results.append( (min_scans, min_wpm) )
            
        self.saveResults(grid_results, grid_to_ticker_results, grid_best_results, grid_to_ticker_best_results)
 
    #################################### Get

    def getGridResults(self, i_r):
        r = np.array(i_r)
        (min_scans, avg_scans, std_scans) = (r[:,0],r[:,1], r[:,2])
        (min_wpm, avg_wpm, std_wpm) = (r[:,3], r[:,4], r[:,5])
        (avg_err_rate, std_err_rate) = (r[:,6], r[:,7])
        (avg_cpc, std_cpc) = (r[:,8], r[:,9])
        return (avg_cpc, std_cpc, avg_wpm, std_wpm, avg_err_rate, std_err_rate) 
        
   
    #################################### Plot trials
    
    def plot(self, i_figs=None, i_save=True, i_ignore_ticker=False):
        #Latency increase, Ticker (5 channels) and Grid
        self.loadParams()
        (grid_to_ticker_results, grid_results,  grid_to_ticker_best_results, grid_best_results) = self.loadResults() 
        std = self.std_range[0]
        for (n, click_time_delay) in enumerate(self.click_time_range): 
            scan_delay = self.scan_delay_range[n]
            disp_str=  "n=%d, std=%1.3f, click time delay=%1.3f,  grid scan delay=%1.3f" % (n, std, click_time_delay, scan_delay)
            disp_str+=(" grid click delay=%1.3f, sentence=%s" % (click_time_delay, self.sentence)) 
            self.utils.dispMsg(True, i_msg=disp_str, i_disp_msg=True)
            self.dispResult(grid_to_ticker_results, grid_results, n)
        xlabels = ['$\\Delta$ $\\mathrm{(seconds)}$', '$\\Delta$ $\\mathrm{(seconds)}$',  '$\\Delta$ $\\mathrm{(seconds)}$']
        o_figs = self.plotResults(grid_results, grid_to_ticker_results,  grid_best_results, grid_to_ticker_best_results, self.scan_delay_range, xlabels, i_save=i_save, 
            i_figs=i_figs, i_plot_params=True, i_plot_2D=False, i_ignore_ticker=i_ignore_ticker)
        p.show()
        return o_figs
     
    
    ################################################# Plot general
    
    def plotResults(self, i_grid_results, i_grid_to_ticker_results, i_grid_best_results, i_grid_to_ticker_best_results, i_x, i_xlabels, i_save=True, i_figs=None, i_plot_params=True, i_plot_2D=True, i_ignore_ticker=False):
        """i_plot_params=True: Plot wpm, cpc and char errors agains the parameter that was varied. 
           i_plot_2D=Plot wpm against cpc and wpm agains char errors."""
        (grid_results, grid_to_ticker_results) = (self.getGridResults(i_grid_results), self.getGridResults(i_grid_to_ticker_results) )
        grid_best_wpm = np.array( self.utils.loadPickle( self.grid_best_file ) )[:,1]
        grid_to_ticker_best_wpm = np.array( self.utils.loadPickle( self.grid_to_ticker_best_file ) )[:,1]
              
        o_figs = []
        fig_offset = 0
        self.disp.setSimulationPlotsDisp()
      
        if i_plot_params:
            figs = self.plotParams(i_figs, i_x, grid_results,  grid_to_ticker_results, grid_best_wpm, grid_to_ticker_best_wpm, i_xlabels,  i_save )
            o_figs.extend(figs)
            fig_offset = 3
            
         
        return o_figs
    
    def plotParams(self, i_figs, i_x, i_grid_results, i_grid_to_ticker_results, i_grid_best_wpm, i_grid_to_ticker_best_wpm, i_xlabels, i_save ):
        """Plot wpm, cpc and char errors agains the parameter that was varied."""
        if i_figs is not None:
            figs = list(i_figs[0:3])
        else:
            figs= [ self.disp.newFigure() for n in range(0,1)]
        ylabels = ['$wpm$']
        
        (avg_cpc_grid, std_cpc_grid, avg_wpm_grid, std_wpm_grid, avg_err_rate_grid, std_err_rate_grid) = i_grid_results   
        grid_line_handles = self.__plotResults(figs, i_x, avg_wpm_grid, std_wpm_grid, 'k', 'Grid', i_save, False)
        
        (avg_cpc_grid_to_ticker, std_cpc_grid_to_ticker, avg_wpm_grid_to_ticker, std_wpm_grid_to_ticker, avg_err_rate_grid_to_ticker, std_err_rate_grid_to_ticker) = i_grid_to_ticker_results
        grid_to_ticker_line_handles = self.__plotResults(figs, i_x, avg_wpm_grid_to_ticker, std_wpm_grid_to_ticker, 'r', 'Grid Latency Model', i_save, False )
        
        grid_best_line_handles = self.__plotResults(figs, i_x,  i_grid_best_wpm,  None,  'k--', 'Best Grid', i_save, False )
        grid_to_ticker_best_line_handles = self.__plotResults(figs, i_x,  i_grid_to_ticker_best_wpm, None, 'r--', 'Best Grid Latency Model', i_save, False )
    
        for (n, fig) in enumerate(figs):
            p.figure(fig.number)
            p.xlabel(i_xlabels[n])
            p.ylabel(ylabels[n])
            fig_name = "%strial" % self.output_dir
            print "*********************************************************"
            file_name = "%s%d_%d" % (fig_name, self.trial_num, n)
            print "Saving to " , file_name
            self.disp.saveFig(file_name )
        return figs
   
    def __plotResults(self, i_figs, i_x, i_avg_wpm, i_std_wpm, i_color, i_label, i_save, i_plot_circle=True ):
        
        line_handles = []
        #wpm
        p.figure(i_figs[0].number)
        line_handles.append(self.__plotErrorBar(i_x, i_avg_wpm, i_std_wpm, i_color, i_label, i_plot_circle))
        return line_handles
       
    def __plotErrorBar(sef, i_x, i_avg, i_std, i_color, i_label, i_plot_circle):
        line_handle, = p.plot(i_x, i_avg, i_color, linewidth=4, label=i_label)
        if i_plot_circle:
            p.plot(i_x, i_avg, i_color+'o', linewidth=2, markersize=7)
            line_handle, = p.plot(i_x, i_avg, i_color, linewidth=2, label=i_label)
        else:
            line_handle, = p.plot(i_x, i_avg, i_color, linewidth=2, label=i_label)
        if i_std is None:
            return line_handle
        for (n, std) in enumerate(i_std):
            p.plot([i_x[n], i_x[n]], [max(0,i_avg[n]-std), i_avg[n]+std], i_color,  linewidth=4,alpha=0.6)
        return line_handle 
    
    ######################################### Display
    
    def dispResult(self, i_grid_to_ticker_results, i_grid_results, i_index):
        print "Grid  : ",
        self.grid.displayResults(np.array(i_grid_results[i_index]), True, False) 
        print "Grid To Ticker  : ",
        self.grid_to_ticker.displayResults(np.array(i_grid_to_ticker_results[i_index]), True, False) 
        
        
def runApps():
    trials = [1,2,3,4]
    for (n, trial_num) in enumerate(trials):
        app = Simulations(i_trial=trial_num)
        app.compute()
        #app.plot()
        
        
if __name__ ==  "__main__":
    runApps()
     
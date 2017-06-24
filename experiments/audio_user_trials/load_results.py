
import sys, copy, os  
sys.path.append("../../")
sys.path.append("../")
from utils import Utils, DispSettings
import numpy as np
import pylab as p
from plot_utils import PlotUtils

"""Todo:
* Add expert user 
  - average on boxplots
  - can average in on other plots, depends on the data 
  - first plot expert and average other users
* Add simulation results on both plots
  - Point estimate on box plots
  - Full curve for saturation plots
* Add legends to saturation plots and make smaller"""

class AudioResultPlots():
    def __init__(self):
        #Init all util classes
        self.utils = Utils()
        self.disp =  DispSettings()
        #Load the results
        self.root_dir = "./results/graphs/"
        
class AudioResultSaturationPlots(AudioResultPlots):
    def __init__(self):
        AudioResultPlots.__init__(self)
        self.plt_utils = PlotUtils(self.root_dir, self.disp)
        self.ticker_results = self.plt_utils.loadBoxPlotResults(self.root_dir + "results_ticker.cPickle")
        self.grid_results = self.plt_utils.loadBoxPlotResults(self.root_dir + "results_grid.cPickle")
        self.tick_sess = 4 
        self.grid_sess = 3
        self.sub_sess = [1,2,3,4,5,6]
         
    ####################################### Load Functions

    def compute(self):
        """* plot the speed against the other two"""
        #Get ticker and Grid data
        print "*****************************************************************************"
        print "TICKER"
        print "*****************************************************************************"
        (cpc_ticker, char_err_ticker, speeds_ticker, speed_ids_ticker) = self.getSaturationData(self.ticker_results, 
            self.tick_sess, self.sub_sess)  
        print "*****************************************************************************"
        print "GRID"
        print "*****************************************************************************"
        (cpc_grid, char_err_grid, speeds_grid, speed_ids_ticker) = self.getSaturationData(self.grid_results, 
            self.grid_sess, self.sub_sess)  
        #Make the speed vs click per char plot
        file_out = self.root_dir + "speed_vs_cpc"   
        self.disp.newFigure()
        self.__saturationPlots( speeds_ticker, cpc_ticker, "k", "Speed", "Clicks Per Character", None)
        self.__saturationPlots( speeds_grid, cpc_grid, "r", "Speed", "Clicks Per Character", file_out)
        #Make the speed vs character error plot
        file_out = self.root_dir + "speed_vs_errors"   
        self.disp.newFigure()
        self.__saturationPlots( speeds_ticker, char_err_ticker, "k", "Speed", "Error rate (%)", None)
        self.__saturationPlots( speeds_grid, char_err_grid, "r", "Speed", "Error rate (%)", file_out)
        
    def __saturationPlots(self, i_x, i_y, i_color, i_x_label, i_y_label, i_file_name):
        p.plot(i_x, i_y, i_color + "x", linewidth=2)
        p.plot(i_x, i_y,  i_color, linewidth=2)
        p.ylabel(i_y_label)
        p.xlabel(i_x_label)
        if i_file_name is not None:
            self.disp.saveFig(i_file_name, i_save_eps=True)
        
    def getSaturationData(self, i_results, i_session, i_sub_sessions): 
        (cpc_mean, char_err_mean, speeds_mean) = ([],[],[])
        (cpc, err, speeds) = ([],[],[])
        speed_ids = []
        for sub_session in i_sub_sessions:
            ss = "%d" % sub_session
            if not i_results['users'].has_key(ss):
                continue
            s = "%d" % i_session
            if not i_results['users'][ss].has_key(s):
                continue 
            for (n, speed) in enumerate(i_results['speeds'][ss][s]):
                if len( np.nonzero(np.array(speed_ids) == speed)[0]) < 1:
                    cpc.append([])
                    err.append([])
                    speeds.append([])
                    speed_ids.append(speed)
                cpc[-1].append( i_results['cpc'][ss][s][n] )
                err[-1].append( i_results['char_err'][ss][s][n] )
                speeds[-1].append( i_results['wpm'][ss][s][n] )
        for n in range(0, len(speed_ids)):
            cpc_mean.append(np.mean(cpc[n]))
            char_err_mean.append(np.mean( err[n] ))
            speeds_mean.append(np.mean(np.array(speeds[n] ))) 
            print "speed setting = ", speed_ids[n], " wpm = ", speeds_mean[-1], " cpc = ", cpc_mean[-1], 
            print " char err = ", char_err_mean[-1]
        return (cpc_mean, char_err_mean, speeds_mean, speed_ids)

class AudioResultBoxPlots(AudioResultPlots):
    """Plotting the raw results for some of the experiments"""
    ############################################################### Init
    def __init__(self):
        AudioResultPlots.__init__(self)
      
    def initResults(self, i_results_label):
        o_results = {}
        if i_results_label == "ticker":
            r = dict(self.ticker_results)
        else:
            r = dict(self.grid_results)
        #Initialise the main keys
        for key in r.keys():
            if not o_results.has_key(key):
                o_results[key] = {}
                for sub_session in self.sub_sessions:
                    ss = "%d" % sub_session
                    o_results[key][ss] = {}
                    for session in self.sessions:
                        s = "%d" % session
                        o_results[key][ss][s] = []  
        return o_results
     
    def initBoxResultPlot(self, i_box_labels, i_ref_y_dist):
        #Student results and plot utils
        x_labels = ["%d" % sub_session for sub_session in self.sub_sessions]
        box_vals = ["%d" % session for session in self.sessions]
        self.plt_utils = PlotUtils(self.root_dir , self.disp, x_labels, box_vals, i_box_labels)
        self.plt_utils.ref_y_dist = i_ref_y_dist
        #Load the results
        self.ticker_results = self.plt_utils.loadBoxPlotResults(self.root_dir + "results_ticker.cPickle")
        self.grid_results = self.plt_utils.loadBoxPlotResults(self.root_dir + "results_grid.cPickle")
        #Initialise the display fonts etc
        #self.disp.setAudioUserTrialBoxPlotDisp()
        #Set some more plot utils variables
        self.plt_utils.draw_min_max = False
        self.plt_utils.draw_outliers = False
        self.plt_utils.draw_box = True
        
    ############################################################### Main
    
    def compute(self, i_ignore_ticker):
        """* Display the results for Grid & Ticker where new sentences were written from complete novice mode
             until complete audio mode and at a reasonable speed - plot the last two 15 minute sessions in this 
             speed (ommit learning)
           * Also display the noisy settings on the same plot."""  
        #Initialise plot parameters
        axis_tol = 0.01          #Blank space from top of the plot (labelling gray block)
        line_length = 0.25       #Median bar line length
        aspect_ratio = 16        #How much to stretch the picture to fit into paper
        y_inc = 0.5              #The y axis tick increments
        precision="%.1f"         #Precision of ytick labels
        ref_y_dist = None        #The height of the plot (can be maintained between multiple plots)
        plot_params = (axis_tol,line_length, aspect_ratio,y_inc,precision) 
        if i_ignore_ticker:
            self.disp.setAudioUserTrialBoxPlotDispGrid()
        else:
            self.disp.setAudioUserTrialBoxPlotDisp()
        
        ref_y_dist = self.plotNoviceResults(plot_params, ref_y_dist, i_ignore_ticker)
        ref_y_dist = self.plotNoiseResults(plot_params, ref_y_dist, i_ignore_ticker)
        
    def plotNoviceResults(self, i_plot_params, i_ref_y_dist, i_ignore_ticker=False):
        self.users = [3]
        if not i_ignore_ticker:
            self.old_sessions = {"ticker":[3], "grid":[2]} 
            self.old_sub_sessions = {"ticker":{3:[3,4]}, "grid":{2:[3,4]}}
            #Reload all the results
            self.sessions_ids = {"ticker": 1, "grid":2}
            self.sessions = [1,2]
        else:   
            self.old_sessions = {"grid":[2]} 
            self.old_sub_sessions = {"grid":{2:[3,4]}}
            #Reload all the results
            self.sessions_ids = {"grid":1}
            self.sessions = [1]
        self.sub_sessions = [1,2] 
        #Relabel the grid and ticker results
        file_suffix = "no_noise"
        if i_ignore_ticker:
            file_suffix += "_grid"
        self.plotResults(i_plot_params, file_suffix, i_ref_y_dist, i_plot_top_speed=False, i_ignore_ticker=i_ignore_ticker)
        return self.plt_utils.ref_y_dist

    def plotNoiseResults(self, i_plot_params, i_ref_y_dist, i_ignore_ticker=False):
        self.users = [3]
        if not i_ignore_ticker:
            self.old_sessions = {"ticker":[5], "grid":[4]} 
            self.old_sub_sessions = {"ticker":{3:[3,4],5:[1,2]}, "grid":{2:[3,4],4:[1,2]}}
                #Reload all the results
            self.sessions_ids = {"ticker": 1, "grid":2}
            self.sessions = [1,2]
        else:
            self.old_sessions = {"grid":[4]} 
            self.old_sub_sessions = {"grid":{2:[3,4],4:[1,2]}}
                #Reload all the results
            self.sessions_ids = {"grid":1}
            self.sessions = [1]
        self.sub_sessions = [1,2] 
        #Relabel the grid and ticker results
        file_suffix = "noisy"
        if i_ignore_ticker:
            file_suffix += "_grid"
        plot_top_speed = not i_ignore_ticker
        self.plotResults(i_plot_params, file_suffix,  i_ref_y_dist, i_plot_top_speed=plot_top_speed, i_ignore_ticker=i_ignore_ticker)
        return self.plt_utils.ref_y_dist
    
    ################################################# Generic
        
    def plotResults(self, i_plot_params, i_file_suffix, i_ref_y_dist, i_plot_top_speed=False, i_ignore_ticker=False): 
        #Initialise the plot according to plot params and session settings
        if not i_ignore_ticker:
            box_labels = ["T", "G"]
        else:
            box_labels = ["G"]
        self.initBoxResultPlot(box_labels, i_ref_y_dist)
        if not i_ignore_ticker:
            #Initialise the relabelled results
            r = self.initResults("ticker")
            #Relabel the results so that ticker & grid are grouped together in the corresponding sub sessions.
            r = self.relabelBoxResults(r, "ticker")
        else:
            r = self.initResults("grid")
        r = self.relabelBoxResults(r, "grid") 
        #Make the box plots
        label_poly = not i_ignore_ticker
        self.plt_utils.plotBoxResults(r, i_plot_params, i_file_suffix, i_plot_top_speed, i_label_poly=label_poly)
          
    def relabelBoxResults(self, i_results, i_results_label):
        sub_session_idx = 0
        o_results = dict(i_results)
        r = self.getResults(i_results_label)
        for session in self.old_sessions[i_results_label]:
                s = "%d" % session
                for sub_session in self.old_sub_sessions[i_results_label][session]:
                    ss = "%d" % sub_session
                    new_session = self.sessions_ids[i_results_label]
                    new_sub_session = self.sub_sessions[sub_session_idx]
                    print i_results_label, " old: sub_session = ", sub_session, " sess = ",  session,
                    print i_results_label, " new: sub_session = ", new_sub_session, " sess = ",  new_session 
                    new_ss = "%d" % new_sub_session
                    new_s  = "%d" % new_session
                    for key in r.keys():
                       o_results[key][new_ss][new_s].extend(list(r[key][ss][s])) 
                    sub_session_idx  += 1
        return o_results
                
    def getResults(self, i_results_label):
        if i_results_label == "ticker":
            r = dict(self.ticker_results)
        else:
            r = dict(self.grid_results)
        return r
        
if __name__ ==  "__main__":
    #app = AudioResultSaturationPlots() 
    #app.compute()
    app2 = AudioResultBoxPlots() 
    app2.compute(i_ignore_ticker=False)
    #app3 = AudioResultBoxPlots() 
    #app3.compute(i_ignore_ticker=True)   
    
    p.show()

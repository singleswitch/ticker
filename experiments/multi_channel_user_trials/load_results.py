import sys
sys.path.append("../../")
sys.path.append("../")
from utils import Utils, DispSettings
from plot_utils import PlotUtils
import numpy as np
import pylab as p

class TickerChannelResults():
    def __init__(self):
        self.utils = Utils()
        self.disp =  DispSettings()
        self.root_dir = "./results/"
        #Student results and plot utils
        self.x_labels = ["3","4","5"] #Channels
        self.box_vals = ["0.12", "0.10", "0.08", ] #Student overlaps
        self.box_labels = ["Slow", "Med", "Fast"] #Label corresponding to each overlap
        self.plt_utils = PlotUtils(self.root_dir, self.disp, self.x_labels, self.box_vals, self.box_labels)
        self.student_audio_file = self.root_dir + "student_audio_results.cPickle"
        #Case study results
        self.patient_audio_file = self.root_dir + "case_study_audio_results.cPickle"
        self.plt_utils.patient_box_vals = {"0.12":"0.08"} #Mapping from patient speed to student speed
        #Initialise the display fonts etc
        self.disp.setBoxPlotDisp()
        #Diagnostic
        self.debug=True
         
    ################################################ Main
    
    def plotResults(self):
        #Initialise plot parameters
        axis_tol = 0.01          #Blank space from top of the plot
        line_length = 0.25       #Median bar line length
        aspect_ratio = 8         #How much to stretch the picture to fit into paper 
        precision="%.1f"         #Precision of ytick labels
        y_inc = 0.5              #The y axis tick increments
        plot_params = (axis_tol, line_length, aspect_ratio,y_inc,precision) 
        #Load the patient data
        (wpm_patient, error_rate_patient, clicks_patient) = self.plt_utils.loadBoxPlotResults(self.patient_audio_file)
        patient_data = (wpm_patient, error_rate_patient, clicks_patient)
        #Load the student data
        r={}
        (r['wpm'],r['char_err'],r['cpc']) =  self.plt_utils.loadBoxPlotResults(self.student_audio_file)
        #Make the final plots
        self.plt_utils.plotBoxResults(r, plot_params,"", False, patient_data)
        #Debug
        self.plotResultsDebug(r['wpm'])
            
        p.show()
        
    ################################# Diagnostic
    
    def plotResultsDebug(self, i_results):
        """Make a matplotlib box and whisker plot to make sure it's the same as ours""" 
        if not self.debug:
            return
        p.figure() 
        
        for (m,speed) in enumerate(self.box_vals):
            data = []
            for (n,channel) in enumerate(self.x_labels):
                data.append(np.array(i_results[channel][speed]))
                d = i_results[channel][speed]
                print "Speed = ", speed, " channel = ", channel, " data shape = ",
                print d.shape, " median = ", np.median(d), " mean = ", np.mean(d)
                 
            p.subplot(3,1,m+1); p.grid('on')
            p.boxplot(data)    
            
            for k in range(0,len(data)):
                print "k= ", k, " data shape = ",  data[k].shape
            
            p.title(" speed= " + str(speed)) 
            

if __name__ ==  "__main__":
    results = TickerChannelResults()
    results.plotResults()
     

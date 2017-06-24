
import numpy as np
import pylab as p
from utils import Utils, DispSettings

class PlotUtils():
    def __init__(self, i_root_dir="./", i_disp_settings=None, i_x_labels=None, i_box_vals=None, i_box_labels=None):
        self.disp = i_disp_settings
        self.root_dir = i_root_dir
        self.x_labels = i_x_labels #Settings in each session
        self.box_vals = i_box_vals #Whole session value
        self.box_labels = i_box_labels #Session label
        self.ref_y_dist = None  #Tweak the aspect ratio of first picture - to be used in rest
        self.draw_min_max = True
        self.draw_outliers = True
        self.draw_box = True
        self.utils = Utils()
    
    def initPlot(self, i_x_label, i_y_label, i_is_patient):
        self.disp.newFigure()
        p.ylabel(i_y_label) 
        p.xlabel(i_x_label)
        x_ticks= range(1,len(self.x_labels)*len(self.box_vals)+1)
        x_ticks_labels = []
        for x_labels in self.x_labels:
            x_ticks_labels.extend(self.x_labels)
        if i_is_patient:
            x_ticks.append(x_ticks[-1] + 1)
            x_ticks_labels.append("5")
        ax = p.gca()
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_ticks_labels)
        (xmin, xmax) = (0.5, len(x_ticks) + 0.5) 
        #Total y limits for all data on the plot
        self.total_min = np.inf 
        self.total_max = -np.inf
        return (xmin, xmax)
    
    ############################################################ Main
    
    def plotResults(self, i_dict, i_x_label, i_title, i_plot_params, i_save_file=None, i_color="k", i_patient_dict=None, i_label_poly=True):
        is_patient = i_patient_dict is not None
        (axis_tol, line_length, aspect_ratio, y_inc, precision) = i_plot_params
        (xmin, xmax) = self.initPlot(i_x_label=i_x_label,  i_y_label=i_title, i_is_patient=is_patient)
        #Draw the student result lines
        (ymin, ymax) = self.drawResultLines(i_dict, line_length, i_color=i_color)
        if is_patient:
            #Draw the patient result lines
            (ymin_patient, ymax_patient) = self.drawResultLines(i_patient_dict, line_length, i_color="r", i_is_student=False)
            #Determine axis limites
            ymin = min(ymin, ymin_patient)
            ymax = max(ymax, ymax_patient) 
        #Complete the plot
        patient_data = i_patient_dict is not None
        #Compute and set the yticks
        (new_ymin, new_ymax) = (self.getClosestVal(ymin, y_inc), self.getClosestVal(ymax, y_inc))
        ymin = min(ymin, new_ymin)
        ymax = max(ymax, new_ymax)
        self.setYTicks(new_ymin, new_ymax, y_inc, precision)
        #Draw the polygons
        (ymin,ymax) = self.fillSpeedPolygons(ymin, ymax, axis_tol, line_length, patient_data, i_label_poly=i_label_poly)
        #Set the axis
        p.axis([xmin, xmax, ymin, ymax]) 
        #Set the aspect ratio
        self.setAspectRatio(ymin, ymax, aspect_ratio)
        #Save the fig
        if i_save_file is not None:
            self.disp.saveFig(i_save_file)
        return (ymin, ymax)        
            
    def drawResultLines(self, i_dict, i_line_length, i_color="k", i_is_student=True):
        """Draw the lines associated with the results (median and std around the median)"""
        ymax = 0.0 
        ymin = np.inf
        for x_key in i_dict.keys():
            x_key_int = int(x_key)
            x_key_idx = np.nonzero(np.array(self.x_labels) == x_key)[0]
            for box_val in i_dict[x_key].keys():
                #Find the speed in "student" metric system
                student_box_val = box_val
                if not i_is_student: 
                    student_box_val = self.patient_box_vals[box_val]
                box_idx = np.nonzero(np.array(self.box_vals) == student_box_val)[0]
                if len(box_idx) < 1:
                    continue
                x =  len(self.x_labels)*box_idx + x_key_idx + 1 
                if not i_is_student:
                    x += 1
                r = self.getResultsStats(i_dict[x_key][box_val])
                (min_i_dict, max_i_dict, q1, q2, q3, outliers_min, outliers_max, total_min, total_max) = r 
                (ymax, ymin) = (max(ymax, total_max), min(ymin, total_min))
                #Plot the outliers
                if self.draw_outliers:
                    p.plot(x*np.ones(len(outliers_min)),outliers_min, 'ko', linewidth=1, markerfacecolor="w", alpha=0.7)
                    p.plot(x*np.ones(len(outliers_max)),outliers_max, 'ko', linewidth=1, markerfacecolor="w", alpha=0.7)
                #Plot the box - 25th, 50th and 75th percentiles
                (long_line,short_line)  = (i_line_length , 0.5*i_line_length)
                (x0, x1) = (x-long_line, x+long_line)
                p.plot([x0,x1],[q2, q2], i_color,linewidth=3)
                if self.draw_box:
                    p.plot([x0,x0,x1,x1,x0],[q1,q3,q3,q1,q1],i_color,linewidth=1)   
                #Plot the min and max
                if self.draw_min_max:
                    p.plot([x,x],[min_i_dict,max_i_dict],i_color,linewidth=1)
                    (x0, x1) = (x-short_line, x+short_line)
                    p.plot([x0, x1],[max_i_dict, max_i_dict], i_color,linewidth=2)
                    p.plot([x0, x1],[min_i_dict, min_i_dict], i_color,linewidth=2)
        return (ymin, ymax)  
    
    def fillSpeedPolygons(self, i_ymin, i_ymax, i_axis_tol, i_line_length, i_patient_data, i_label_poly=True):
        ydist = i_ymax - i_ymin
        ytol = ydist*i_axis_tol
        ymax = i_ymax + ytol
        ymin = i_ymin - ytol
        #Fill rectangular regions within a plot that are associated with the same experiment (speed in this case)
        for n in range(0, len(self.box_vals)):
            idx = n*len(self.x_labels) + 1.0
            xs = idx-i_line_length
            delta_x = len(self.x_labels) + 2.0*i_line_length -1.0
            #Add one more for the fast channel (patient data)if multi-channel experiment is analysed
            if i_patient_data:
                if n == (len(self.box_vals)-1):
                    delta_x += 1
            xe = xs+delta_x
            if i_label_poly:
                p.text(xs, ymax, self.box_labels[n]) 
            p.fill( [xs, xs,xe,xe,xs],[ymin,ymax,ymax,ymin,ymin],color='0.75',alpha=0.2, edgecolor='k') 
        return (ymin,ymax)
    
    ############################################################ Other generic
    
    def plot2DCircle(self, i_m0, i_m1, i_r, i_color="k", i_plot_r=True, i_marker='o'):
        """Plot means of a circle, optional radius (at a few scales of it)"""
        p.plot([i_m0], [i_m1], i_color+i_marker)
        theta = np.linspace(0, 2.0*np.pi, 100)
        if i_plot_r:
            #p.plot( i_m0 + i_r*np.cos(theta), i_m1 + i_r*np.sin(theta), i_color)
            p.plot( i_m0 + 2.0*i_r*np.cos(theta), i_m1 + 2.0*i_r*np.sin(theta), i_color)
    
    
    ############################################################# Set
    
    def adjustYTicks(self, i_ymin=-np.inf):
        (yticks, yticks_labels) = p.yticks() 
        (yticks, yticks_labels)  = self.adjustTicks(yticks, i_ymin)
        p.yticks(yticks,  yticks_labels)
            
    def adjustTicks(self, i_ticks, i_min=-np.inf):
        idx = np.nonzero(i_ticks >= i_min)[0]
        o_ticks = i_ticks[idx]
        o_labels = np.array(["$%.1f$" % tick for tick in o_ticks])
        (o_labels , idx) = np.unique(o_labels, return_index=True)
        return (o_ticks[idx], o_labels)
        
    def setYTicks(self,i_ymin, i_ymax, i_y_inc, i_precision="%.1f"):
        y_ticks = np.arange(i_ymin, i_ymax+i_y_inc, i_y_inc)
        y_tick_labels = []
        for tick in y_ticks:
            #Make sure there are no duplicates due to precision rounding
            if i_precision == "%d":
                tick_val = int(0.5 + tick)
            else:
                tick_val = tick
            tick_label = i_precision % tick_val
            y_tick_labels.append(tick_label)
        y_tick_labels = np.array(y_tick_labels)
        ax = p.gca()
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_tick_labels)  
        labels = ax.get_yticklabels()
        for label in labels:
            label.set_rotation(90) 
    
    def setColorBar(self,i_max_val, i_title, i_min=-np.inf): 
        if  i_title is None:
            return
        ticks= np.linspace(0, i_max_val, 4 ) 
        (ticks, labels) = self.adjustTicks(ticks, i_min)
        ax = p.gca()
        cbar = p.colorbar(format='%.1f', ticks=ticks)
        cbar.set_label(i_title)
        cbar.ax.set_yticklabels(labels)
        return cbar

    def setAspectRatio(self, i_ymin, i_ymax, i_aspect_ratio):
        ax = p.gca()
        ydist = i_ymax - i_ymin
        if self.ref_y_dist is None:
            self.ref_y_dist = i_ymax - i_ymin
            aspect_ratio = i_aspect_ratio
        else: 
            aspect_ratio = i_aspect_ratio*self.ref_y_dist / ydist
        ax.set_aspect(aspect_ratio)
     
    ############################################################# Get
    
    def getClosestVal(self, i_val, i_inc):
        val = np.round(i_val)
        vals = np.arange(val-2.0*i_inc, val+2.0*i_inc, i_inc)
        dist = np.abs(vals - i_val)
        o_val = vals[np.argmin(dist)]
        return o_val
    
    def getResultsStats(self, i_list):
        vals = np.array(i_list).flatten()
        if len(vals) < 1:
            (min_val, max_val, q1, q2, q3, outliers_min, outliers_max) = (0,0,0,0,0,0,0) 
            return (min_val, max_val, q1, q2, q3, outliers_min, outliers_max, self.total_min, self.total_max)
        q1 = np.percentile(vals, 25)
        q2 = np.median(vals)
        q3 = np.percentile(vals, 75)
        min_val = np.percentile(vals, 5)
        max_val = np.percentile(vals, 95)
        outliers_min = vals[np.nonzero(vals < min_val)]
        outliers_max = vals[np.nonzero(vals > max_val)]   
        if self.draw_outliers:
            total_max = np.max(vals)
            total_min = np.min(vals)
        elif self.draw_min_max:
            total_max = max_val
            total_min = min_val
        elif self.draw_box:
            total_max = q3
            total_min = q1
        else:
            total_max = q2
            total_min = q2
        if np.abs(total_max-total_min) < 1E-1:
            total_max += 0.05
            total_min -= 0.05
        self.total_min = min(total_min, self.total_min)
        self.total_max = max(total_max, self.total_max)
        return (min_val, max_val, q1, q2, q3, outliers_min, outliers_max, self.total_min, self.total_max)
    
    ################################################# Results: Box plots (typical)
    
    #Application specific plots 
    def plotBoxResults(self, i_results, i_plot_params, i_file_suffix, i_plot_top_speed=False, i_patient_data=None, i_label_poly=True):
        if i_patient_data is not None:
            (wpm_patient, error_rate_patient, clicks_patient) = i_patient_data
        else:
            (wpm_patient, error_rate_patient, clicks_patient) = (None, None, None)
        #Box plots used in multi-channel experiment and audio experiment 
        color="k"
        file_suffix = i_file_suffix
        if not i_file_suffix == "":
            file_suffix = "_" + i_file_suffix 
        self.plotWpm(i_results, i_plot_params, file_suffix, i_plot_top_speed, color, wpm_patient, i_label_poly)
        self.plotClicksPerChar(i_results, i_plot_params, file_suffix, color, clicks_patient, i_label_poly)
        self.plotErrorRate(i_results, i_plot_params, file_suffix, color, error_rate_patient, i_label_poly)

    def plotWpm(self,i_results, i_plot_params, i_file_suffix, i_plot_top_speed=False, i_color="k", i_patient_data=None, i_label_poly=True):
        save_file = self.root_dir + "wpm" + i_file_suffix 
        self.plotResults(i_results['wpm'],"Sesssion", "Entry rate (wpm)", i_plot_params, save_file, i_color, i_patient_data, i_label_poly)
        self.plotTopSpeedTicker(i_plot_top_speed, i_results, save_file, i_plot_params)
    
    def plotClicksPerChar(self, i_results, i_plot_params, i_file_suffix,i_color="k", i_patient_data=None, i_label_poly=True):
        save_file = self.root_dir + "clicks" + i_file_suffix
        self.plotResults(i_results['cpc'],"Session", "Clicks Per Character", i_plot_params, save_file, i_color, i_patient_data, i_label_poly)   
        
    def plotErrorRate(self,i_results, i_plot_params, i_file_suffix, i_color="k", i_patient_data=None, i_label_poly=True):
        #Get the plot params if they need to be changed
        (axis_tol,line_length, aspect_ratio,y_inc,precision) = i_plot_params
        save_file = self.root_dir + "error_rate" + i_file_suffix  
        y_inc = 20
        precision = "%d"
        plot_params = (axis_tol,line_length, aspect_ratio,y_inc,precision)         
        self.plotResults(i_results['char_err'],"Session", "Error rate (%)", plot_params, save_file,i_color, i_patient_data, i_label_poly)
    
    def plotTopSpeedTicker(self, i_plot_top_speed, i_results, i_save_file, i_plot_params):
        """Only for audio experiment if the speed setting was recorded"""
        if not i_plot_top_speed:
            return
        #Plot the top speed that the user can obtain if the delay at end was removed
        top_speed = max(i_results['speeds']['2']['1'])
        (axis_tol,line_length, aspect_ratio,y_inc,precision)= i_plot_params
        print "Top speed = ", top_speed
        p.plot([1-line_length,2+line_length],[top_speed,top_speed],'-', linewidth=2)
        self.disp.saveFig(i_save_file)
        
    def loadBoxPlotResults(self, i_file_name):
        results  =  self.utils.loadPickle(i_file_name)
        if isinstance(results, tuple):
            results = list(results)
            o_results = list(results)
            for (n,key) in enumerate(results):
                results[n] = self.removeNone(key)
            results = tuple(results)
        else:
            o_results = {}
            for key in results.keys():
                o_results[key] = self.removeNone(results[key])
        return results
    
    def removeNone(self, i_dict):
        o_dict = dict(i_dict)
        for key_1 in i_dict.keys():
            for key_2 in i_dict[key_1].keys():
                o_dict[key_1][key_2] = np.array([val for val in i_dict[key_1][key_2] if val is not None])
        return o_dict
  
    #################################################### Simulation plots
    
    def plotErrorBar(self, avg, y, std=None, color='b', scale_center = 2.0, scale_width=25.0, width=2, i_min_x=None, i_max_x=None):
        max_y = np.max(y)
        p.plot([ avg,  avg],[0, max_y], color+'-', linewidth=width)   
        if std is None:
            return
        err_bar_width =  max_y /scale_width
        err_bar_centre = max_y/scale_center
        if i_min_x is not None:
            min_x = max(avg-std, i_min_x)
            print "min_x = ", min_x
        else:
            min_x = avg-std
        if i_max_x is not None:
            max_x = min(i_max_x, avg+std)
        else:
            max_x = avg+std
        p.plot([min_x, max_x],[err_bar_centre,  err_bar_centre], color+'--', alpha=0.3, linewidth=width)  
        p.plot([min_x, min_x],[err_bar_centre - err_bar_width, err_bar_centre + err_bar_width ], color+'-', alpha=0.3, linewidth=width)  
        p.plot([max_x, max_x],[err_bar_centre - err_bar_width, err_bar_centre + err_bar_width ], color+'-', alpha=0.3, linewidth=width)  
   

    def plotDiscrete(self, x, y):
        """ * Plot data as discrete distribution
           * Dots with lines going from y=0 to the pmf value."""
        for n in range(0, len(x)):
            if np.abs(y[n]) > 1E-20:
                p.plot( [x[n],x[n]], [0,y[n]],'k')
                p.plot( [x[n]], [y[n]],'ko')
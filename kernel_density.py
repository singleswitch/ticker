import sys, time
import pylab as p
from utils import Utils, PylabSettings
import numpy as np
import scipy.stats.distributions as sd
 
class NonParametricPdf():
    ############################################# Initialisation functions    
    def __init__(self, i_learning_rate=1.0, i_max_samples=1000, i_disp=False):
        self.utils = Utils()
        self.min_std = 0.03                  #Minimum kernel density smoothing factor  
        self.max_bins = 50                   #The number of display bins for plotting purposes
        self.min_N = 1.0                     #Minimum value for the number of samples 
        self.nstd = 4.0                      #The number of std considered in initial samples, and drawing
        self.min_weight = 1E-10              #Weights smaller than this will be pruned
        self.min_val = 1E-12                 #Clip min value returned by likelihood to this value            
        self.learning_rate = i_learning_rate #Online learning parameter
        self.max_samples = i_max_samples     #Maximum number of samples to use during likelihood computation
        self.disp = i_disp
        self.clear()  
       
    def reset(self, i_delay, i_gauss_std):
        self.clear()
        xn = np.sort(np.random.normal( i_delay, i_gauss_std, self.max_samples ).flatten())
        xn = np.clip(xn, i_delay-self.nstd*i_gauss_std, i_delay+self.nstd*i_gauss_std) 
        gauss_std = self.setStdFromData(xn)
        self.saveDataPoints(xn, i_weights=None) 
        
    def clear(self):
        self.xn = np.array([]) #Data points used for kernel density estimation
        self.xn_std = np.array([]) #All the previous stored kernel deviations
        self.weights =np.array([]) #The weights of each Gaussian in kernel density estimation 
        self.min_x = np.inf #Minimum x used for plot
        self.max_x = -np.inf #Maximum x used for plot
        self.bin_width = None #Bin width used in plots 
            
    ############################################# Main functions   
    
    def saveDataPoints(self, i_data_points, i_weights=None ):
        self.savePlotParameters(i_data_points)
        print "KERNEL DENSITY LEARNING RATE = ", self.learning_rate
        #Store the normalised observations (kernel std can also be updated as we go along)
        if len(self.xn) < 1:
            self.xn = np.array(i_data_points.flatten())
            self.xn_std = self.kernel_std*np.ones(len(self.xn))
            self.weights = np.ones(len(self.xn))
            self.weights *= (1.0 / (np.sum(self.weights) * self.kernel_std)) 
        else:
            self.xn = np.hstack( (self.xn.flatten(), i_data_points.flatten()) )
            self.xn_std = np.hstack((self.xn_std.flatten(), self.kernel_std*np.ones(len(i_data_points.flatten()))))
            #Store the weights
            old_weights = (1.0-self.learning_rate)*(self.weights.flatten())
            if i_weights is None:
                new_weights = np.ones(len(i_data_points.flatten()))
            else:
                new_weights = np.array(i_weights.flatten())
            new_weights *= (self.learning_rate / (np.sum(new_weights) * self.kernel_std))
            self.weights = np.hstack( ( old_weights, new_weights) )
        self.pruneWeights()
        #Save in the correct format
        self.xn = np.atleast_2d(self.xn).transpose()
        self.weights = np.atleast_2d(self.weights).transpose()
        self.xn_std = np.atleast_2d(self.xn_std).transpose()
    
    def likelihood(self, i_x, i_params=None, i_log=False ):
        y = p.normpdf( (self.xn - i_x.flatten()) / self.xn_std , 0.0, 1.0) 
        y *= self.weights
        y = np.sum(y, axis=0)
        idx = np.nonzero( np.abs(y) < self.min_val)[0]
        y[idx] = self.min_val
        y = y.reshape(i_x.shape)
        if i_log:
            return np.log(y)
        return y
        
    ################################### Private Save Data Points
    
    def savePlotParameters(self, i_data_points):
        data_min =  np.min(i_data_points.flatten()) - self.nstd*self.kernel_std 
        data_max =  np.max(i_data_points.flatten()) + self.nstd*self.kernel_std
        self.min_x = min(self.min_x, data_min) 
        self.max_x = max(self.max_x, data_max)
        self.bin_width = (self.max_x - self.min_x) / float(self.max_bins - 1.0)
             
    def pruneWeights(self):
        """* Remove very small weights first
           * Then remove more samples, if maximum number of samples is exceeded"""
        #Prune weights that are zero
        idx = np.nonzero(np.abs(self.weights) > self.min_weight)[0]
        self.weights = self.weights[idx]
        self.xn =  self.xn[idx]
        self.xn_std = self.xn_std[idx]
        #Truncate the weights and data points if necessary
        if len(self.xn) > self.max_samples:
            self.xn = self.xn[-self.max_samples:]
            self.weights = self.weights[-self.max_samples:]
            self.xn_std = self.xn_std[-self.max_samples:]
        
    ###########################################################Set Functions
    
    def setStdFromData(self, i_data):
        N = float(len(i_data.flatten()))     
        gauss_std = np.std(i_data) * N / (N-1.0) 
        self.setStd(gauss_std, N)
        return gauss_std
     
    def setStd(self, i_gauss_std, i_N ):
        #Set the kernel bandwith parameter from gaussian approximation of the data
        #Include the number of points that were used to make gaussian std approximation
        self.gauss_std = i_gauss_std
        self.kernel_std =  (i_N**(-0.2))* 1.06*self.gauss_std 
        if self.kernel_std < self.min_std:
            self.kernel_std = self.min_std
              
    def setKernelBandWidth(self, i_kernel_std):
        #Only new data points will be affected by this change
        self.kernel_std = i_kernel_std
   
    #####################################Draw 
    
    def draw(self, i_color="k", i_histogram=True): 
        (x, w, y) = self.getHistogramRects() 
        p.plot(x + 0.5*self.bin_width, y, i_color, linewidth=2)
        if i_histogram: 
            for n in range(0, len(x)):
                p.bar(x[n], y[n], w[n], alpha=0.2)
    
    def getXYPairs(self):
        x = np.linspace( self.min_x, self.max_x, self.max_bins )
        y = self.likelihood( x )
        return (x, y)
    
    def getHistogramRects(self):
        (x, y) = self.getXYPairs()
        distr_sum = np.sum(y)*self.bin_width
        if self.disp:
            print "Kernel density distribution sum = ", distr_sum
        w = self.bin_width*np.ones(len(x))
        x -= 0.5*self.bin_width
        return (x, w, y)
        
                
    ###################################### Examples
    
class KernelDensityExamples():
    ################################################# Init
    
    def __init__(self):
        self.plots = PylabSettings()
        self.nstd = 4.0 #Number std to plot from gauss 
        
    def loadMixtureParams(self):
        iterations = 16                                      #Number of online iterations
        n_test = 20                                          #Number of test samples per iteration
        learning_rate = 0.3
        n_plot = 100                                         #Number of plot points for ground truth values
        (t_means, t_sigmas) = ([0.4, 1.0], [0.2,0.05])       #Test parameters to learn
        (g_mean, g_sigma) = (0.1, 0.1)                       #Initial values
        mixture_weights = np.ones(len(t_means)) / len(t_means) #The mixture weights 
        return (iterations, n_test, learning_rate, n_plot, t_means, t_sigmas, g_mean, g_sigma, mixture_weights)
        
    def loadGaussData(self):
        iterations = 10                                      #Number of online iterations
        n_test = 10                                          #Number of test samples per iteration
        learning_rate = 0.3
        n_plot = 100                                         #Number of plot points for ground truth values
        (t_means, t_sigmas) = ([0.4],[0.2])                  #Test parameters to learn
        (g_mean, g_sigma) = (0.1, 0.1)                       #Initial values
        mixture_weights = np.ones(len(t_means)) / len(t_means) #The mixture weights 
        return (iterations, n_test, learning_rate, n_plot, t_means, t_sigmas, g_mean, g_sigma, mixture_weights)
      
    ################################################ Main
    
    def examples(self):
        self.batchGaussExample()
        #Online Gaussian example
        params = self.loadGaussData()
        ex.onlineMixtureExample(params)
        #Gaussian mixture model example - learning a mixture via online training
        params = self.loadMixtureParams()
        ex.onlineMixtureExample(params)
        p.show()
      
    ################################################ Separate Examples
    
    def batchGaussExample(self):
        """* Use Gaussian Kernel no posterior weights
           * Test batch learning (see if Gaussian distribution can be learned with n_samples)"""
        self.plots.newFigure()
        click_pdf =  NonParametricPdf(i_learning_rate=1.0, i_max_samples=1000, i_disp=True)
        (sigma , g_mean ) = (0.05 ,  0.2)                #Ground-truth parameters
        nsamples = 300                                   #The number of independent samples to plot the posterior from 
        npoints_eval = 100                               #The number of points to evaluate
        samples = np.clip( np.random.normal( 0, sigma,  nsamples ), -click_pdf.nstd*sigma, click_pdf.nstd*sigma) + g_mean 
        click_pdf.reset( g_mean, sigma )
        x_eval = np.linspace( -10*sigma, 10*sigma,  npoints_eval) + g_mean 
        y_eval = sd.norm.pdf(x_eval, loc=g_mean, scale=sigma)
        p.plot(x_eval, y_eval, 'r' )
        click_pdf.draw()
        
    def onlineMixtureExample(self, i_params):
        #Load the params
        (iterations,n_test,learning_rate,n_plot,t_means,t_sigmas,g_mean,g_sigma,mix_weights) = i_params         
        #Initialise the click distr
        click_pdf =  NonParametricPdf(i_learning_rate=learning_rate, i_max_samples=1000, i_disp=True)
        click_pdf.reset(i_delay=g_mean, i_gauss_std=g_sigma)
        #The initial plot values
        (g_x, g_y) = self.getGaussData(g_mean, g_sigma)
        #Test plot values
        (t_x, t_y) = self.mixtureData(t_means, t_sigmas, mix_weights)
        #Update the kernel density with n_test points at a time
        (n_rows, n_cols) = self.prepareOnlineGrid(iterations)
        for n in range(0, iterations+1):
            p.subplot(n_rows, n_cols, n+1); p.axis('off')
            if n > 0:
                test_points = self.sampleMixture(mix_weights, t_means, t_sigmas, n_test)
                click_pdf.setStdFromData(test_points)
                weights = np.ones(len(test_points.flatten()))
                t=time.time() 
                click_pdf.saveDataPoints(test_points, i_weights=weights )
                print "update time = ", 1000.0*(time.time() - t), " ms"
                p.plot(test_points, np.zeros(len(test_points)), 'kx')
            click_pdf.draw()
            p.plot(g_x, g_y, 'b' )
            p.plot(t_x, t_y, 'r' )
      
    ############################################# Private 
    
    def mixtureData(self, i_means, i_sigmas, i_weights):
        n_plot = 100
        means = np.array(i_means)
        sigmas = np.array(i_sigmas)
        min_x = np.min(means - self.nstd*sigmas)
        max_x = np.max(means + self.nstd*sigmas)
        x  = np.linspace(min_x, max_x, n_plot) 
        y = np.zeros(x.shape)
        for n in range(0, len(i_means)):
            y += (i_weights[n]*sd.norm.pdf(x, loc=i_means[n], scale=i_sigmas[n]))  
        return (x,y)
    
    def sampleMixture(self, i_cluster_weights, i_means, i_sigmas, i_n_samples):
        data = []
        w = np.array(i_cluster_weights).flatten() #Cluster weights
        for n in range(0, i_n_samples):
            cluster = np.nonzero( np.abs(np.random.multinomial(1,w)) > 0)[0][0]
            data.append(np.random.normal(i_means[cluster], i_sigmas[cluster], 1 ))
        return np.array(data).flatten()
    
    def getGaussData(self, i_mean, i_sigma):
        n_plot = 100 #Number of point used for plot
        x  = np.linspace( -10*i_sigma, 10*i_sigma, n_plot) + i_mean 
        y = sd.norm.pdf(x, loc=i_mean, scale=i_sigma)  
        return (x, y)  
        
    def prepareOnlineGrid(self, i_iterations):
        #Setup the grid plot
        n_rows = int(np.ceil(np.sqrt(i_iterations+1)))
        n_cols = 1
        while (n_rows*n_cols) < (i_iterations+1):
            n_cols +=1
        self.plots.newFigure()
        return (n_rows, n_cols)
         
if __name__ ==  "__main__":
    ex =  KernelDensityExamples()
    ex.examples()
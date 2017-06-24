

import pylab as np
import scipy.special

class ClickDistribution():
    
    ############################################# Initialisation functions    

    def __init__( self ):
        params = dict({ 'learning_rate' :  0.8,    #Percentage of previous pdf  when updating new pdf
                        'update' : True,           #Update click distribution - use non parametric version
                        'click_std' : 0.1,         #Initial standard deviation of Gaussian click distribution
                        'click_delay': 0.1,        #Measurement delay (the number of time units)
                        'max_bins' : 100} )        #Maximum number of bins to use    
        self.__min_std = 0.03
        self.setParams(params)
        self.initNonParametric()
    
        
    def initNonParametric(self):
        #Initialise non-parametric distribution
        #Gaussian with zero mean
        g_mean = self.__params['click_delay']
        sigma = self.__params['click_std']
        n_std = 6
        self.__possible_click_times  = np.linspace( g_mean - n_std*sigma, g_mean + n_std*sigma, 
                                            self.__params['max_bins'])
        n_lambda = self.__params['max_bins']
        self.__computeClickKernelStd(self.__params['click_std'], n_lambda)
        gauss_pdf = np.normpdf(self.__possible_click_times, g_mean, sigma)
        width = 0.5*np.mean( self.__possible_click_times[1:] - self.__possible_click_times[0:-1])
             
        self.__non_parametric = np.array([self.__possible_click_times-width, self.__possible_click_times+width, gauss_pdf]).transpose()
        start_boundary = np.array([-np.inf,  self.__non_parametric[0,0], 0.0])
        end_boundary = np.array([self.__non_parametric[-1,1], np.inf, 0.0])
        self.__non_parametric = np.vstack([start_boundary, self.__non_parametric, end_boundary])   
        self.__bin_width = width
        self.__clicks = []
      
        
    ############################################# Main functions    
    
    def logLikelihood(self, i_observations):
        #return np.log( np.normpdf(i_observations, self.__params['click_delay'], self.__params['click_std']))
        return np.log(self.__likelihood(i_observations)).flatten()
    
    def update(self):
        if  not self.__params[ 'update']: 
            print "No update"
            return
        #The observations
        if len(self.__clicks) >= 2:
            xn =  np.array(self.__clicks).flatten()  
            #The possible click times
            x_min = min(np.array([min(self.__non_parametric[1:,0]), min(xn)]))
            x_max = max(np.array([max(self.__non_parametric[0:-1,1]), max(xn)]))
            x = np.arange( x_min, x_max+2.0*self.__bin_width, 2.0*self.__bin_width).flatten()
            previous_pdf = self.__likelihood(x)
            current_pdf  = self.__computeNonParametricPdf(xn, x)
            w = self.__params['learning_rate'] 
            updated_pdf = w * previous_pdf + (1.0 - w) * current_pdf
            self.__non_parametric = np.array([x-self.__bin_width, x+self.__bin_width, updated_pdf]).transpose()
            start_boundary = np.array([-np.inf,  self.__non_parametric[0,0], 0.0])
            end_boundary = np.array([self.__non_parametric[-1,1], np.inf, 0.0])
            self.__non_parametric = np.vstack([start_boundary, self.__non_parametric, end_boundary])
        self.__clicks = []
    
    def storeClick(self,i_click):
        self.__clicks.append(i_click)
        
    ###################################################### #####Set Functions
    
    def setParams(self, i_params):
        self.__params = dict(i_params)
        
    def setClickDev(self, i_std):
        self.__params['click_std'] = i_std
        self.initNonParametric()
       
    def setUpdate(self, i_update):
        self.__params[ 'update'] = i_update #True or False
        
    def setClickDelay(self, i_delay):
        self.__params['click_delay'] = i_delay
        self.initNonParametric()
        
    ###########################################################Get Functions
        
    def getParams(self):
        return self.__params
    
    def getClickDelay(self):
        return self.__params['click_delay']
    
    def getNonParametricDistribution(self):
        return self.__non_parametric
        
    def getHistogramRects(self):
        pdf = self.__non_parametric
        #print "pdf = ", pdf
        top_left_x = pdf[1:-1,0].flatten()
        top_left_y = pdf[1:-1,2].flatten()
        width = (pdf[1:-1,1] - pdf[1:-1,0]).flatten()
        height = np.absolute(top_left_y)
        idx = np.nonzero(top_left_y >1E-10)[0]
        return (top_left_x[idx], width[idx], height[idx]) 
    
    ###########################################################Private Function
    
    def __computeNonParametricPdf(self, xn, x):
        #Compute the new non parametric pdf from the data at the values x
        xn = xn.reshape([len(xn),1])
        n_lambda = len(xn)
        std_samples =  np.sqrt(np.sum((xn - xn.mean())**2)/ (np.float64(len(xn)-1.0)))
        self.__computeClickKernelStd(std_samples, n_lambda)
        #Compute the pdf heights
        dist = (x - xn) 
        g_var = self.__gauss_kernel_std**2
        c = len(xn) *  self.__gauss_kernel_std * np.sqrt(2*np.pi)
        c = 1. / c 
        dist = np.exp(-dist*dist / (2*g_var))
        pdf = c*np.sum(dist, axis=0)
        return pdf
       
    def __computeClickKernelStd(self, i_data_std, i_n_lambda):
        self.__gauss_kernel_std = ( (i_n_lambda**(-0.2)) * 1.06*i_data_std)
        if self.__gauss_kernel_std < self.__min_std:
            self.__gauss_kernel_std = self.__min_std 
         
    def __likelihood(self, i_x):
        """Retrieve the non parameteric pdf values for all i_x"""
        x = np.array(np.float32(i_x))
        pdf = np.array(np.float32(self.__non_parametric))
        y = []
        for n in range(0, len(x)):
            idx = np.nonzero(np.logical_and( x[n] >= pdf[:,0], x[n] <= pdf[:,1]))[0]
            if len(idx) < 1:
                print "ERROR in click_distribution!", " idx = ", idx, " n = ", n
                print " x[n] = ", x[n]
                print "pdf = ", pdf
                for k in range(0, len(pdf[:,0])):
                    print "k = ", k, " x = ", x[n], " pdf = ", pdf[k,0], " ", pdf[k,1], " >= ", x[n] >= pdf[k,0], " < ", x[n] <= pdf[k,1], " pdf[0] == pdf[1]:", pdf[k,0] == pdf[k,1]
                print "and result: ", np.logical_and( x[n] >= pdf[:,0], x[n] <= pdf[:,1])
                raise ValueError("ERROR in click_distribution, __likelihood!")
            pdf_vals = pdf[idx, 2]
            if len(idx) > 1:
                y.append(max(pdf_vals))
            else:
                y.append( pdf_vals[0])
        return np.array(y).flatten()
 
    #####################################Diagnostic functions

    def pdfArea(self,i_x=None):
        if i_x is None: 
            x = np.array(self.__possible_click_times)
        else:
            x = np.array(i_x)
        y = self.__likelihood(x)
        x = np.vstack([x[0], x.reshape([len(x),1])])
        x = np.vstack([x.reshape([len(x),1]), x[-1]])
        x = np.vstack([x.reshape([len(x),1]),  x[0]])
        y = np.vstack([0.0,y.reshape([len(y),1])])
        y = np.vstack([y.reshape([len(y),1]),  0.0])
        y = np.vstack([y.reshape([len(y),1]),  0.0])
        o_area = 0.5 * np.sum( x[1:]*y[0:-1] - x[0:-1]*y[1:])
        return o_area 
    
    def __pdfParametricMarginalise(self, i_possible_click_times ):
        #This the parametric version of area under pdf
        #This should be approximately the same as value returned by pdfArea
        f_t1 =  (i_possible_click_times[:,1]).transpose()
        f_t0 =  (i_possible_click_times[:,0]).transpose()
        f_t1 /= (self.__params['click_std'] * np.sqrt(2))
        f_t0 /= (self.__params['click_std'] * np.sqrt(2))
        o_score = (0.5*(scipy.special.erf(f_t1) - scipy.special.erf(f_t0))).flatten()
        return p.log(o_score) 
        
if __name__ ==  "__main__":
    "Run click_distribtuion_example.py to see an example of this class"
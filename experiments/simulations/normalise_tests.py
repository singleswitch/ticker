import sys, time, itertools
sys.path.append('../')
sys.path.append('../../')
import pylab as p
import numpy as np
import matplotlib._cntr as cntr
import numpy.ma as ma
from scipy.optimize import fsolve
import scipy.optimize.lbfgsb as solver
import scipy.stats as s
from scipy.integrate import dblquad, quad 
import scipy.special as special
from scipy.misc import factorial 
import scipy.stats.distributions as sd
from utils import Utils, DispSettings

"""Numerical validation of click distribution normalisation constants in Ticker."""
 
class TestGauss():
    
    def __init__(self):
        self.eps = 1E-3
        self.n_samples = 10000
        self.disp = DispSettings()
        self.disp.settingsNormaliseTests()
        self.utils = Utils()
        
    ############################## Generic    
    
    
    ############################## Init
    def defaultParams(self):
        n_std = 7.0
        sigma = 0.8
        m1 = 0.0
        m2 =  1.2
        x = np.linspace( min(m1,m2) -  n_std*sigma, max(m1,m2) + n_std*sigma, self.n_samples)
        min_x = np.min(x) 
        min_x *= np.sign(min_x)
        x += min_x
        m1 += min_x
        m2 += min_x
        return (x, m1, m2, min_x, sigma )   
    
    def getRange(self):
        (x, m1, m2, min_x, sigma ) = self.defaultParams()
        return x
    
    ############################# Distributions
  
    def gauss(self, i_x, i_mean, i_sigma):
        data = i_x - i_mean
        sigma_sq = i_sigma**2
        y = np.exp( -0.5*(data**2) / sigma_sq  ) 
        y /=  np.sqrt( 2*np.pi*sigma_sq )
        return y
    
    def uniformUniformDistr(self, x, m, sigma, kstd=4.0 ):
        """Return two distributions, one with uniform prob over whole interval 
           and the second with uniform prob over shorter interval specified by
           k_std.""" 
        T  = np.max(x) - np.min(x)
        k_std = 4.0
        Ts = m - k_std*sigma
        Te = m + k_std*sigma
        y1= np.ones(len(x)) / T 
        y2 =  np.ones(len(x)) / (Te - Ts)
        y2[np.nonzero( x < Ts )[0]] = 0.0
        y2[np.nonzero( x > Te )[0]] = 0.0 
        return (y1, y2)
    
    def clickDistributionMultiple(self, sigma, min_std,  C, is_uniform, is_overlap):
        """* Return the false positive distribution, and locations of the true positives, and the data range x
           * The true positive means are randomly chosen, whereas the false positives are always
           * is_uniform: Randomly select the locations of the click distributions or spread them 
                         uniformly over the data range.
           * is_overlap: True, if the distributions are allowed to overlap with one another. Only looked at if is_uniform=False"""
        means = np.array( min_std*sigma*np.linspace(1.0, 2.0*C - 1.0, C) )
        x = np.linspace(0, np.max(means) + min_std*sigma, self.n_samples)
        (min_x, max_x) = ( min_std*sigma,  np.max(x) - min_std*sigma)
        m_str = ["%.3f  " %means[0] ]
        for c in range(1,  C):
            if not is_uniform: 
                if not is_overlap:
                    means[c] =  s.uniform.rvs(  loc=(means[c-1]+sigma*min_std), scale=sigma*min_std )
                else:
                    means[c] = s.uniform.rvs(  loc=min_x, scale=(max_x - min_x) )
            m_str.append( "%.3f  " %means[c] ) 
        idx = np.argsort(means)  
        m_str = "means: " + " ".join( list( np.array(m_str)[idx] ))
        x = np.linspace(0, np.max(means) + min_std*sigma, self.n_samples)
        T = np.max(x)
        y1 =  np.ones(len(x)) / T #The false positive signal      
        return (x, y1, means[idx], m_str)
 
    def multipleClickUniform(self, x, means, sigma, k_std=1.0):
        """  * Return  number click distributions (uniform).
            * Means: locations of the uniform distributions
            * The edge is k_std standard deviations from the location."""
        #The uniform click distribution        
        y_click = np.zeros([len(means), len(x)])
        m_str = "means: "
        for c in range(0,  len(means)): 
            m = means[c]
            Ts = m - k_std*sigma
            Te = m + k_std*sigma
            y2 =  np.ones(len(x)) / (Te - Ts)
            y2[np.nonzero( x < Ts )[0]] = 0.0
            y2[np.nonzero( x > Te )[0]] = 0.0
            y_click[c,:] = np.array(y2)  
        return y_click
    
    def multipleClickGaussian(self, x, means, sigma ):
        """  * Return  number click distributions (uniform).
            * Means: locations of the gaussian distributions """
        #The uniform click distribution        
        y_click = np.zeros([len(means), len(x)])
        m_str = "means: "
        for c in range(0,  len(means)): 
            m = means[c]
            y_click[c,:] =  self.gauss(x, m, sigma)
        return y_click
    
    ################################## Display
      
    def compareScores(self, score1, score2, disp_str):
        """Compare two scores that should be the same, also print disp string as part of error message"""
        diff = np.abs(score1 - score2)
        if diff > 1E-6:
            print "disp_str" + " score1 = ", score1, " score2 = ", score2
            raise ValueError(disp_str + " Scores are not the same")
        
    def plotVals(self, x, y1, y2, means, i_title):
        """Plot the points in means separately (corresponding in most examples to the click distribution means)"""
        self.disp.newSubFigure()
        p.plot(x, y1, 'r')
        y2_all = np.atleast_2d(y2)
        for n in range(0, y2_all.shape[0]):
            p.plot(x,  y2_all[n,:], 'k')
        max_val = max(np.max(y1), np.max(y2))
        for m in means:
            p.plot([m, m], [-0.1*max_val, 0], linewidth=1, color='b')
        p.axis('tight')
        p.title(i_title)
      
    ############################## Permutations (labels)
    
    def clickLabels(self, N, C):
        """Return the unique label permutations if there are N zeroes and C ones"""
        label_seq = []
        for n in range(0, N):
            label_seq.append('0')
        for c in range(0, C):
            label_seq.append('1')
        return np.unique( np.array([''.join(p) for p in self.utils.xpermutations(label_seq)]) )
    
    def letterLabels(self, C, L, letter_order):
        """Return the permutations of all the letter labels
           letter_order: True if letter sequences are ordered"""
        label_seq = [str(c) for c in range(0, L)]
        all_letters = np.unique(np.array([ ''.join(s[-C:]) for s in self.utils.xpermutations(label_seq)]))
        o_seq = []
        if letter_order:
            for label in all_letters:
                label_arr = np.int32(np.array(list(label)) ) 
                sorted_arr = np.sort(label_arr)
                if (np.abs(label_arr - sorted_arr) < 1E-5).all():
                    o_seq.append(label)
        return np.array(o_seq)
    
    ################################ Normalisation (Z)
         
    def computeZFromDistr(self, distr, delta_x):
        """Compute the numerical integral over T samples, and M distributions 
           distributions M X T subject to the constraint that t1<t2<..TM"""
        if distr.shape[1] == 1:
            return np.cumprod(distr.flatten())[-1]*delta_x
        else:
            prev_sum = np.cumsum(distr[0,:]) 
            for m in range(1, distr.shape[0]):
                prev_sum = np.cumsum( prev_sum[0:-1]*distr[m, m:] ) 
            return prev_sum[-1] * delta_x
    
    ################################ Main Click Distribution
    
    def marginaliseClickLogLikelihoods(self, i_f, i_C):
        """MxL likelihoods:
           * M: Supports observations labelling (false pos, true pos)
           * L: Supports letter labelling (L1, L2, .....)
           * i_f:  [p(t1 | l1)  p(t1 | l2) ... p(t1 | lL)
                       :
                    p(tM | l1   ...            p(tM | lL)]"""
        if not(np.ndim(i_f)) == 2:
            raise ValueError("Dimension should be 2, but is " + str(np.ndim(i_f)) + " instead!")
        (M,L) = i_f.shape
        n_steps = min(M, i_C) #The number of recursion loops
        f_sum = self.updateCumSum(i_f)
        for n in range(1, n_steps):
            f_new = np.atleast_2d(i_f[0:-n,0:-n])*np.atleast_2d(f_sum[1:,1:])
            f_sum = self.updateCumSum(f_new)
        return f_sum[0,0]
    
    def updateCumSum(self, i_data):
        if not(np.ndim(i_data)) == 2:
            raise ValueError("Dimension should be 2, but is " + str(np.ndim(i_f)) + " instead!")
        f_sum_row = self.updateCumSumRows(i_data)
        f_sum_col = self.updateCumSumCols(f_sum_row)     
        return f_sum_col    
            
    def updateCumSumCols(self, i_data):
        if i_data.shape[1] == 1:
            return i_data
        return np.fliplr( np.cumsum( np.fliplr(i_data), axis=1 ) )
            
    def updateCumSumRows(self, i_data):
        if i_data.shape[0] == 1:
            return i_data
        return np.flipud( np.cumsum( np.flipud(i_data), axis=0 ) )
    
    ############################ Example 1
    def example1(self):
        """One true click on spurious click, showing that the order does matter
           for different distributions but do sum to one, as we include a uniform
           prior over each."""
        (x, m1, m2, min_x, sigma ) = self.defaultParams()
        self.disp.newFigure()
        p.subplot(3,1,1); self.gaussianGaussian_1(x, m1, m2, sigma)
        p.subplot(3,1,2); self.uniformGaussian_1(x, m2, sigma)
        p.subplot(3,1,3); self.uniformUniform_1(x, m2, sigma)
        
    def gaussianGaussian_1(self, x, m1, m2, sigma): 
        y1 = self.gauss(x, m1, sigma)
        y2 = self.gauss(x, m2, sigma)
        z = self.computeZ_1(x, y1, y2)
        disp_str = "Ex1: Gaussian first (red), Gaussian second (black): \n C=%d, N=%d, Z= %.4f" % (1, 1, z)
        self.plotVals(x, y1, y2, [m1, m2],  disp_str)
        
    def uniformGaussian_1(self, x, m2, sigma): 
        T  = np.max(x) - np.min(x)
        y1 = np.ones(len(x)) / T 
        y2 = self.gauss(x, m2, sigma)
        z = self.computeZ_1(x, y1, y2)
        disp_str = "Ex1: Uniform first (red), Gaussian second (black): \n C=%d, N=%d, Z= %.4f" % (1, 1, z)
        self.plotVals(x, y1, y2, [m2], disp_str)
        
    def uniformUniform_1(self, x, m2, sigma ):
        (y1, y2) = self.uniformUniformDistr(x, m2, sigma, kstd=4.0 )
        z = self.computeZ_1(x, y1, y2)
        T = max(x) - min(x)
        za = m2 / T
        disp_str = "Ex1: Uniform first (red), Uniform second (black): \n C=%d, N=%d, Z= %.4f" % (1, 1, z)
        self.plotVals(x, y1, y2, [m2], disp_str)
        
    def computeZ_1(self, x, y1, y2):
        #It is assumed that the data is sorted: small to large
        delta_x = (x[1] - x[0])**2
        y1_sum = np.cumsum(y1)
        y_sum = np.sum( y1_sum[:-1]*y2[1:] ) * delta_x
        test_val = y1_sum[-1] * np.sum( y2 ) * delta_x
        if np.abs(1.0 - test_val) > self.eps:
            raise ValueError('Testval = %.6f, should be 1!' % test_val) 
        return y_sum

    ############################ Example 2
        
    def example2(self):
        """Multiple clicks, one is the true click and the rest are false 
           positives. All false positives come from a uniform distribution here, 
           later we'll test more different distributions."""
        (x, m1, m2, min_x, sigma ) = self.defaultParams()
        N = 4 #The number of false positives 
        (y1, y2) = self.uniformUniformDistr(x, m2, sigma, kstd=4.0 )
        self.disp.newFigure()
        p.subplot(3,1,1); self.gaussianGaussian_2(x, m1, m2, sigma, N+1, N)
        p.subplot(3,1,2); self.uniformGaussian_2(x, m2, sigma, N+1, N)
        p.subplot(3,1,3); self.uniformUniform_2(x, m2, sigma, N+1, N)
        
    def gaussianGaussian_2(self, x, m1, m2, sigma, M, N): 
        y1 = self.gauss(x, m1, sigma)
        y2 = self.gauss(x, m2, sigma)
        z = self.computeZ_2(x, y1, y2, M, N)
        disp_str = "Ex2: Gaussian first (red), Gaussian second (black): \n C=%d, N=%d, Z= %.4f" % (M-N, N, z)
        self.plotVals(x, y1, y2, [m1,m2], disp_str)
    
    def uniformGaussian_2(self, x, m2, sigma, M, N): 
        T  = np.max(x) - np.min(x)
        y1 = np.ones(len(x)) / T 
        y2 = self.gauss(x, m2, sigma)
        z = self.computeZ_2(x, y1, y2, M, N)
        disp_str = "Ex2: Uniform first (red), Gaussian second (black):  \n C=%d, N=%d, Z= %.4f" % (M-N, N, z)
        self.plotVals(x, y1, y2, [m2], disp_str)
        
    def uniformUniform_2(self, x, m2, sigma, M, N ):
        (y1, y2) = self.uniformUniformDistr(x, m2, sigma, kstd=4.0 )
        z = self.computeZ_2(x, y1, y2, M, N)
        disp_str = "Ex2: Uniform first (red), Uniform second (black): \n C=%d, N=%d, Z= %.4f" % (M-N, N, z)
        self.plotVals(x, y1, y2 ,[m2], disp_str)
        
    def computeZ_2(self, x, y1, y2, M, N):
        delta_x = (x[1] - x[0])**M #all the deltas of the integration 
        y_total = 0.0
        n_codes =  factorial(M) / (factorial(N)*factorial(M-N)) 
        Zk = factorial(M-1)   #Constant factor to make everything sum to one
        print "----------------------------------------------------------------"
        print "M = ", M, " N = ", N, " n_codes = ", n_codes, " prob per code = ",
        print 1.0 / n_codes, " Zk = ", Zk
        print "----------------------------------------------------------------"
        for c in range(0, int(n_codes)):
            #c gives the index of the true distribution
            distr = np.ones([M, len(x)])
            distr *= np.array( np.atleast_2d( y1) )
            distr[c,:] = np.array(y2)
            y_sum = self.computeZFromDistr(distr, delta_x)
            y_total += y_sum
            print "c = ", c, " y_sum = ", y_sum, " y_total = ", y_total * Zk
        return y_total*Zk
    
    ##################################### Example 3
    
    def example3(self):
        """ * Multiple clicks, more than true clicks are observed with false negatives.
            * All false positives come from a uniform distribution here, 
            * It is assumed that click distributions can overlap
            * Verify the normalisation constant. """
        C = 2
        N = 2
        L = 3
        sigma = 0.2   
        min_std = 8.0  #The min number of standard deviations of the letter locations from the edges
        display = True
        uniform = False
        overlap = False
        letter_order = True
        fr = -1       
        self.disp.newFigure()
        if display:
            print "*****************************"; print "UNIFORM" ;print"******************************"
        (x, y1, means, m_str) = self.clickDistributionMultiple(sigma, min_std,  L, uniform, overlap)
        p.subplot(2,1,1); self.uniformUniform_3( x, y1, means, C, N, L, sigma, display, letter_order, m_str, fr )
        if display:
            print "*****************************"; print "GAUSS" ;print"******************************"
        p.subplot(2,1,2); self.uniformGaussian_3( x, y1, means, C, N, L, sigma, display, letter_order, m_str, fr )

    def uniformUniform_3(self, x, y1, means, C, N, L, sigma, display, letter_order, m_str, fr ):
        normalise = True  
        y2 = self.multipleClickUniform(x, means, sigma, k_std=1.5)
        z = self.computeZ_3(x, y1, y2, C, N, L, display, letter_order, normalise, fr)
        print "Z = ", z, "  MEAN DIST = ", means[1:] - means[0:-1]
        disp_str =  "Ex3: FP Uniform (red),  TP Uniform (black): Z=%.4f, Eps=%.4f  \nN=%d, L=%d, C=%d, %s" % ( z, self.eps, N, L, C, m_str)
        self.plotVals(x, y1, y2, means, i_title=disp_str)
    
    def uniformGaussian_3(self, x, y1, means, C, N, L, sigma, display, letter_order, m_str, fr ):
        normalise = True
        y2 = self.multipleClickGaussian(x, means, sigma)
        z=self.computeZ_3(x, y1, y2, C, N, L, display, letter_order, normalise, fr)
        print "Z = ", z, "  MEAN DIST = ", means[1:] - means[0:-1]
        disp_str =  "Ex3: FP Uniform (red), TP Gaussian (black): Z=%.4f, Eps=%.4f  \nN=%d, L=%d, C=%d, %s" % ( z, self.eps, N, L, C, m_str)
        self.plotVals(x, y1, y2, means, i_title=disp_str)
        
    def computeZ_3(self, x, y1, y2, C, N, L, display, letter_order, normalise, fr):
        """  y1: 1xT Vector: False positive click distribution (Uniform), the same for all N false positives
            y2: LxT Vector: Set of True click distributions (for all L). 
            N: The number of False positives
            C: The number of observed clicks C<=L
            x: The data - if len(x)=1, delta_x is stored in x
            letter_order: True if l1 < l2 < l3 .... So the letters are read in sequence (only allowed if distribution can not overlap)
            normalise: Normalise the distribution with ZL * ZN. Also if len(x) =1, delta_x in x[0] will be used when multiplying in the integral compuation
            fr:  Use false rejection ratio to comput ZL instead of the binomial const has to be considered. Set to -1 if fr should be ignored and only the 
                binomial factor should be used."""
        M = C + N
        click_labels =  self.clickLabels(N, C)
        n_click_labels =  factorial(M) / (factorial(N)*factorial(M-N)) 
        if not( n_click_labels == len(click_labels) ):
            raise ValueError("There are " + str(len(click_labels)) + " classify labels, not " +  str(n_click_labels) )
        letter_labels =  self.letterLabels(C, L, letter_order)
        if not letter_order:
            n_letter_labels =  factorial(L) / factorial(L-C) 
            if not( n_letter_labels == len(letter_labels) ):
                raise ValueError("There are " + str(len(letter_labels)) + " classify labels, not " +  str(n_letter_labels) )
        y_total =0.0
        if len(x) == 1:
            delta_x = x[0]
        else:
            delta_x = (x[1] - x[0])**M  
        if fr > 0:
            ZL = (fr**(L-C)) * (  (1.0 -fr)**C) 
        else:
            ZL = factorial(L-C)*factorial(C) / factorial(L)
        ZN = factorial(N)
        for letter_label in letter_labels:
            letter_sum  = 0.0
            if display:
                print "----------------------------------------------------------------------------------------------------------------------------------------------------"
            for click_label in click_labels:
                distr = np.ones([M, y2.shape[1]])
                click_label_arr = np.array(list(click_label))
                idx_fp = np.nonzero( click_label_arr == '0' )[0]
                idx_tp = np.nonzero( click_label_arr == '1' )[0]
                letter_label_arr = np.int32(np.array(list(letter_label)))
                distr[idx_fp,:] = np.array(y1)
                distr[idx_tp,:] = np.array(y2[letter_label_arr,:])
                y_sum = self.computeZFromDistr(distr, delta_x)
                y_total += y_sum
                letter_sum += y_sum
                if display:
                    print click_label, ",", letter_label , ":" , "M=", M, " C=", C, " N=", N, 
                    print " y2 shape=", y2.shape,  " y_sum=", y_sum, " y_total=", y_total*ZN*ZL,
                    print "nclicks=", len(click_labels), " nletters=",
                    print len(letter_labels), " letSum=", letter_sum
        if normalise:
            y_total *= (ZN*ZL)
        if display:
            print "ZL = ", ZL, " ZN = ", ZN
        return y_total
    
    ##################################### Example 4

    def example4(self):
        """ * Multiple clicks, more than true clicks are observed with no false negatives.
            * It is assumed that true clicks do no not overlap during multiple readings of the 
              alphabet, i.e., letter_order=True.
            * Main purpose is to test marginaliseClickLogLikelhood function 
            * This example is about optimising the evaluation of the click distribution"""
        N = 3
        L = 3
        C = 1
        sigma = 0.2   
        display = False
        uniform = False
        overlap= False
        self.n_samples = 500 #1000
        min_std = 5.0 #The number of standard deviations from edges (click locations are separated 2.0*min_std)        
        (x, y1, means, m_str) = self.clickDistributionMultiple(sigma, min_std,  L, uniform, overlap)
        self.disp.newFigure()
        p.subplot(2,1,1); self.uniformUniform_4( x, means, C, N, L, sigma, display, m_str)
        p.subplot(2,1,2); self.uniformGaussian_4(x, means, C, N, L, sigma, display, m_str)
        
    def uniformGaussian_4(self, x, means, C, N, L, sigma, display, m_str  ):
        T = np.max(x)
        y2 = self.multipleClickGaussian(x, means, sigma) 
        z = self.compute_Z4(N, C, y2, T, x, display)
        print "Z = ", z, "  MEAN DIST = ", means[1:] - means[0:-1]
        y1 = np.ones(len(x)) / T
        disp_str =  "Ex4: FP Uniform (red),  TP Gaussians (black): Z=%.4f, Eps=%.4f  \nN=%d, L=%d, C=%d, %s" % ( z, self.eps, N, L, C, m_str)
        self.plotVals(x, y1, y2, means, i_title=disp_str)

    def uniformUniform_4(self,  x, means, C, N, L, sigma, display, m_str  ):
        T = np.max(x)
        y2 = self.multipleClickUniform(x, means, sigma, k_std=1.0) 
        z = self.compute_Z4(N, C, y2, T, x, display) 
        print "Z = ", z, "  MEAN DIST = ", means[1:] - means[0:-1]
        y1 = np.ones(len(x)) / T
        disp_str =  "Ex4: FP Uniform (red),  TP Uniform (black): Z=%.4f, Eps=%.4f  \nN=%d, L=%d, C=%d, %s" % ( z, self.eps, N, L, C, m_str)
        self.plotVals(x, y1, y2, means, i_title=disp_str)
        
    def compute_Z4(self, N, C, y2, T, x, display):
        (M, L) = (N + C, y2.shape[0])
        (time_idx, cur_index, z) = (np.int32(-1*np.ones(M)), 0, 0.0)
        (time_idx, cur_index, z) = self.__compute_Z4( y2, N, C, time_idx, cur_index, z )
        print "z = ", z
        (x_tmp, y1, letter_order, normalise) = ( np.array([1.0]), np.ones(len(x)),True, False)
        z2 = self.computeZ_3(x_tmp, y1, y2, C, N, L, display, letter_order, normalise)
        self.compareScores(z, z2, "score1=z, score2=z2")
        verify_two_clicks = False
        if M==2 and verify_two_clicks:
            (y_total1, y_total2) = self.__compute_Z4_2( y2,C, N)
            self.compareScores(z, y_total1, "score1=z, score2=y_total1")
        Zn = factorial(N)
        Zl = factorial(L-C)*factorial(C) / factorial(L)
        delta_x = (x[1]-x[0])**M
        pn = Zn / (T**N)
        z *= ( delta_x*pn*Zl)
        return z
 
    def __compute_Z4(self,  y2, N, C, time_idx, cur_index, score ):
        """Calling function marginaliseClickLogLikelihoods recursively, so that
        t1 < t2 < t3 ....."""
        if cur_index == 0:
            start_val = 0
        else:
            start_val = time_idx[cur_index-1]+1 
        for t in range(start_val, self.n_samples):
            time_idx[cur_index] = t
            if (cur_index+1) == len(time_idx):
                f = np.array(y2[:,time_idx])
                score  +=  self.marginaliseClickLogLikelihoods(f.transpose(), C)
            else:
                if cur_index == 0:
                    if np.abs(time_idx[cur_index] % 10) < 1E-6:
                        print "Z4: ", time_idx[cur_index]
                cur_index += 1
                (time_idx, cur_index, score) = self.__compute_Z4(y2, N, C, time_idx, cur_index, score)
        time_idx[cur_index] = -1
        cur_index -= 1
        return (time_idx, cur_index, score)
    
    def __compute_Z4_2(self, y2, N, C):
        (L, n_samples) = y2.shape
        (y_total1, y_total2) = (0.0, 0.0)
        (x, y1) = ( np.array([1.0]), np.array([1.0]) )
        tmp_labels = self.letterLabels( C, L, letter_order=True)
        letter_labels = np.array( [np.int32(np.array(list(label))) for label in tmp_labels] )
        for t1 in range(0, n_samples):
            if np.abs(t1%10) < 1E-6:
                print t1
            for t2 in range(t1+1, n_samples):
                cols = np.array([t1, t2])
                data = np.array(y2[:,cols])
                y_total1 +=  self.marginaliseClickLogLikelihoods(data.transpose(), C)
                y_total2 += self.computeZ_3(x, y1,data,C,N,L, display=False, letter_order=True, normalise=False )
                self.compareScores(y_total1, y_total2, "score1=y_total1, score2=y_total2, x idx= " + str(cols))
        return (y_total1, y_total2)
   
    ##################################### Example 5
    
    def example5(self):
        """ * Multiple clicks, more than true clicks are observed with false negatives.
            * All false positives come from a uniform distribution here, 
            * False rejection ratio is included in here, binomial coefficients should cancel out
            * Verify the normalisation constant. """
        N = 4
        L = 2
        fr = 0.25        
        sigma = 0.2   
        min_std = 8.0  #The min number of standard deviations of the letter locations from the edges
        display = False
        uniform = False
        overlap = False
        letter_order = True
        self.disp.newFigure()
        (x, y1, means, m_str) = self.clickDistributionMultiple(sigma, min_std,  L, uniform, overlap)
        p.subplot(2,1,1);  self.uniformUniform_5(x, y1, means, N, L, sigma, display, letter_order, m_str, fr )
        p.subplot(2,1,2); self.uniformGaussian_5( x, y1, means, N, L, sigma, display, letter_order, m_str, fr )
    
    def uniformUniform_5(self, x, y1, means, N, L, sigma, display, letter_order, m_str, fr ):
        normalise = True  
        y2 = self.multipleClickUniform(x, means, sigma, k_std=1.5)
        z_all = [  self.computeZ_3(x, y1, y2, C, N, L, display, letter_order, normalise, fr) for C in range(1, L+1)]
        z_all.insert(0, fr**L)
        z = np.sum(np.array(z_all))
        print "Uniform: z_all = ", z_all, " TOTAL ", z
        disp_str =  "Ex5: FP Uniform (red),  TP Uniform (black): Z=%.4f, Eps=%.4f  \nN=%d, L=%d, C=%d, %s" % ( z, self.eps, N, L, C, m_str)
        self.plotVals(x, y1, y2, means, i_title=disp_str)
    
    def uniformGaussian_5(self, x, y1, means, N, L, sigma, display, letter_order, m_str, fr ):
        normalise = True  
        y2 = self.multipleClickGaussian(x, means, sigma)
        z_all = [  self.computeZ_3(x, y1, y2, C, N, L, display, letter_order, normalise, fr) for C in range(1, L+1)]
        z_all.insert(0, fr**L)
        z = np.sum(np.array(z_all))
        print "Gaussian: z_all = ", z, " TOTAL ", z
        disp_str =  "Ex5: FP Uniform (red),  TP Gaussian (black): Z=%.4f, Eps=%.4f  \nN=%d, L=%d, C=%d, %s" % ( z, self.eps, N, L, C, m_str)
        self.plotVals(x, y1, y2, means, i_title=disp_str)
    
    ##################################### Example 6
    def example6(self):
        """ * Same as example 5 except we compute P(t | M, N, C, n, c) as well."""
        M =3
        L = 3
        sigma = 0.2   
        min_std = 8.0    #The min number of standard deviations of the letter locations from the edges
        display = True
        uniform = False  #The Gaussians associated with the repetitions are not equally spaced
        overlap = False  #The Gaussians are not allowed to overlap
        self.disp.newFigure()
        (x, y1, means, m_str) = self.clickDistributionMultiple(sigma, min_std,  L, uniform, overlap)
        self.uniformGaussian_6( x, y1, means, M, L, sigma, display, m_str )
    
    def uniformGaussian_6(self, x, y1, means, M, L, sigma, display, m_str  ):
        normalise = True  
        #y2 = self.multipleClickGaussian(x, means, sigma)
        
        y2 = self.multipleClickUniform( x, means, sigma, k_std=1.0)
        max_C = min(M, L)
        z_all = [ str(self.computeZ_6(x, y1, y2, C, M-C, L, display )) for C in range(1, max_C+1)]
        z_all = " ".join(z_all) 
        print "Gaussian: z_all = ", z_all
        disp_str =  "Ex6: FP Uniform (red),  TP Gaussian (black): Z=%s, \nM=%d, L=%d, max_C =%d, %s" % ( z_all, M, L, max_C, m_str)
        self.plotVals(x, y1, y2, means, i_title=disp_str)
    
    
    def computeZ_6(self, x, y1, y2, C, N, L, display ):
        """ y1: 1xT Vector: False positive click distribution (Uniform), the same for all N false positives
            y2: LxT Vector: Set of True click distributions (for all L). 
            N: The number of False positives
            C: The number of observed clicks C<=L
            x: The data  
            letter_order: l1 < l2 < l3 .... So the letters are read in sequence (only allowed if distribution can not overlap)
            """
        M = C + N
        click_labels =  self.clickLabels(N, C)
        n_click_labels =  factorial(M) / (factorial(N)*factorial(M-N)) 
        letter_labels =  self.letterLabels(C, L, True)
        y_total =0.0
        delta_x = (x[1] - x[0])**M  
        ZL = factorial(L-C)*factorial(C) / factorial(L)
        ZN = factorial(N)
        ZT =  factorial(M) / factorial(C) 
        
        if display:
            print "****************************************************************"
            print "M=", M, " C=", C, " N=", N, " y2 shape=", y2.shape,
            print "nclicks=", len(click_labels), " nletters=", len(letter_labels)
            print "ZL = ", ZL, " ZN = ", ZN, " ZT = ", ZT, " 1/ZT = ", 1.0/ZT
            print "****************************************************************"
        for letter_label in letter_labels:
            letter_sum  = 0.0
            if display:
                print "----------------------------------------------------------------------------------------------------------------------------------------------------"
            for click_label in click_labels:
                distr = np.ones([M, y2.shape[1]])
                click_label_arr = np.array(list(click_label))
                idx_fp = np.nonzero( click_label_arr == '0' )[0]
                idx_tp = np.nonzero( click_label_arr == '1' )[0]
                letter_label_arr = np.int32(np.array(list(letter_label)))
                distr[idx_fp,:] = np.array(y1)
                distr[idx_tp,:] = np.array(y2[letter_label_arr,:])
                y_sum = self.computeZFromDistr(distr, delta_x)
                y_total += y_sum
                letter_sum += y_sum
                if display:
                    print click_label, ",", letter_label , ":", 
                    print " y_sum=", y_sum, " y_total=", y_total*ZN*ZL, " letSum=", letter_sum 
        return y_total*ZN*ZL
    
    
    ##################################### Example 7
    
    def example7(self):
        """ * Sample from poisson process and Gaussian - count to see if we get the same distribution
           * The samples are not ordered here - associated with the letter repetition criterium."""
        (M, C, T, n_samples, w, uniform ) = (2, 1, 5, 1E6, 0.1, False)
        u = np.array( [2.0, 4.5])  
        if uniform:
            w = 1.5
        samples = np.zeros([n_samples, 2])
        #Draw the true positive samples
        samples[:,0]  = sd.norm.rvs(loc=0, scale=w, size=(1,n_samples)).flatten()  
        #Draw the false positives
        samples[:,1] = sd.uniform.rvs(loc=(-T), scale=(2.0*T), size=(1,n_samples)).flatten()
        #Enforce the constraint time constraint
        arg_sort = np.argsort(samples, axis=1)
        samples = np.sort(samples, axis=1) 
        [rows, idx_tp] = np.nonzero( arg_sort < 1)
        if len(u) < 2:
            samples += u[0]
        else:
            #Assign labels to the true positives 
            tp_labels = np.int32( np.logical_not( sd.bernoulli.rvs(1.0/len(u),size=n_samples).flatten() ) )
            for (nc, uc) in enumerate(u):
                idx_tp_cluster = np.nonzero(tp_labels == nc)[0]
                cols =  idx_tp[ idx_tp_cluster] 
                samples[idx_tp_cluster, :] += uc
        #Choose only betweens in interval (0,T)
        idx = np.nonzero(np.logical_and(samples[:,0] < T, samples[:,0] > 0))[0]
        (samples, arg_sort) = ( samples[idx,:], arg_sort[idx] )
        idx = np.nonzero(np.logical_and(samples[:,1] < T, samples[:,1] > 0))[0]
        (samples, arg_sort) = ( samples[idx,:], arg_sort[idx] )
        #Display
        [rows, idx_tp] = np.nonzero( arg_sort < 1)
        [rows, idx_fp] = np.nonzero( arg_sort > 0)
        self.plotDensity(samples, u, T, w, uniform ) 
        self.plotSamples(samples, u, T, w, uniform, samples[range(0,len(idx_tp)),idx_tp], samples[range(0,len(idx_tp)),idx_fp])
        #self.reflect2D( np.atleast_2d( samples[idx,:] ), False) 
        #self.testReflect2D()
        
    def testReflect2D(self):        
        self.disp.newFigure()
        samples = np.atleast_2d( np.array([[2.0, 1.0], [3.0, 0.5], [4.0, 0.25]]) )
        (crosspoints, samples_after) = self.reflect2D(samples, True)
        p.plot( np.linspace(0,5,100), np.linspace(0,5,100), 'k' )
        p.plot(samples[:,0], samples[:,1], 'rx')
        p.plot(samples_after[:,0], samples_after[:,1], 'rx')
        p.plot( crosspoints[:,0], crosspoints[:,1], 'x')
        for n in range(0, samples.shape[0]):
            p.plot([samples[n,0], crosspoints[n,0]], [samples[n,1], crosspoints[n,1]], 'k--')
            p.plot([samples_after[n,0], crosspoints[n,0]], [samples_after[n,1], crosspoints[n,1]], 'k--')
        p.axis('image')
        p.show()        
            
    def reflect2D(self, samples, i_return_cross=False): 
        c = samples[:,0] + samples[:,1]
        m = 1.0 
        l1 = np.array([-m, 1.0, 0 ])
        l2 = np.ones([samples.shape[0], 3])
        l2[:,0] /= m
        l2[:,2] = -(samples[:,1] + (1.0/m)* samples[:,0])
        crosspoints = np.cross(l1, l2)
        crosspoints = crosspoints[:,0:2] /  np.atleast_2d(crosspoints[:,-1]).transpose()
        if i_return_cross:
            return (crosspoints,  2.0*crosspoints - samples)
        else:
            return np.atleast_2d( 2.0*crosspoints - samples )
        
    def plotSamples(self,  samples, u, T, w, uniform, tp_samples, fp_samples):
        n_samples = np.atleast_2d(samples).shape[0]
        (min_x, max_x , min_y, max_y) = self.axisLimits(  uniform, u, T, w)
        self.disp.newFigure()
        #The true positive samples
        p.plot( tp_samples,  min_y/4.0* np.ones( len(tp_samples)),  'kx', alpha=0.05)
        p.plot( u, np.zeros(len(u)), 'ko', alpha=1.0)
        #The false positive samples
        p.plot( fp_samples,  3.0*min_y* np.ones(len(fp_samples))/4.0,  'rx', alpha=0.05)
        p.plot( [0.5*T], [0.5*min_y], 'ro', alpha=1.0)
        self.plotDistributions_7(uniform, u, T, w)
        #p.axis((min_x, max_x , min_y, max_y)) 
        
    def plotDensity(self,  samples, u, T, w, uniform ):
        #Bin the samples and make 2D plot 
        self.disp.newFigure()
        p.plot( np.linspace(0,T,100), np.linspace(0,T,100), 'k' )
        p.plot( np.linspace(0,T,100), np.linspace(0,T,100), 'w' )
        #p.plot( u, u, 'ro')
        #p.plot(samples[:,0], samples[:,1], 'r.')
        ngrid = 300
        hist2D, xedges, yedges = np.histogram2d(samples[:,1], samples[:,0], bins=ngrid,
            normed=True,  range=[[0, T], [0, T]])
        extent = [ xedges[0], xedges[-1], yedges[0], yedges[-1]]
        p.imshow(hist2D, origin='lower', extent=extent,interpolation='nearest') #(0,T,0,T) )
        p.xlabel('t1'); p.ylabel('t2'); p.colorbar()

    def axisLimits(self, uniform, u, T, w):
        if uniform:
            max_y = max( 1.0 / w, 1.0 / T)
        else:
            max_y = max( 1.0 / T, 1.0 / (np.sqrt(2*np.pi) * w ))
        min_x =  min(-0.1,  np.min(u)  -5.0*w )
        max_x = max(5.0*w + max(u), T+0.1)  
        (min_y, max_y) = ( -3.0*max_y/50.0, max_y + max_y/100.0 ) 
        print "max_x = ", min_x, " max_x = ", max_x, " min_y = ", min_y, " max_y = ", max_y
        return  (min_x, max_x , min_y, max_y)
        
    def plotDistributions_7(self, uniform, u, T, w):
        (x_fp, y_fp) = self.samples_7(T,0.5*T, True)
        p.plot(x_fp, y_fp ,  'r')
        for uc in u:
            (x_tp, y_tp) = self.samples_7(w, uc, uniform)
            p.plot(x_tp, y_tp , 'k')
    
    def samples_7(self, w, u, i_uniform):
        """ * Return a set of discrete samples to represent a Uniform pdf, centred at u with width w - for display purposes"""
        if i_uniform:
            x = list( np.linspace(-0.5*w, 0.5*w, 100) + u)
            y =  list( np.ones(len(x)) / w)
            y.insert(0, 0)
            y.append(0)
            x.insert(0, x[0])
            x.append(x[-1])
        else:
            x = np.linspace(-5.0*w,  5.0*w, 100) + u 
            y = np.exp(  ( -(x - u)**2 ) /  ( 2.0 * (w**2))) / (np.sqrt(2.0 * np.pi) * w)
        return (np.array(x), np.array(y))

    ##################################### Example  8
    
    def example8(self):
        """ * Same as example 7 except that the true positive labelling is also considered. """
        ( T, n_samples, min_std, uniform, overlap, self.n_samples) = ( 5, int(1E6), 10, False, False, 100 )
        N = 1  # Number of false positives
        C = 1  # Number of true positives
        L = 3  #  Number of letter repetitions 
        sigma = 0.1 #Gaussian standard deviation]
        """ Extract symmetrical samples according to the number of fp and fn
            * For fp there will be nsamples x N uniform samples, over range 2T, symmetrical over the origin. 
            * For tp there will be nsamples x C uniform samples,  symmetrical over the origin. 
            * The true positive mean assignments will be done according to our labelling scheme.
            * The true positive means are return (random)""" 
        (samples, u, tp_samples, fp_samples) =  self.getSamples(T, n_samples, sigma, min_std,  L, uniform, overlap, C, N)
        #Choose samplesd.permutations that fall  in interval (0,T)
        """for n in range(0, samples.shape[1]):
            idx = np.nonzero(np.logical_and(samples[:,n] < T, samples[:,n] > 0))[0]
            samples = samples[idx,:] 
            if C > 0:
                tp_samples = tp_samples[idx,:]
            if N > 0:
                fp_samples = fp_samples[idx,:]"""
        #Display
        self.plotDensity(samples, u, T, sigma, uniform ) 
        self.plotSamples(samples, u, T, sigma, uniform, tp_samples, fp_samples)
     
    def random_permutations(permuation_range, n_samples):
        while n:
            np.random.shuffle(permuation_range)
        yield list(l)
        permuation_range
        
    def getSamples(self, T, n_samples, sigma, min_std,  L, uniform, overlap, C, N):
        M = C + N
        (u, tp_samples, fp_samples, idx_tp_clusters) = ( np.array([]), np.array([]), np.array([]), np.array([]))
        samples = np.zeros([n_samples, M])         #We draw from a C dimensional hyper cube for each false positive
        """
        * Draw the true positive samples
        * Decide which Gaussians correspond to the true positives
        * Sample from high dimensional sphere
        * The translation add the end relies on the fact that the random indices into 
          the permutations of the letter labellings are sorted as well as the menas 
          of Gaussian bumps.
        """
        if C > 0:
            (tmp_x, tmp_y1, u, m_str) = self.clickDistributionMultiple(sigma, min_std,  L, uniform, overlap)
            #u = np.array([2.0, 3.0, 4.5])
            samples[:,0:C] = sd.norm.rvs(loc=0, scale=sigma, size=(n_samples,C)) 
            tp_clusters =  np.sort( np.array(list(itertools.combinations(range(L), C))), axis=1)
            tp_clusters_rand_int = np.random.randint(0, tp_clusters.shape[0], size=n_samples )
            idx_tp_clusters = np.array(tp_clusters[tp_clusters_rand_int,:])
            samples[:, 0:C] +=  u[idx_tp_clusters] 
        """
        *Draw the false positives
        *Decide which means should be added to the false positive samples:
        * This step is equivalent to sampling from a hypercube dim L and then throwing aways points 
          in some dimensions.
        """ 
        if N > 0:
            samples[:,C:] = sd.uniform.rvs(loc=(-T), scale=(2.0*T), size=(n_samples, N))
            fp_clusters_rand_int = np.random.randint(0, L, size=(n_samples,N) )
            samples[:,C:] += u[fp_clusters_rand_int]
        #Enforce the time-ordering constraint (symmetrical problem)
        arg_sort = np.argsort(samples, axis=1)
        samples = np.sort(samples, axis=1)
        #Identify the true positives and false positives
        if C > 0:
            [rows, idx_tp] = np.nonzero( arg_sort < C)
            tp_samples = np.array(samples[rows, idx_tp])
        if N > 0:
            [rows, idx_fp] = np.nonzero( arg_sort >= C)
            fp_samples = np.array(samples[rows, idx_fp])
        return (samples, u, tp_samples, fp_samples)

if __name__ ==  "__main__":
    """A 1D visualisation of the posterior associated with click noise"""
    g = TestGauss()
    g.example3()
    #g.example8()
    p.show()

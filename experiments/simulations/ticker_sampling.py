
import scipy.stats.distributions as sd
import numpy as np
import itertools

class TickerSampling():
 
    ################################# Main
   
    def sample(self, i_click_distr, i_letter, i_return_labels=False,  i_n_rejection_samples=1E6, i_display=False):
        """
        Input:
            * Draw one sample from posterior conditioned on the letter:
            * i_letter: For example "a"
            * i_return_labels: Return false positive and true positives separately
            * 
        Output: 
            * samples: A vector of length M (also drawn randomly), where M=N+C
            * gauss_mu: the mean of the gaussian associated with letter i_letter   
        """ 
        L = i_click_distr.loc.shape[1]  
        (N, C) = (0, 0)
        N = self.sampleNumberFalsePositives(i_click_distr, 1)
        C = self.sampleNumberTruePositives(i_click_distr, L, 1)
        return self.samplesJoint(i_click_distr, i_letter, L, C, N, 1, i_return_labels, i_n_rejection_samples, i_display )
    
    ################################# Sampling Routines
    
    def sampleNumberFalsePositives(self, i_click_distr, n_samples ):
        fp_rate =  i_click_distr.fp_rate
        if np.abs(fp_rate) < 1E-10:
            return 0
        #Sample the number of false positives from a poisson 
        poisson_mu  = fp_rate * i_click_distr.T
        N = sd.poisson.rvs(poisson_mu,loc=0,size=n_samples)
        return N

    def sampleNumberTruePositives(self, i_click_distr, L, n_samples):
        #Sample the number of true positives from a binomial: L =max number of alphabet repetitions
        C = sd.binom.rvs(L, 1.0-i_click_distr.fr,loc=0,size=n_samples)
        return C
    
    def samplesJoint(self, i_click_distr, i_letter, L, C, N, n_samples, i_return_labels, i_n_rejection_samples, i_display=False):
        """
        * Sample from the joint probability of everything given the letter to write
        * If i_return_labels is true, return the false positives and true positives 
          with the samples."""
        M = C + N
        (gauss_mu, tp_samples, fp_samples, samples ) = ( np.array([]), np.array([]), np.array([]), np.array([]))
        if M < 1:
            if  i_return_labels:
                return (gauss_mu, tp_samples, fp_samples, samples ) 
            return (np.array([[]]), N, C)
        rejection_samples = np.int32(i_n_rejection_samples)  
        T = i_click_distr.T
        samples = np.zeros([n_samples, M]) #We draw from a C dimensional hyper cube for each false positive
        arg_sort = np.zeros([n_samples,M], dtype=np.int32)
        cur_samples = 0 #The number of kept samples
        while cur_samples < n_samples:
            tmp_samples = np.zeros([rejection_samples, M])
            if C > 0:
                (tmp_samples[:, 0:C], gauss_mu) = self.gaussLetterSampling(i_click_distr, 
                    C, i_letter, rejection_samples)
            if N > 0:
                tmp_samples[:,C:] = self.poissonProcesTimesSample(i_click_distr, 
                    rejection_samples, L, N, i_letter)
            #Enforce the time-ordering constraint (symmetrical problem)
            tmp_arg_sort = np.argsort(tmp_samples, axis=1)
            tmp_samples = np.sort(tmp_samples, axis=1)
            new_samples = 0
            #print "cur_samples = ", cur_samples, " n_samples = ", n_samples, " N = ", N, " C = ", C
            for n in range(0, tmp_samples.shape[1]):
                idx = np.nonzero(np.logical_and(tmp_samples[:,n] < T, tmp_samples[:,n] > 0))[0]
                if len(idx) < 1:
                    break
                tmp_samples = np.atleast_2d(tmp_samples[idx,:])
                tmp_arg_sort = tmp_arg_sort[idx,:]
                new_samples = tmp_samples.shape[0] 
                if i_display:
                    print "n=", n, " C = ", C, " N = ", N, " nsamples = ", n_samples, " tmp_samples shape = ", 
                    print tmp_samples.shape, " nsamples = ", n_samples, " cur_samples = ", cur_samples,
                    print " new_samples= ", new_samples, " T = ", T
            if new_samples > 1:
                last_sample = cur_samples+new_samples
                if last_sample > n_samples:
                    last_sample = n_samples
                idx = range(cur_samples, last_sample)
                samples[idx,:] = np.array(tmp_samples[0:len(idx),:])
                arg_sort[idx,:] = np.array(tmp_arg_sort[0:len(idx):])
                cur_samples  += new_samples
        if i_return_labels:
            #Identify the true positives and false positives
            if C > 0:
                [rows, idx_tp] = np.nonzero( arg_sort < C)
                tp_samples = np.array(samples[rows, idx_tp])
            if N > 0:
                [rows, idx_fp] = np.nonzero( arg_sort >= C)
                fp_samples = np.array(samples[rows, idx_fp]) 
            return (samples, gauss_mu, tp_samples, fp_samples)
        return (samples, N, C)
    
    def poissonProcesTimesSample(self, i_click_distr, n_samples, L, N, i_letter):
        """
        * Draw from P(t1,...,tN | N, fp_rate, T)
        * Draw from Uniform distribution, origin at zero. 
        * Then select any of the L letter repetition intervals in which the 
          random variable falls (this selects the dimension of the hypercube
          encompassing the hypersphere).
        """
        gauss_mu = i_click_distr.getLetterLocation(i_letter, i_with_delay=True)
        T = i_click_distr.T
        letter_times= sd.uniform.rvs(loc=(-T), scale=(2.0*T), size=(n_samples, N))
        fp_clusters_rand_int = np.random.randint(0, L, size=(n_samples,N) )
        letter_times +=  gauss_mu[fp_clusters_rand_int]
        return letter_times
    
    def poissonProcessInterarrivalTimeSampling(self, i_click_distr): 
        """ * Generate samples from a 1D Poisson process
              by sampling the interarrival times
            * In this case, the false posivies in the gesture
              switch are modelled using a Poisson Process
            * From Luc Devroye's book - Chapter 6"""
        if np.abs( i_click_distr.fp_rate ) < 1E-20:
            return np.array([])
        o_samples = []
        t = 0
        while True:
            #Inter arrival times have an exponential distribution
            t = t + s.expon.rvs(loc=0,scale=(1.0/i_click_distr.fp_rate),size=1)  
            if t >= i_click_distr.T:
                return np.array( o_samples )
            else:
                o_samples.append( t )
            
    def gaussLetterSampling(self, i_click_distr, i_C, i_letter, i_n_samples):
        gauss_mu = i_click_distr.getLetterLocation(i_letter, i_with_delay=True)
        sigma = i_click_distr.std
        #Get the sample times
        sample_times = sd.norm.rvs(loc=0, scale=sigma, size=(i_n_samples, i_C)) 
        L = len(gauss_mu)
        tp_clusters =  np.sort( np.array(list(itertools.combinations(range(L), i_C))), axis=1)
        #Select the means to which the gaussians belong
        tp_clusters_rand_int = np.random.randint(0, tp_clusters.shape[0], size=i_n_samples )
        idx_tp_clusters = np.array(tp_clusters[tp_clusters_rand_int,:])
        sample_times +=  gauss_mu[idx_tp_clusters]
        return (sample_times, gauss_mu)
  
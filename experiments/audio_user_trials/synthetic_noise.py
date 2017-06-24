
import numpy as np
import scipy.stats.distributions as sd

class SyntheticNoise():
    
    def __init__(self, i_sub_session, i_extra_wait_time=0.0):
        if i_sub_session == 1:
            #Always one practise sentence and 1-2 real sentences
            self.gauss_delay = 0.8 
            self.gauss_std = 0.05
            self.fp_rate = 0.0 
            self.fr = 0.0
        elif i_sub_session == 2:
            self.gauss_delay = 1.5
            self.gauss_std = 0.05
            self.fp_rate = 1.0 / 3
            self.fr = 0.1
        self.delay = self.gauss_delay + 3.0*self.gauss_std + i_extra_wait_time
        print "DELAY NOISE = ", self.delay
    
    #################################### Main 
    
    def isFalseRejection(self):
        if np.abs(self.fr) < 1E-10:
            return False
        is_fr = bool(sd.bernoulli.rvs(self.fr))
        return is_fr
    
    def sampleFalsePositives(self, i_period):
        N = self.sampleNumberFalsePositives(i_period)
        if N < 1:
            return np.array([])
        return np.sort(i_period*sd.uniform.rvs(size=(1, N))).flatten()
        
    def sampleGaussOffset(self):
        if np.abs(self.gauss_std) < 1E-10:
            return self.gauss_delay
        sample = sd.norm.rvs(loc=self.gauss_delay, scale=self.gauss_std) 
        if sample < 0.0:
            sample = 0.0
        return sample
    
    ################################## Private
    
    def sampleNumberFalsePositives(self, i_T):
        if np.abs(self.fp_rate) < 1E-10:
            return 0
        #Sample the number of false positives from a poisson 
        poisson_mu  = self.fp_rate * i_T / 60.0
        N = sd.poisson.rvs(poisson_mu)
        return N
    
    ################################## Examples
    
    def example(self):
        T = 7.3693625
        minutes = 4.0
        K = int( minutes*60.0 / T + 0.5)
        N_tot = 0
        for n in range(0, K):
            N = self.sampleNumberFalsePositives(T)
            N_tot += N
        print "N total = ", N_tot
        print "Just the period: ", self.sampleNumberFalsePositives(minutes*60.0 )
    
if __name__ ==  "__main__":
    s  = SyntheticNoise(i_sub_session=2)
    s.example()
   
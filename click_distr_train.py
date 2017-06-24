
import numpy as np
import sys, time, itertools
sys.path.append('../../')
from scipy.misc import factorial 
from utils import Utils
from kernel_density import NonParametricPdf
import scipy.stats.distributions as sd
import copy

class ClickDistributionTraining():
    ########################################## Init
    def __init__(self):
        self.disp = False
        self.ignore_priors = False
        self.initParams()
        self.max_iterations = 10  #In case of iterative training
        self.max_fp = 10          #Maximum number of clicks that will be evaluated
        self.eps_dist = 1E-1      #Difference in score
        self.utils = Utils()
         
    def initParams(self):
        #Set the fixed hyperparameters
        if self.ignore_priors:
            self.delay_prior = {'mean': 0.001, 'scale': 0}  #mean: 0.35 (switch), choose broad prior 
            self.precision_prior = {'a': 0.5, 'b':0.0  }
            self.fp_prior = {'a' : 1.0, 'b' : 0.0} #0.01: Roughly one every 2 minutes  
            self.fr_prior = {'a' : 1.0, 'b' : 1.0} #fr=0.1 without any obsevations    
            return 
        self.delay_prior = {'mean': 0.1, 'scale': 0.01} 
        self.precision_prior = {'a': 2.0, 'b':0.001 }            
        self.fp_prior = {'a' : 1.49, 'b' : 61.0} #0.01: Roughly one every 2 minutes (here it is number per seconds)
        self.fr_prior = {'a' : 2.0, 'b' : 10.0} #fr=0.1 without any observations   
       
    def initTraining(self, i_observations, i_R, i_T, i_params):
        self.R = i_R  #Number of times the alphabet has been repeated
        self.T = i_T  #Duration of alphabet repetition in seconds (for false pos calculations)  
        self.obs = i_observations #Shallow copy of observations
        self.H = float(len(i_observations))
        self.M_sum = 0.0
        for h in range(0, len(i_observations)):
            M = i_observations[h].shape[1]
            if  (M - i_R)  > self.max_fp:
                print "Warning:: Maximum number of false positives received : Note processing ", M-i_R
                continue
            self.M_sum += M
        log_prior_score = self.logPriorScore(i_params)
        (delay, std, fr, fp_rate) = i_params
        (log_score, empty_weights) = self.logLikelihood(i_params, self.utils.likelihoodGauss, std, False)
        return log_prior_score + log_score
   
    ####################################### Main
 
    def train(self, i_observations):
        return  

    #################################### Scores
   
    def normaliseWeights(self, io_weights, i_z ):
        for i in range(0, len(io_weights)):
            io_weights[i] = np.exp( io_weights[i] - i_z)
        return io_weights

    def logLikelihood(self, i_params,  i_score_func, i_score_params, i_save_weights=False):
        (delay, std, fr, fp_rate) = i_params 
        (z, weights) = (0.0, [])
        for h in range(0, int(self.H)):
            (z_h, log_weights_h, M ) = (-np.inf, [], self.obs[h].shape[1] )
            click_time_scores = (i_score_func(self.obs[h].flatten()-delay, i_score_params, i_log=True))
            click_time_scores = click_time_scores.reshape(self.obs[h].shape)
            if  (M - self.R)  > self.max_fp:
                continue
            for C in range(0, min(M, self.R)+1): 
                noise_term = self.getNoiseTerm(C, M, fr, fp_rate)
                if C < 1:
                    z_h = self.utils.elnsum(z_h, noise_term )
                    #if i_save_weights:
                    #print " h=%d, M=%d, C=%d, R=%d, z_h=%.2f, r=0, t=-, w=[%.4f]" % (h,M,C, self.R,z_h, z_h)
                else:
                    (idx_letters, idx_tp) = self.getTrueClickIdx(C, M)
                    for r in range(0,idx_letters.shape[0]):
                        idx = np.array(idx_letters[r, :])
                        true_pos_scores = np.atleast_2d( click_time_scores[idx, idx_tp] )
                        click_terms = np.atleast_1d( np.sum(true_pos_scores , axis=1).flatten()) + noise_term
                        z_h = self.utils.elnsum(z_h, self.utils.expTrick(np.atleast_2d(click_terms))[0]) 
                        if i_save_weights:
                            log_weights_h.append(np.array(click_terms))
                            w_s =  self.utils.stringVector(log_weights_h[-1])
                            t_s = self.utils.stringVector(self.obs[h][idx, idx_tp].flatten())
                            c_s =  self.utils.stringVector(click_terms.flatten())
                            #print " h=%d, M=%d, C=%d, R=%d, z_h=%.2f, r=%d, t=%s, w=%s, nt=%.4f, ct=%s" % (h,M,C, self.R,z_h,r,t_s,w_s, noise_term, c_s)
            z += z_h 
            if i_save_weights:
                weights.append(self.normaliseWeights(log_weights_h, z_h))
        return (z, weights )
    
    def logPriorScore(self, i_params):
        (delay, std, fr, fp_rate) = i_params
        if self.ignore_priors:
            prior_score = 0.0
        else:
            gauss_scale = std / np.sqrt(self.delay_prior['scale'])
            pi_gauss = np.log( sd.norm.pdf(delay,loc=self.delay_prior['mean'] ,scale=gauss_scale))
            beta = 1.0 / (std**2) 
            beta_scale = 1.0/self.precision_prior['b']
            pi_beta =  np.log(sd.gamma.pdf( beta ,self.precision_prior['a'], scale=beta_scale))
            if np.abs(self.fp_prior['b']) < 1E-6:
                (pi_fp, fp_scale) = (0.0, 0.0) 
            else:
                fp_scale = 1.0/self.fp_prior['b'] 
                pi_fp = np.log( sd.gamma.pdf( fp_rate  ,self.fp_prior['a'], scale=fp_scale ) )
            pi_fr = np.log( sd.beta.pdf(fr, self.fr_prior['a'], self.fr_prior['b']) )
            prior_score = pi_gauss + pi_beta + pi_fp + pi_fr 
            if self.disp:
                print "pi_gauss = ", pi_gauss, " val = ", delay, " mean = ", self.delay_prior['mean'] , " scale = ", gauss_scale
                print "pi_beta = ", pi_beta,  " std = ", std, " val = ",   1.0 / (std**2), " a = " ,self.precision_prior['a'],  "  scale = ", beta_scale
                print "pi_fp = ", pi_fp, " val = ", fp_rate, " a = " ,self.fp_prior['a'],  "  scale = ", fp_scale
                print "pi_fr = ", pi_fr, " val = ", fr, " a = " , self.fr_prior['a'],  "  scale = ", self.fr_prior['b']
        return prior_score
    
    def getNoiseTerm(self, i_C, i_M, i_fr, i_fp_rate ):
        N = i_M - i_C
        noise_term =0.0 
        #False rejection terms
        if np.abs(self.R-i_C) > 1E-12: 
            noise_term =  (self.R-i_C)*np.log(i_fr) 
        if np.abs( i_C ) > 1E-12:
            noise_term += (i_C*np.log(1.0-i_fr))
        #Poisson Process term
        if np.abs(N) > 1E-12:
            noise_term += (N*np.log(i_fp_rate))
        noise_term -= (self.T*i_fp_rate) 
        return noise_term
     
    def getTrueClickIdx(self, i_C, i_M):
        idx_letters = np.sort( np.array(list(itertools.combinations(range(self.R), i_C))), axis=1)
        idx_tp =  np.sort( np.array(list(itertools.combinations(range(i_M), i_C))), axis=1)
        return (idx_letters, idx_tp)
    
    def getNoiseTerms(self, i_M, i_fr, i_fp_rate):
        noise_terms = [self.getNoiseTerm(C, i_M, i_fr, i_fp_rate)     for C in range(0, min(i_M, self.R)+1)]
        return noise_terms 
    
    def getNormalisingConst(self, i_noise_terms, i_click_time_scores, i_params, i_M):
        (delay, std, fr, fp_rate) = i_params 
        z_h = -np.inf
        for C in range(0, min(i_M, self.R)+1): 
            if C < 1:
                z_h = self.utils.elnsum(z_h, i_noise_terms[C] )
            else:
                (idx_letters, idx_tp) = self.getTrueClickIdx(C, i_M)
                for r in range(0,idx_letters.shape[0]):
                    idx = np.array(idx_letters[r, :])
                    true_pos_scores = np.atleast_2d( i_click_time_scores[idx, idx_tp] )
                    click_terms = np.atleast_1d( np.sum(true_pos_scores , axis=1).flatten()) + i_noise_terms[C]
                    z_h = self.utils.elnsum(z_h, self.utils.expTrick(np.atleast_2d(click_terms))[0]) 
        return z_h
    
    #################################### Parameter normalisation
    
    def clipParams(self, i_params):
        (delay, std, fr, fp_rate) = i_params
        fr = np.clip(fr, 1E-6, 1.0) 
        fp_rate = np.clip(fp_rate, 1E-12, 100) 
        std = np.clip(std, 1E-3, 100.0) 
        return (delay, std, fr, fp_rate)
     
    def normaliseParams(self, i_delay, i_cov,  i_gamma_sum):
        #Click-time delay
        delay_offset = self.getDelayOffsetTerm( i_gamma_sum) 
        delay  = self.normaliseGaussMean(i_delay , i_gamma_sum) + delay_offset
        #click-time covariance
        cov_offset = self.getCovOffsetTerm( delay, i_gamma_sum)
        cov = self. normaliseGaussCov( i_cov, i_gamma_sum) + cov_offset
        std = np.sqrt(cov)
        #Noise parameters
        (fr, fp_rate) = self.computeNoiseParams(i_gamma_sum)
        return (delay, std, fr, fp_rate)

    def normaliseGaussParams(self, i_delay, i_cov,  i_gamma_sum):
        delay = self.normaliseGaussMean(  i_delay , i_gamma_sum)
        cov = self.normaliseGaussCov( i_cov, i_gamma_sum)
        return (delay, cov)
    
    def normaliseGaussMean(self,  i_delay , i_gamma_sum):
        return  i_delay /  (self.delay_prior['scale'] + i_gamma_sum)
        
    def normaliseGaussCov( self, i_cov, i_gamma_sum):
        return  i_cov  / (2.0*self.precision_prior['a'] -1.0 + i_gamma_sum) 
    
    def getDelayOffsetTerm(self,   i_gamma_sum):
        delay = self.normaliseGaussMean(self.delay_prior['scale'] * self.delay_prior['mean'], i_gamma_sum)
        return delay
    
    def getCovOffsetTerm(self, i_delay, i_gamma_sum):
        cov  = (self.delay_prior['mean']**2)*self.delay_prior['scale']
        const = (2.0*self.precision_prior['a'] -1.0 + i_gamma_sum) 
        prior = 2.0*self.precision_prior['b'] +  (self.delay_prior['mean']**2)*self.delay_prior['scale']
        prior /= const
        cov +=  2.0*self.precision_prior['b']
        cov -= ( (i_delay**2) * (self.delay_prior['scale']  + i_gamma_sum))
        cov = self.normaliseGaussCov( cov, i_gamma_sum)
        return cov
    
    def computeNoiseParams(self, i_gamma_sum):
        #Compute false positive and negative params based on softly assigned number of true positives
        #false rejection prob
        fr  =  self.R*self.H + self.fr_prior['a'] - 1.0 - i_gamma_sum
        fr_normaliser = self.R*self.H + self.fr_prior['a'] + self.fr_prior['b'] - 2.0
        fr /= fr_normaliser  
        #false positive rate
        fp_rate  =  self.fp_prior['a'] - 1.0 + self.M_sum - i_gamma_sum
        fp_normaliser = self.fp_prior['b'] + self.T*self.H
        fp_rate /= fp_normaliser
        return (fr, fp_rate)
   
    ##################################### Display    

        

class ClickDistrKernelDensityTrainining(ClickDistributionTraining):
    def __init__(self ):
        ClickDistributionTraining.__init__(self)
        self.min_weight = 1E-5 #Prune weights smaller than this value 
       
    def initTraining(self, i_new_letters, i_observations, i_R, i_T):
        self.R = i_R  #Number of times the alphabet has been repeated
        self.T = i_T  #Duration of alphabet repetition in seconds (for false pos calculations)  
        self.obs = list(i_observations[-i_new_letters:]) #deep copy of new data
        self.H = float(len(self.obs))
        self.M_sum = 0.0
        for h in range(0, len(self.obs)):
            data = np.array(self.obs[h])
            M = data.shape[1]
            if  (M - i_R)  > self.max_fp:
                print "Warning:: Maximum number of false positives received : Note processing ", M-i_R
                continue
            self.M_sum += M
        
    def update(self, i_histogram, i_params, i_learn_fp_rate): 
        (weights, true_pos, log_score) = self.updateWeights(i_histogram, i_params)
        """* Weights were just used to filter out really bad outliers (assuming fp_rate is non-zero)
             and other repetions of the same letter. 
           * If fp_rate should not be learned, it means that all the identified clicks are true positives
           * Set weights then to 1.0""" 
        if not i_learn_fp_rate:
            weights = np.ones(weights.shape)
        gamma_sum = np.sum(weights)
        delay = np.sum(weights*true_pos)
        cov = np.sum(weights*(true_pos**2))
        (delay, std, fr, fp_rate) = self.normaliseParams(delay, cov,  gamma_sum) 
        return (weights, true_pos, delay, std, fr, fp_rate, log_score, gamma_sum )
     
    def updateWeights(self, i_histogram, i_params):
        (delay, std, fr, fp_rate) = i_params 
        f = i_histogram.likelihood
        score_params = None
        distr_params = self.clipParams(i_params)
        (z, weights) = self.logLikelihood((0.0, None, fr, fp_rate), f, score_params, i_save_weights=True)
        (data, o_weights, gamma_sum) = ([],[], 0.0)
        
        print "In click distr train: input data = "
        for dd in self.obs:
            self.utils.printMatrix(dd)
    
    
        for h in range(0, int(self.H)):
            M = self.obs[h].shape[1] 
            if  (M - self.R)  > self.max_fp:
                continue
            (weights_h, gamma_h, cnt_weights) = (np.zeros(self.obs[h].shape), 0.0, 0) 
            for C in range(1, min(M, self.R)+1): 
                (idx_letters, idx_tp) = self.getTrueClickIdx(C, M)
                for r in range(0,idx_letters.shape[0]):
                    idx = np.array(idx_letters[r, :])
                    w = weights[h][cnt_weights].flatten()
                    for k in range(idx_tp.shape[0]):
                        weights_h[idx, idx_tp[k,:]] +=  w[k]
                    cnt_weights += 1
            #self.utils.printMatrix(self.obs[h], i_var_name="h=%d, obs"% (h))
            #self.utils.printMatrix(weights_h, i_var_name="h=%d, weights"% (h))
            #obs_s = self.utils.stringVector(obs)
            #print "h=%d, M=%d, n_pos=%.3f, obs=%s, N_obs=%d" % (h,M, gamma_sum, obs_s, len(data))
            (rows , cols) = np.nonzero( weights_h  > self.min_weight )
            if len(rows) < 1:
                continue
            obs =  np.array(self.obs[h][rows,cols]).flatten()
            data.extend(list(obs))
            o_weights.extend(list(weights_h[rows,cols].flatten()))
        (o_weights, data) = (np.array(o_weights), np.array(data))
        return (o_weights, data, z)
   
class ClickDistributionOnlineTraining(ClickDistributionTraining):
    """* Online training class that avoids storing a large number of weights in memory
       * Iterates twice through the data
       * Results should be the same as batch training"""
    
    def __init__(self ):
        ClickDistributionTraining.__init__(self)
    
    def update(self, i_params):
        (delay, std, fr, fp_rate) = i_params 
        (delta, cov, gamma_sum) = (0.0, 0.0, 0.0) 
        for h in range(0, int(self.H)):
            M  =  self.obs[h].shape[1] 
            if  (M - self.R)  > self.max_fp:
                continue
            click_time_scores = self.utils.likelihoodGauss(self.obs[h]-delay, std, i_log=True) 
            noise_terms = self.getNoiseTerms( M,  fr,  fp_rate)
            z_h = self.getNormalisingConst(noise_terms, click_time_scores, i_params, M)
            for C in range(1, min(M, self.R)+1): 
                (idx_letters, idx_tp) = self.getTrueClickIdx(C, M)
                for r in range(0,idx_letters.shape[0]):
                    idx = np.array(idx_letters[r, :])
                    tp_scores = np.atleast_2d( click_time_scores[idx, idx_tp] )
                    weights = np.exp( np.atleast_1d( np.sum(tp_scores , axis=1).flatten()) + noise_terms[C] - z_h )
                    obs = np.array( self.obs[h][idx, idx_tp] )
                    true_pos_times = np.sum( obs, axis= 1).flatten()
                    true_pos_times_sqr = np.sum( obs**2, axis= 1).flatten()
                    delta +=  ( np.sum(weights*true_pos_times))
                    cov += ( np.sum(weights*true_pos_times_sqr))
                    gamma_sum += (np.sum(weights)*C)
        new_params = self.normaliseParams( delta, cov , gamma_sum)
        (delay_new, std_new, fr_new, fp_rate_new) = new_params
        new_params = (delay_new, std_new, fr_new  , fp_rate_new )
    
        log_prior_score = self.logPriorScore(new_params)
        (log_score, empty_weights) = self.logLikelihood(new_params, self.utils.likelihoodGauss, std_new, False)
        return (new_params,   log_score + log_prior_score)

class ClickDistributionBatchTraining(ClickDistributionTraining):
    """* This class is for diagnostic purposes
       * Batch update of click distribution parameters
       * It stores all the weights for all possible latent variable explanations
         and then update the Gaussian, False Postive and Fale Negative distributions."""
    
    def __init__(self ):
        ClickDistributionTraining.__init__(self)

    ###################################### Main
    
    def update(self,  i_old_params):
        weights = self.updateWeights(i_old_params)
        new_params  =  self.clipParams(self.updateParams(weights))
        log_prior_score = self.logPriorScore(new_params)
        weights = self.updateWeights(new_params)
        (delay, std, fr, fp_rate) = new_params
        f = self.utils.likelihoodGauss
        (log_score, empty_weights) = self.logLikelihood(new_params, f, std, False)
        return (new_params, log_prior_score + log_score)
    
    def updateWeights(self, i_params):
        (delay, std, fr, fp_rate) = i_params
        f = self.utils.likelihoodGauss
        (z, weights) = self.logLikelihood(i_params, f, std, True)
        return weights

    def updateParams(self, i_weights):
        (delay, cov, gamma_sum) = (0.0, 0.0, 0.0)
        for h in range(0, int(self.H)):
            M = self.obs[h].shape[1] 
            if  (M - self.R)  > self.max_fp:
                continue
            cnt_weights = 0
            for C in range(1, min(M, self.R)+1): 
                (idx_letters, idx_tp) = self.getTrueClickIdx(C, M)
                for r in range(0,idx_letters.shape[0]):
                    idx = np.array(idx_letters[r, :])
                    w = i_weights[h][cnt_weights].flatten()
                    obs = np.array( self.obs[h][idx, idx_tp] )
                    true_pos_times = np.sum( obs, axis= 1).flatten()
                    true_pos_times_sqr = np.sum( obs**2, axis= 1).flatten()
                    w_s =  self.utils.stringVector(w)
                    t_s = self.utils.stringVector(true_pos_times)
                    #print" h=%d, M=%d, C=%d, r=%d, t=%s, w=%s" % (h,M,C, r,t_s,w_s)
                    delay +=  ( np.sum(w*true_pos_times))
                    cov += ( np.sum(w*true_pos_times_sqr))
                    gamma_sum += (np.sum(w)*C)
                    cnt_weights += 1
        (delay, std, fr, fp_rate) = self.normaliseParams( delay, cov , gamma_sum)
        return (delay, std, fr, fp_rate)

class ApproximateMapTraining(ClickDistributionTraining):
    """ *This class is currently only for diagnostic purposes
        *It's makes an approximation to the EM solution by imposing a hard rule to decide 
         which of the clicks are true positives. 
        *Instead of assigning soft weights to all possible latent variables, all clicks < R
         that lie within 3 standard deviations of the selected letter means, are chosen as true clicks.
        *In good conditions (low fr and fp) this should give similar results to EM solution.
        *For fr=0, fp_rate=0, the solution should be exact to EM solution.
        *For fp_rate=0 this solution should be very similar to EM.
        *Training can perhaps be initialised with this solution.
        *gamma_sum = probabilistic true click sum
        *Withou priors ML and MAP hack should be the same"""
    def __init__(self):
        ClickDistributionTraining.__init__(self)
        
    def update(self, i_params):
        new_params = self.updateML(i_params)
        new_params = self.updateMAP(i_params)
        
    def getTrueClicks(self, i_obs, i_std, i_n_std):
        obs =  np.atleast_2d(i_obs)
        min_idx =  np.argmin( np.abs(obs), axis=1) 
        min_vals =  obs[range(0,obs.shape[0]), min_idx].flatten()
        idx = np.nonzero(np.abs(min_vals) < i_std*i_n_std)[0]
        C = len(idx) #The number of true clicks
        if self.disp:
            min_str = self.utils.stringVector(min_vals)
            print "obs = "
            self.utils.printMatrix(obs, i_precision="%.4f")
            print " "
            print " min_idx = ",self.utils.stringVector(min_idx,i_type="%d"), " min_vals = ", min_str
        true_click_times = min_vals[idx]
        return (C,true_click_times)
        
    def updateMAP(self, i_params):
        (delay, std, fr, fp_rate) = i_params
        (gamma_sum, delay, cov, n_std) = (0.0, 0.0, 0.0, 6)
        for h in range(0, int(self.H)):
            M =  self.obs[h].shape[1]
            (C, true_click_times) = self.getTrueClicks(self.obs[h], std, n_std)
            if C  < 1:
                continue
            gamma_sum += C
            delay += np.sum(true_click_times)
            cov += np.sum(true_click_times**2)
        new_params = self.normaliseParams( delay, cov , gamma_sum)
        if self.ignore_priors:
            print_heading = "MAP HACK NO PRIORS  :"
        else:
            print_heading = "MAP HACK WITH PRIORS:"
        self.utils.printParams(self.utils.getPrintParams(new_params, self.T), i_heading_str=print_heading)
        return new_params
    
    def updateML(self, i_params):
        (m, n_std, tp, fp_rate) = ([], 6, 0.0, 0.0)
        (delay, std, fr, fp_rate) = i_params
        max_tp = self.R*self.H
        for h in range(0, int(self.H)):
            M =  self.obs[h].shape[1]
            (C, true_click_times) = self.getTrueClicks(self.obs[h], std, n_std)
            if C  < 1:
                fp_rate += M
                continue
            tp += C 
            fp_rate += (M - C)
            m.extend(list(true_click_times))
        m = np.array(m)
        delay = np.mean(m)
        std  = np.std(m)
        fr = 1.0 - (tp/max_tp)
        fp_rate /= (self.T*self.H)
        new_params = (delay, std, fr, fp_rate) 
        self.utils.printParams( self.utils.getPrintParams(new_params, self.T), i_heading_str="ML HACK             :")
        return new_params
 


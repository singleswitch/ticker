

import numpy as np
import sys, time 
from utils import Utils
from click_distr_train import ClickDistributionOnlineTraining, ClickDistrKernelDensityTrainining
#Debug files
from click_distr_train import  ClickDistributionBatchTraining, ApproximateMapTraining 
import scipy.stats.distributions as sd
import pylab as p
from kernel_density import NonParametricPdf
import copy

class ClickDistribution():
    ################################################### Init
    """This class is the whole click distribution - a Gaussian distribution is assumed here"""
    def __init__( self ): 
        self.utils = Utils()   
        #Initialise the distribution parameters
        self.initParams()
        #Clear all when reinit params
        self.clear() 
        #Only set the histogram to None here
        self.clearHistogram()
       
    def initParams(self):
        """The priors:
           * Gaussian prior over delay
           * Gamma prior over precision of the delay   
           * Estimation is made over all the data points - kernel parameters inferred from this
           * It is assumed that the letter mean times have already been subtracted from the input times""" 
        #The first level distribution parameters
        self.delay = -0.2        
        self.std = 0.1
        self.nstd = 5.0         #System delay after last letter has been read self.nstd*self.std
        self.fp_rate  = 0.01    #Number of false positives per second
        self.fr  = 0.2          #False rejection probability 
        self.min_val = 1E-10    #Clip the value of the pdf
        self.allow_param_clip = True
        (self.fp_rate_min, self.fr_min, self.std_min) = (0.001, 0.01, 0.03)
        self.learning_rate = 0.3
        self.extra_wait_time = 0.0 #Extra time to wait at end of last letter
        #Training parameters
        self.max_samples = 1000 #Maximum number of letters to store clicks for
        self.min_train = 3      #Minimum number of letters before training is allowed
        self.is_train = True
        self.noise_simple = False
        self.train_gauss=False
        #Learn specific parameters
        (self.learn_delay, self.learn_std, self.learn_fp, self.learn_fr) = (True, True, True, True)
        #Debug
        self.debug = False     #Run diagnostic tests
        self.disp = True       #Display output
        self.draw = False      #Draw pictures of the click distr
        
        #FIXME
        self.train_gauss = False
        
    def initHistogram(self):
        self.histogram = NonParametricPdf(self.learning_rate, self.max_samples) 
        self.histogram.reset(self.delay, self.std)
        self.__drawHistogram()
       
    def reset(self, i_channel_config):
        """* Compuate the times associated with each letter in the alphabet.
           * Each letter is repeated L times, and there are A letters in the alphabet. 
           * self.alphabet: Vector of length A (containing all the lettters in the alphabet
           * self.loc: - The location of the distribtion associated with each reading.
                       - Matrix of size AxL. """
        
                    
        self.alphabet = i_channel_config.getAlphabetLoader().getUniqueAlphabet( i_with_spaces=False)
        self.click_times = i_channel_config.getClickTimes()
        self.letter_idx = i_channel_config.getAlphabetLoader().getLetterPositions() 
        self.resetLetterLocations(i_channel_config)
        self.clear()
    
    def resetLetterLocations(self, i_channel_config):
        """ * Result the letter offsets - this function should be called whenever
              any of the click distribution parameters changes 
            * Boundary delay is there just to allow the distribution to fit in the time the audio
              is represented to the user.
            * self.loc contains the locations of the letters positive time, where the audio should be played
            * The gaussian delay is not added (function of model, stored as a separate parameter)"""
        self.loc = np.array( self.click_times[self.letter_idx] )
        sound_times =  i_channel_config.getSoundTimes() 
        self.T =  sound_times[-1,-1] + self.extra_wait_time #Add extra time allowed for delay to wait for extra clicks
   
    def clear(self):
        self.train_obs = []
        self.obs_letters = []     
        
    def clearHistogram(self):
        self.histogram = None
               
    ############################### Main

    def logLikelihood(self, i_observations, i_log=False):
        """ * It is assumed that the user runs Ticker up L readings of the alphabet.
            * It is assumed that each letter has the same distribution, but with a different 
              offset
            * i_observations: A vector with M timings
            * i_channel_config: Contains the letter offsets"""
        t = time.time()
        A = self.loc.shape[0]        #The unique number of letters in the alphabet
        o_scores = np.zeros(A)
        for letter_num in range(0, A):
            o_scores[letter_num] = self.logLikelihoodLetterNum(letter_num, i_observations, i_log)
        return o_scores
    
    def logLikelihoodLetter(self, i_letter, i_observations, i_log):
        alphabet = np.array(self.alphabet)
        idx = np.nonzero(alphabet == i_letter)[0]
        letter_num = alphabet[idx]
        return self.logLikelihoodLetterNum(letter_num, i_observations, i_log)
 
    def logLikelihoodLetterNum(self, i_letter_num, i_observations, i_log):
        #MxL matrix
        obs = np.atleast_2d(i_observations).transpose()
        if self.histogram is None:
            test_data = obs-self.delay-self.loc[i_letter_num,:] 
            click_time_scores = self.utils.likelihoodGauss(test_data, self.std, i_log=True)
        else:
            test_data = obs-self.loc[i_letter_num,:]  
            click_time_scores = self.histogram.likelihood(test_data, i_log=True)
        click_time_scores = np.clip(click_time_scores, np.log(self.min_val), np.max(click_time_scores))
        if self.noise_simple:
           o_score  = self.updateNoiseSimple(i_letter_num, test_data , click_time_scores, i_display_scores=False)
           if not i_log:
                o_score = np.exp(o_score)
        else:
            ZC = self.labelProbs(obs.shape[0])   #Normalisation factor per number of true clicks hypothesis  
            o_score = self.sumLabels(np.exp(click_time_scores), ZC )
            if o_score < self.min_val:
                o_score = self.min_val
            if i_log:
                o_score = np.log(o_score)
        return o_score
    
    ################################# Training

    def storeObservation(self, i_observations):
        self.train_obs.append( np.atleast_2d( np.array(i_observations )) )
       
    def train(self, i_words_written):
        if not self.is_train:
            print "IS TRAIN IS FALSE"
            return
        #Make a copy of the training data before changing the data shape 
        (observations, new_letters) = self.__updateTrainingData(i_words_written)
        if observations is None:
            return
        if self.train_gauss:
            self.histogram = None
            self.trainGaussian(observations)
        else:
            self.trainHistogram(observations, new_letters)
        #Resture the training data
        self.train_obs = list(observations)
          
    def trainHistogram(self,i_observations, i_new_letters): 
        if self.disp:
            print "********************************************************"
            print "Kernel density training" 
            print "********************************************************"
        #Init histogram if it has not been initialised yet
        if self.histogram is None:
            self.initHistogram()
        R = self.loc.shape[1] #The number of times the alphabet is repeated
        #Init trainer
        trainer = ClickDistrKernelDensityTrainining()
        trainer.initTraining(i_new_letters, self.train_obs, R, self.T)
        #Init params
        params = (self.delay, self.std, self.fr, self.fp_rate) 
        histogram = copy.deepcopy(self.histogram)
        (old_params, old_histogram) = (tuple(params), copy.deepcopy(self.histogram))
        prev_score = -np.inf
        t=time.time()        
        if self.disp:
            self.utils.printParams(self.utils.getPrintParams(params, trainer.T), " learning rate=%.3f Init Params=:" % self.learning_rate) 
        for k in range(1, trainer.max_iterations):
            params = (self.delay, self.std, self.fr, self.fp_rate)
            (weights, true_pos, delay, std, fr, fp_rate, log_score, N_tp) = trainer.update(histogram, params, self.learn_fp) 
            new_params = (delay, std, fr, fp_rate )
            #Save the new parameters with the learning rates
            #Update all parameteric distribution values
            self.updateGaussianParams(old_params, new_params)
            #The non-parameteric distribution
            histogram = copy.deepcopy(self.histogram)
            if self.learn_std:
                histogram.setStd(std, N_tp) 
            histogram.saveDataPoints(np.array(true_pos), i_weights=weights)
            dist = log_score
            if not np.isinf(prev_score):
                dist -= prev_score
            prev_score = log_score
            if self.disp: 
                disp_params = self.utils.getPrintParams((delay, std, fr, fp_rate), trainer.T)
                disp_heading =  " k = %d, Score=%.4f, dist=%.4f, N_tp=%2.4f, New Params=:" % (k,log_score, dist, N_tp)
                self.utils.printParams(disp_params, disp_heading)
            if (dist <  trainer.eps_dist) and (k > 1):
                break
        if self.disp:
            print "Training time = ", 1000.0*(time.time() - t), "ms"
        self.histogram = copy.deepcopy(histogram)
        if self.disp: 
            disp_params = self.utils.getPrintParams((delay, std, fr, fp_rate), trainer.T)
            self.utils.printParams(disp_params, "Final params: " )
        self.__drawHistogram()
        
    def trainGaussian(self, i_observations):
        R = self.loc.shape[1] #The number of times the alphabet is repeated
        old_params = (self.delay, self.std, self.fr, self.fp_rate)
        #Diagnostic training
        words_written = "".join(self.obs_letters)
        self.trainDiagnostic(R, old_params, i_observations, words_written)
        #Train the Gaussian Parameters
        old_params = (self.delay, self.std, self.fr, self.fp_rate)
        self.__trainGaussian(R, old_params, i_observations, words_written)
        #Online updating
        new_params = (self.delay, self.std, self.fr, self.fp_rate)
        self.updateGaussianParams(old_params, new_params)
        
    def updateGaussianParams(self, i_old_params, i_new_params ):
        (old_delay, old_std, old_fr, old_fp_rate) = i_old_params
        (new_delay, new_std, new_fr, new_fp_rate) = i_new_params
        if self.learn_delay:
            self.delay = (1.0 - self.learning_rate)*old_delay + self.learning_rate*new_delay
        else:
            self.delay = old_delay
        if self.learn_std:
            self.std = (1.0 - self.learning_rate)*old_std + self.learning_rate*new_std
        else:
            self.std = old_std
        if self.learn_fr:
            self.fr = (1.0 - self.learning_rate)*old_fr + self.learning_rate*new_fr
        else:
            self.fr = old_fr
        if self.learn_fp:
            self.fp_rate = (1.0 - self.learning_rate)*old_fp_rate + self.learning_rate*new_fp_rate
        else:
            self.fp_rate = old_fp_rate
            
    def __trainGaussian(self, i_R, i_params, i_observations,   i_words_written): 
        if self.disp:
            print "********************************************************"
            print "E-M training online with priors: MAP estimate"
            print "words = ",  i_words_written 
            print "********************************************************"
        trainer = ClickDistributionOnlineTraining()
        params = tuple(i_params)
        prev_score = trainer.initTraining(self.train_obs, i_R, self.T, params)
        for k in range(1, trainer.max_iterations):
            if self.disp:
                disp_params = self.utils.getPrintParams(params, trainer.T)
                self.utils.printParams(disp_params, " k = %d, Old Params Online :" %k ) 
            t = time.time()
            (params,  score) =  trainer.update( params )
            dist = score - prev_score
            t_tot = 1000.0*(time.time() -  t) 
            if self.disp: 
                self.utils.printParams(self.utils.getPrintParams(params, trainer.T), " k = %d, New Params Online:" %k ) 
                print " k = %d, prev_score = %.5f, score = %.5f,  dist = %.5f, time=%.2fms" % (k, prev_score, score,  dist,t_tot )
            prev_score = score
            if dist < -1E-3:
                raise ValueError("Score did not decrease!")
            if dist <  trainer.eps_dist:
                break
        (self.delay, self.std, self.fr, self.fp_rate) = params
        if self.allow_param_clip:
            self.fr = max(self.fr_min, self.fr)
            self.fp_rate = max(self.fp_rate_min, self.fp_rate)
            self.std = max(self.std_min, self.std) 
            
    def __updateTrainingData(self, i_words_written ):
        """* First extract the original bounday delay 
               * This delay is a const not function of user
               * The audio has been played to the user with the original boundary delay
               * Substract the letter offset, so that only the delay is left
               * The training data is the same for all iterations"""
        #Update the observed letters
        new_letters = len(i_words_written)
        for letter in i_words_written:
            self.obs_letters.append(letter) 
        if len(self.obs_letters) > self.max_samples:
            self.train_obs = self.train_obs[-self.max_samples:]
            self.obs_letters = self.obs_letters[-self.max_samples:]
        observations = list(self.train_obs)
        
        
        self.disp = True
        
        
        for (idx, letter) in enumerate(self.obs_letters):
            letter_loc = self.getLetterLocation( letter, i_with_delay=False) 
            if self.disp:
                print "training: letter = = ", letter, " obs = ",  self.train_obs[idx] 
            self.train_obs[idx] = (self.train_obs[idx].transpose() - letter_loc).transpose()
        return (observations, new_letters) 
    
    def __drawHistogram(self):
        if self.draw:
            p.figure()
            print "hist std = ", self.histogram.kernel_std
            print "My std = ", self.std
            self.histogram.draw( i_color="r", i_histogram=True )
            x_eval = np.linspace( -10*self.std, 10*self.std, 200) + self.delay
            y_eval = sd.norm.pdf(x_eval, loc=self.delay, scale=self.std)
            p.plot(x_eval, y_eval, 'k') 
            
    ############################## Get
    
    def getParams(self):
        return (self.delay, self.std, self.fr, self.fp_rate)
    
    def getAllLetterLocations(self, i_with_delay):
        delay = 0.0 
        if i_with_delay:
            delay = self.delay
        return self.loc + self.delay
    
    def getLetterLocation(self, i_letter, i_with_delay):
        delay = 0.0
        idx = np.nonzero( np.array(self.alphabet) == i_letter)[0]
        if i_with_delay:
            delay = self.delay
        return self.loc[idx,:].flatten() + delay
   
    def getHistogramRects(self):
        if self.histogram is None:
            raise ValueError("No histogram!")
        return self.histogram.getHistogramRects()

    ############################### Set
    
    def setGaussParams(self, i_channel_config, i_delay, i_std):
        self.std = i_std
        self.delay = i_delay
        self.reset(i_channel_config)
    
    def setClickDev(self, i_channel_config, i_std):
        self.std = i_std
        self.reset(i_channel_config)
        
    def setClickDelay(self, i_channel_config, i_delay):
        self.delay = i_delay
        self.reset(i_channel_config)
    
    def setParams(self, i_is_train, i_channel_config, i_delay, i_std, i_fp_rate, i_fr, i_learning_rate, i_extra_wait_time):
        self.delay = i_delay
        self.std = i_std
        self.fp_rate = i_fp_rate
        self.fr = i_fr
        self.learning_rate = i_learning_rate
        self.is_train  = i_is_train
        self.extra_wait_time =  i_extra_wait_time
        self.reset(i_channel_config)
        
    def setFixLearning(self,  i_learn_delay, i_learn_std, i_learn_fp, i_learn_fr):
        (self.learn_delay, self.learn_std) = ( i_learn_delay, i_learn_std )
        (self.learn_fp, self.learn_fr) = (i_learn_fp, i_learn_fr)

    
    ############################## P(click_time | letter)
    
    def labelProbs(self, i_M):
        """Return the normalisers per true click hypothesis: C=0, C=1, ... CK, 
           where K = min(M, L), M is the number of observations, L is the number of 
           opportunities the user had to select a letter (number of alphabet repetitions).
           i_M: The number of observations"""
        L = self.loc.shape[1]        #The maximum number of true clicks
        C_range =  np.arange(0.0,  min(i_M, L) + 1 ) 
        N_range = i_M - C_range 
        fr_range = L - C_range
        Z = np.exp( -(self.fp_rate* self.T)) * (self.fp_rate**N_range) * ( self.fr**fr_range  ) * (  (1.0 - self.fr)**C_range ) 
        #print "FR = ", self.fr, " FP = " , self.fp_rate, " C_range = ", C_range, " N_range = ",  N_range, " fr_range = ", fr_range, " Z = ", Z
        return Z

    def sumLabels(self, i_f, ZC ):
        """Sum over possible instances of the same letter than could have been responsible 
           for a specific letter, and all possible false positive/ true positive labellings, 
           given than i_C clicks have been observed.
           * i_f: MxL likelihoods:
           * L: Supports letter labelling (L1, L2, .....)
           * i_f:  [p(t1 | l1)  p(t1 | l2) ... p(t1 | lL)
                       :
                    p(tM | l1   ...            p(tM | lL)]"""
        if not(np.ndim(i_f)) == 2:
            raise ValueError("Dimension should be 2, but is " + str(np.ndim(i_f)) + " instead!")
        #The scores for zero true clicks
        click_scores = np.ones(len(ZC))
        #Compute the scores C=1, no products involved
        f_sum = self.updateCumSum(i_f) #Update scores for one click
        click_scores[1] = np.float64(f_sum[0,0])
        for C in range(2, len(ZC)):
            f_new = np.atleast_2d(i_f[0:-(C-1),0:-(C-1)])*np.atleast_2d(f_sum[1:,1:])
            f_sum = self.updateCumSum(f_new)
            click_scores[C] = np.float64(f_sum[0,0])
        click_scores *= ZC
        return np.sum(click_scores)
    
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
    
    ############################################# Debug
    
    def trainDiagnostic(self, i_R, i_init_params, i_observations,  i_words_written): 
        if not self.debug:
            return
        self.trainDiagnosticMapApproximate(i_R, i_init_params)
        #Batch training
        trainer = ClickDistributionBatchTraining()
        trainer.disp = False
        print "********************************************************"
        print "E-M training no priors: ML estimate"
        print "********************************************************"
        trainer.ignore_priors = True
        self.trainDiagnosticBatchEM(trainer, i_R, i_init_params, i_observations,  i_words_written)
        print "********************************************************"
        print "E-M training with priors: MAP estimate"
        print "********************************************************"
        trainer.ignore_priors = False
        (self.delay, self.std, self.fr, self.fp_rate) = i_init_params
        self.trainDiagnosticBatchEM(trainer, i_R, i_init_params, i_observations,  i_words_written)
        (self.delay, self.std, self.fr, self.fp_rate) = i_init_params
        
    def trainDiagnosticBatchEM(self, i_trainer, i_R, i_params, i_observations,  i_words_written):  
        params = tuple(i_params)
        i_trainer.initParams()
        prev_score = i_trainer.initTraining(self.train_obs, i_R, self.T, params)
        prior_score = i_trainer.logPriorScore( params)
        test_score = self.__trainingScore( i_observations,  i_words_written, params,  prior_score )
        print "Initialise: batch score = ", prev_score, " prior score = ", prior_score, " test_score = ", test_score    
        for k in range(1, i_trainer.max_iterations):
            print "----------------------------------------------------------------------"
            t = time.time()
            self.utils.printParams(self.utils.getPrintParams(params, i_trainer.T), " k = %d, Old Params Batch :" %k) 
            (params, score) = i_trainer.update( params )
            prior_score = i_trainer.logPriorScore( params)
            (self.delay, self.std, self.fr, self.fp_rate) = params
            t_tot = 1000.0*(time.time() -  t) 
            test_score = self.__trainingScore( i_observations,  i_words_written, params,  prior_score )
            #Compare against score that doesn't compute over all possible enumerations
            dist = score - prev_score
            self.utils.printParams(self.utils.getPrintParams(params, i_trainer.T), " k = %d, New Params Batch :" %k) 
            print "prev_score = %.5f, score = %.5f,  test_score=%.5f, dist = %.5f, time=%.2fms" % (prev_score, score,  test_score, dist, t_tot)
            prev_score = score
            if dist < -1E-3:
                raise ValueError("Score did not decrease!")
            if np.abs(score - test_score) > 1E-3:
                raise ValueError("Current score and internal score computation not the same!")
            if dist <  i_trainer.eps_dist:
                break
        print "----------------------------------------------------------------------"
  
    def trainDiagnosticMapApproximate(self, i_R, i_params):
        #Train with hack to determine which clicks are true positive
        print "----------------------------------------------------------------------"
        trainer = ApproximateMapTraining()
        trainer.disp = False 
        trainer.ignore_priors = True
        trainer.initTraining(self.train_obs, i_R, self.T, i_params)
        (approximate_params) =trainer.update(i_params)
        trainer.ignore_priors = False
        trainer.initParams()
        trainer.initTraining(self.train_obs, i_R, self.T, i_params)
        (approximate_params) = trainer.update(i_params)  
        print "----------------------------------------------------------------------"
    
    def __trainingScore(self, i_observations,  i_words_written, i_params, i_prior_score ):
        (self.delay, self.std, self.fr, self.fp_rate) = i_params
        log_score = 0.0 
        for h in range(0, len(i_observations)):
            letter_num = np.nonzero( np.array(self.alphabet) == i_words_written[h] )[0]
            scores = np.log(self.logLikelihood(np.atleast_2d(i_observations[h]), i_log=False) )
            log_score += scores[letter_num]
        return log_score + i_prior_score
    
    def updateNoiseSimple(self, i_letter_num, i_test_data , i_click_time_scores, i_display_scores=False):
        marginal_letters = self.utils.expTrick(i_click_time_scores)
        if i_display_scores:
            print_score_vec = self.utils.stringVector(i_click_time_scores.flatten())
            print_marg_vec = self.utils.stringVector(marginal_letters)
            test_data_str =  self.utils.stringVector(i_test_data.flatten())
            print "Letter =  ",self.alphabet[i_letter_num],
            print " scores = ", print_score_vec, " marginals = ", marginal_letters, " obs=", test_data_str
        o_score = np.sum(marginal_letters + np.log(np.sqrt(2)))
        if np.isnan(o_score):
            o_score = -np.inf
        min_val = np.log(self.min_val)
        if o_score < min_val:
           o_score = min_val
        return o_score
    
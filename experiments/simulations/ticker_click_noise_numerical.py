
import pylab as p
import numpy as np
import sys, time
sys.path.append("../../")
sys.path.append('../')
from channel_config import  ChannelConfig
#Contour refinement functions
import matplotlib._cntr as cntr
import numpy.ma as ma
from scipy.optimize import fsolve
import scipy.optimize.lbfgsb as solver
from utils import Utils, WordDict, PlotUtils
from scipy.integrate import dblquad, quad
from pylab_settings import DispSettings

#Fast scores
#import pyximport; pyximport.install()
#from ticker_scores import TickerScoresFast 

#Diagnostic global variables: Remove
global is_fast
is_fast =  True

"""To do:
1. Remove display functions from channel_config, all in here? """

class TickerClickNoise2D():
    def __init__(self):
        self.eps = 1E-20
        self.colors = {'white':'w', 'black':'k', 'red':'r', 'blue':'b'}
        self.disp = DispSettings()
        self.recovery_time = 0.0
        self.utils = Utils()
        self.dict = WordDict("../../dictionaries/nomon_dict.txt")
        self.plt_utils = PlotUtils()
        
    ################################################# MAIN 
    def compute(self):
        channel_config = ChannelConfig(i_nchannels=5, i_sound_overlap=0.1, i_root_dir="../../")
        gauss_std=0.1
        i_prob_thresh = 0.95
        display = True
        is_plot = False
        display_top_words = False
        target_word = "is_"
        n_repetitions = 2
        optimised = True
        fr = 0.2
        print "dict length = ", len( self.dict.words )
  
        #Load the alphabet and click times
        alphabet_loader =  channel_config.getAlphabetLoader()
        alphabet = np.array(alphabet_loader.getAlphabet( i_with_spaces=False))
        alphabet_unique = alphabet_loader.getUniqueAlphabet( i_with_spaces=False)
        click_times =  channel_config.getClickTimes()
        idx = alphabet_loader.getLetterPositions()
        click_times_2D  = np.array(click_times[idx])
        n_samples = 100     
        n_std = 3.0

        #Store the alphabet positions in dictionary
        letter_indices = {}
        for n in range(0, len(alphabet_unique)):
           letter_indices[alphabet_unique[n]] = n
        #Store all the word positions in dict
        word_indices = {}
        for n in range(0, len(self.dict.words)):
            word_indices[self.dict.words[n]] = n
            
        target_word_idx = word_indices[target_word]
        gauss_var = gauss_std**2
        word_priors =  np.exp(np.array(self.dict.log_probs)) 
        
        if optimised:
            #Memory assignments
            target_scores = np.zeros([n_samples, 1], dtype=np.double)
            probs = np.zeros([n_samples, len(self.dict.words)], dtype=np.double)
            letter_scores = np.zeros([n_samples, click_times_2D.shape[0] ], dtype=np.double)
            short_tmp_letter_scores = np.zeros( click_times_2D.shape[0], dtype=np.double)
            long_tmp_letter_scores = np.zeros( len(self.dict.words), dtype=np.double)
            normalisers = np.zeros(n_samples , dtype=np.double)
            factor = np.zeros([n_samples, 1], dtype=np.double)
            sf =  TickerScoresFast( gauss_var, i_prob_thresh, n_samples) 
            click_00 = np.array(word_priors)
            click_10 = np.zeros( word_priors.shape, dtype=np.double)
            click_01 = np.zeros( word_priors.shape, dtype=np.double) 
            click_11 = np.zeros( word_priors.shape, dtype=np.double)
             
        for letter_idx in range(0, n_repetitions*len(target_word)):
            print "***************************************************************************************"
            t = time.time()
            t_start = t
            #Map the long letter idx to short one
            short_letter_idx = self.getWordLetterIdx(letter_idx, letter_indices, target_word_idx)
            target_letter =  target_word[short_letter_idx]
            target_letter_idx = letter_indices[target_letter]
            #Get the score for specific input times
            (t1, t2) = (click_times_2D[target_letter_idx,0], click_times_2D[target_letter_idx,1])
            (min_x, max_x)  = (t1 - n_std*gauss_std, t1+n_std*gauss_std)
            (min_y, max_y) =  (t2 - n_std*gauss_std, t2+n_std*gauss_std )
            step_size_x = (max_x - min_x) / np.float(n_samples-1)
            t1  = np.hstack(  (np.arange(min_x, t1, step_size_x), np.arange(t1, max_x, step_size_x ) ) )
            step_size_y = (max_y - min_y) / np.float(n_samples-1)
            t2  = np.hstack(  (np.arange(min_y, t2, step_size_y), np.arange(t2, max_y, step_size_y ) ) )
            letter_list = self.getWordLetterList(letter_idx, letter_indices)
            m1 = click_times_2D[:,0]
            m2 = click_times_2D[:,1]   
            
            classify_probs = np.zeros( len(self.dict.words), dtype=np.double)
            click_00 = np.array(word_priors)
            
            (click_01, classify_probs) = sf.updateWordPriors( np.double(t1),
                np.double(m1), int(target_letter_idx), 
                int(target_word_idx), np.double(click_00) , np.int32(letter_list), False,
                target_scores, probs, letter_scores, short_tmp_letter_scores,
                long_tmp_letter_scores, normalisers, factor, classify_probs )
            (click_10, classify_probs) = sf.updateWordPriors( np.double(t1),
                np.double(m1), int(target_letter_idx), 
                int(target_word_idx),  np.double(click_00) , np.int32(letter_list), False,
                target_scores, probs, letter_scores, short_tmp_letter_scores,
                long_tmp_letter_scores, normalisers, factor, classify_probs )
            (click_11, classify_probs) = sf.updateWordPriors( np.double(t2),
                np.double(m2), int(target_letter_idx), 
                int(target_word_idx),   np.double(click_01) , np.int32(letter_list), False,
                target_scores, probs, letter_scores, short_tmp_letter_scores,
                long_tmp_letter_scores, normalisers, factor, classify_probs )
            word_priors = (fr**2)*click_00 + (1-fr)*fr*click_01 + fr*(1-fr)*click_10 + ( click_11*(1-fr)**2)
      
            if display:
                print target_letter, " : time=", (time.time()-t), "s; word prob = ", word_priors[target_word_idx],
                #print ", classify prob = ", classify_probs[target_word_idx], " sum classify = ", 
                #print np.sum(classify_probs), 
                print "; sum words = ", np.sum(word_priors)
            
            if display_top_words:
                sort =np.flipud(  np.argsort( word_priors )[-10:] )
                top_words = self.dict.words[sort]
                scores = word_priors[sort]
                print "Top words : ",  top_words
                print "Their scores = ", scores
                print "sum of word_priors: ", np.sum(word_priors), " sum of top words : ",  np.sum(scores)

    def probCorrect(self, i_t1, i_t2, i_args):
        t= time.time()
        (click_times_2D,target_letter_idx, gauss_std, target_word_idx, log_word_priors, letter_list, display, prob_thresh) = i_args
        N = click_times_2D.shape[0]
        t = time.time()
        log_letter_scores = np.array( [self.getScores(i_t1, i_t2, click_times_2D[n,:], gauss_std, i_prior=None,  i_clip=False, i_log=True) for n in range(0, N)])
        word_log_probs =  log_letter_scores[letter_list] + log_word_priors
        ay =  word_log_probs[target_word_idx] 
        az = np.log( np.sum( np.exp( word_log_probs)) - np.exp(ay) )
        x = self.utils.elnprod(ay, - az)
        prob_correct = self.classify( prob_thresh, x, i_rate=30.0)
        print "The rest", 1000*(time.time() - t), " ms", " az = ", az, " exp =", np.exp(az),  " ay = ", ay, " exp = ", np.exp(x), " x = ", ay-az, " x = ", x, " prob_correct=  ", prob_correct
        return prob_correct

    def saveFig(self, i_file_name):
        pdf_fname = i_file_name + ".pdf"
        eps_fname = i_file_name + ".eps"
        #pdf_crop_fname = i_file_name + 
        p.savefig(pdf_fname, dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', papertype='a4', format=None,
            transparent=False, bbox_inches="tight", pad_inches=0.05)
        p.savefig(eps_fname, dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', papertype='a4', format=None,
            transparent=False, bbox_inches="tight", pad_inches=0.05)
    
    def letterDecisionBoundaries(self , i_alphabet, i_pos, i_target_index, i_gauss_std, i_prob_thresh, i_priors, i_plot, i_display, i_score_mask, i_T, i_n_std_boundaries ):
        """i_score_mask: Also found the regions where the target score posterior is bigger than the rest"""
        if is_fast:
            n_samples_init_grid = 300         #The number of points used in grid approximation to find region of interest
        else: 
            n_samples_init_grid = 300 
        plotContoursSeparate = False          #Plot All contours separately in a figure, only used if i_plot=True
        plotContours = False                  #Plot the decision boundaries with the click distribution, only used if i_plot=True
        #Get the boundary segments 
        (X, Y, scores, bounds)=self.getGridScores(i_priors, n_samples_init_grid, i_n_std_boundaries, i_target_index, i_pos, i_gauss_std, i_T )
        (posterior_scores, segs, score_masks) = self.__letterDecisionBoundaries( X,  Y, scores, bounds, i_prob_thresh, i_target_index, i_pos, i_gauss_std, i_priors, i_display, i_score_mask, plotContoursSeparate)
        if i_plot:
            #Make the final plot
            letter = i_alphabet[i_target_index]
            c_title = "$P(t_{1}, t_{2}, M=2 \mid  \\boldsymbol{\\theta}, \ell=\mathrm{%s})$" % letter
            self.__plotAllLetterScores( i_alphabet, i_pos,scores, X, Y, c_title, i_target_index=i_target_index)
            p.grid('on', color='white')
            #if plotContours:
            #    self.__plotContourSegs(segs, i_color='k',  i_linewidth=2)
        return segs

    def allLetterDecisionBoundaries(self,  i_alphabet, i_pos,  i_gauss_std, i_priors, i_prob_thresh, i_display, i_plot, i_T, i_n_std ):
        """* Compute the likelihoods associated with each letter
           * Each letter score is normalised by the same value, so firstly we take the maximum score
           * We store the indices associated with the maximum score if the latter score is bigger
             than self.eps.
           * Invalid indices (if score < self.eps) are negative"""
        if is_fast:
            n_samples  = 300         #The number of points used in grid approximation to find region of interest
        else:
            n_samples = 300
        (min_x, max_x, min_y, max_y, X, Y) = self.__getAxis(i_pos[0,:],  n_samples,  i_n_std, i_gauss_std, i_pos.flatten(), i_pos.flatten(),i_T)
        bounds = ([min_x, max_x], [min_y, max_y])
        fig = p.figure(facecolor='w'); p.grid('on')

        self.__plotLetters(i_alphabet, i_pos[:,0], i_pos[:,1], i_delta_x=0.00, i_alpha=0.6, i_color='white', i_marker=None)
        
        mask = np.zeros(X.shape)
        o_segs = []
        (rows,cols) = np.nonzero( Y >= X)
        for n in range(0, i_pos.shape[0]): 
            print "********************************************************"
            print "Extracting contours of letter ", n, " = ", i_alphabet[n]
            print "********************************************************" 
            if i_priors is not None:
                prior = i_priors[n] 
            else:
                prior = None
            scores = np.zeros(X.shape)
            scores[rows,cols]  = self.getScores(X[rows,cols], Y[rows,cols], i_pos[n,:], i_gauss_std, prior,  i_clip=False)
            (posterior_scores, segs, score_masks) = self.__letterDecisionBoundaries( X,  Y, scores, bounds, i_prob_thresh, n, i_pos, i_gauss_std, i_priors, i_display=True, i_score_mask=True, i_plot=False)
            (row, col) = np.nonzero( score_masks > self.eps)
            mask[row, col] = np.array(posterior_scores[row, col])
            #if i_plot:
            #     self.__plotContourSegs(segs, i_color='k' , i_linewidth=2)
            o_segs.append(list(segs))
        c_title = "$P( \ell^{*} \mid  \\boldsymbol{\\theta}, t_{1}, t_{2}, M=2)$"
        self.__dispLetterProbs( mask, X,  Y,  i_color_bar_title=c_title, is_gray=False )
        p.ylabel("$t_{2}$ $\\mathrm{(seconds)}$")
        p.xlabel("$t_{1}$ $\\mathrm{(seconds)}$")
        return o_segs
    
    ##################################### Integral computations
       
         
    def updateSingleWordPrior(self, i_t1, m,target_letter_idx, gauss_var,target_word_idx, word_priors,letter_list):
        """Update the priors if no decision has been made"""
        letter_scores = self.__getScores1D( m, gauss_var, i_t1, i_clip=False, i_log=False) 
        target_letter_score = letter_scores[target_letter_idx]
        word_probs =  letter_scores[letter_list] * word_priors 
        ay =  word_probs[target_word_idx]
        if np.abs(ay) < self.eps:
            return self.eps 
        az =  np.sum( word_probs )
        o_val = target_letter_score * ay / az
        return o_val
    
    def updateNotCorrect2(self, i_t1, i_t2, click_times_2D,target_letter_idx, gauss_std,target_word_idx, log_word_priors,letter_list, display, prob_thresh):
        """Update the priors if no decision has been made"""
        N = click_times_2D.shape[0]
        log_letter_scores = np.array( [self.getScores(i_t1, i_t2, click_times_2D[n,:], gauss_std, i_prior=None,  i_clip=False, i_log=True) for n in range(0, N)])
        word_log_probs =  log_letter_scores[letter_list] + log_word_priors
        ay =  word_log_probs[target_word_idx] 
        az = np.log( np.sum( np.exp( word_log_probs)))
        word_prob = self.utils.elnprod(ay, - az)
        word_prob = self.utils.elnprod(word_prob, log_letter_scores[target_letter_idx])
        #print "t1 =  ", i_t1, " i_t2 = ", i_t2, " prob = ", np.exp(word_prob), " bounds = ", bounds
        return np.exp(word_prob)
    
    def getWordLetterIdx(self, i_letter_idx, i_letter_pos, i_target_idx):
        letter_idx = -i_letter_idx / self.dict.word_lengths[i_target_idx]
        letter_idx *= self.dict.word_lengths[i_target_idx]
        letter_idx += i_letter_idx
        return letter_idx
     
    
    def getWordLetterList(self, i_letter_idx, i_letter_pos ):
        letter_indices = -i_letter_idx / self.dict.word_lengths
        letter_indices *= (self.dict.word_lengths)
        letter_indices += i_letter_idx
        letter_list =  np.array( [i_letter_pos[self.dict.words[n][ idx]]  for n, idx in enumerate( letter_indices, start=0) ] )
        return letter_list

    def updateLogWordPriors(self, i_log_priors, i_letter_idx, i_log_letter_scores, i_letter_pos, i_display ):
        letter_indices = -i_letter_idx / self.dict.word_lengths
        letter_indices *= (self.dict.word_lengths)
        letter_indices += i_letter_idx
        t = time.time()
        word_log_probs =  np.array( [ i_log_letter_scores[i_letter_pos[self.dict.words[n][ idx]]]  for n, idx in enumerate( letter_indices, start=0) ] )
        word_log_probs += i_log_priors
        if i_display:
            print "Time to iterate through the words = ", 1000.0*(time.time() - t), "  ms" 
        return word_log_probs

    def printSigmoid(self):
        T = 0.95
        k = 30.0
        for T2 in np.linspace(T-0.1, np.clip(T+0.1,T, 0.999), 1000):
            x1 = np.log(T2/(1-T2))
            y = 1.0 / (1.0 +np.exp(-k*x1 + k*np.log(T/(1-T))))
            print "T = ", T, " T2 = ", T2, " y = ", y
         
    
    ##################################### Score computations
     
    def getGridScores(self, i_priors, i_n_samples, i_n_std, i_target_index, i_pos, i_gauss_std, i_T ):
        """Extract a grid of size i_n_sampes x i_nsamples, according to the independent Gaussian
        click distribution, respecting boundary conditions. Also compute the costs if the user aims 
        at letter associated with i_target_index (index in unique alphabet). Return the Grid coordinates, 
        scores and axis bounds. Here two clicks are assumed and values where the second click time is 
        bigger than the first are ignored (i_clip = True)."""
        target_mean = i_pos[i_target_index,:]
        #The target score p(ym | click_times)
        (min_x, max_x, min_y, max_y, X, Y) = self.__getAxis(target_mean, i_n_samples, i_n_std, i_gauss_std, i_pos.flatten(), i_pos.flatten(), i_T)
        if i_priors is not None:
           prior = i_priors[i_target_index,:] 
        else:
            prior = None
        letter_scores = np.zeros( X.shape )
        (rows, cols) = np.nonzero( Y >= X )
        letter_scores[rows,cols] = self.getScores(X[rows,cols] , Y[rows,cols] , target_mean, i_gauss_std, prior,  i_clip=False)
        bounds = [(min_x, max_x), (min_x, max_y)]
        return (X, Y,  letter_scores, bounds)
    
    def getScores(self, i_x, i_y, i_mean, i_gauss_std, i_prior,  i_clip, i_log=False):
        """Compute the unnormalised scores if the first click is at i_x seconds, and the second click 
        at i_y seconds, according to a gaussian with mean i_mean and i_gauss_std. If i_clip=True, 
        the score with be set to zero if i_y <= (i_x + self.reaction_delay)"""
        scores  = 2.0*(0.5**2)*self.getAllModelScores(i_x, i_y, i_mean, i_gauss_std, i_clip, i_log)
        total_score = np.array( np.sum(scores, axis=0) ) 
        if i_prior is not None:
            if i_log:
                total_score += i_prior
            else:
                total_score *= i_prior
        return total_score
   
    def getAllModelScores(self, i_x, i_y, i_mean, i_gauss_std, i_clip, i_log ):
        """Compute all four mixture components associated with the different hypothesis about 
        which model is responsible for which click"""
        model_idx = np.array([[0,0],[0,1],[1,0],[1,1]])
        model_means =   i_mean[model_idx]
        gauss_var = i_gauss_std**2
        scores = np.array([self.__getScores(model_means[m,:], gauss_var, i_x, i_y, i_clip, i_log) for m in range(0, model_means.shape[0])])
        return scores
    
    def getNormalisedScore(self, i_t1, i_t2, i_means, i_target_idx, i_gauss_std, i_priors, i_clip ):
        """Compute the normalised posterior probability. Work out"""
        if i_priors is not None:
           prior =np.array( i_priors[i_target_index,:])
        else:
            prior = None
        target_score =  self.getScores( i_t1, i_t2, i_means[i_target_idx,:], i_gauss_std, prior, i_clip)
        total_score = 0.0
        for n in range(0, i_means.shape[0]):
            if i_priors is not None:
                prior =np.array( i_priors[i_target_index,:])
            else:
                prior = None
            tmp_score = self.getScores( i_t1, i_t2, i_means[n,:], i_gauss_std, prior, i_clip)
            total_score += tmp_score 
        return self.__normaliseLikelihoods(target_score, total_score)
    
    def getBoundaryCost(self, t1_t2, i_target_idx, i_means, i_gauss_std, i_priors, i_thresh, i_clip, i_approx_grad):
        """Objective Function used by refineBoundary accuracy (minimised), where an initial 
        guess is made to where the decision boundary is. Here a minimiser is used to find the 
        closest point where the posterior probabilityof a letter is closest to the input threshold""" 
        t1,t2 = t1_t2 
        (score_m, alpha_m) = self.__getIntermediateGradBoundary( i_means, i_target_idx, i_gauss_std, t1, t2,i_priors)
        if np.abs(score_m) < self.eps:
            if i_approx_grad:
                return i_thresh
            else:
                return (i_thresh,  np.array([0.0, 0.0]) )
        total_score = np.float(score_m)
        total_grad = np.float64(alpha_m)
        
        for n in range(0, i_means.shape[0]):
            if not (n == i_target_idx): 
                (other_score, other_alpha) = self.__getIntermediateGradBoundary(i_means,n, i_gauss_std, t1, t2, i_priors)
                total_score += other_score
                total_grad  += other_alpha
        score_m /= total_score
        
        k = score_m / (total_score*i_gauss_std**2)  
        grads = -k*(total_score*i_means[i_target_idx,:] - total_grad)       
        f = np.abs(i_thresh - score_m) 
    
    
        if i_approx_grad:
            return f
        return f , grads
        
    def __getIntermediateGradBoundary( self, i_means, i_idx, i_gauss_std, i_t1, i_t2, i_priors):
        """ * Get intermediate values for gradient computation of the scores
            * model_idx = np.array([0,0],[0,1],[1,0],[1,1] """
            
        m = np.float64(i_means[i_idx,:])
        all_scores = self.getAllModelScores(i_t1, i_t2, m , i_gauss_std, i_clip=False )
        score = np.sum(all_scores, axis=0) 
        if i_priors is not None:
            score *= (self.__getPrior(i_priors, i_target_idx) )
        #alpha_1 =  (i_t1 - m[0])*(all_scores[0] + all_scores[1]) 
        #alpha_1 += (i_t1 - m[1])*(all_scores[2] + all_scores[3]) 
        # alpha_2 =  (i_t2 - m[0])*(all_scores[0] + all_scores[2]) 
        #alpha_2 += (i_t2 - m[1])*(all_scores[1] + all_scores[3]) 
        return (score, m*score)
        
    
    def __getPrior(self, i_priors, i_index):
        if i_priors is not None:
            prior = i_priors[index]
        else:
            prior = None
        return prior
        
    def __getScores(self, i_mean, i_gauss_var, i_x, i_y, i_clip, i_log ):
        """The unnormalised Gaussian scores for the click distribution (constants cancel out)"""
        dist = -((i_x - i_mean[0])**2 + (i_y - i_mean[1])**2) / (2.0*i_gauss_var)
        if not i_log:
            dist = np.exp(dist) / np.sqrt(2*np.pi*i_gauss_var)  
            zero_val = 0.0
        else:
            zero_val = -np.inf
        if i_clip:
            diff = i_y-i_x
            if np.ndim(diff) < 1:
                if diff <= self.recovery_time:
                    return zero_val
            elif np.ndim(diff) < 2:
                idx = np.nonzero( diff <= self.recovery_time)[0]
                dist[idx] = zero_val
            else:
                (row, col) = np.nonzero( diff <= self.recovery_time )
                dist[row, col] = zero_val
        return dist
    
    def __normaliseLikelihoods(self, i_target_scores, i_normaliser):
        """Divide target scores by i_normaliser and set values in target scores smaller than epsilon equal to zero."""
        final_scores = i_target_scores / i_normaliser
        if np.ndim(i_target_scores) < 1:
            if i_target_scores < self.eps:
                return 0.0
            return final_scores
        elif np.ndim(i_target_scores) < 2:
            idx = np.nonzero( np.abs(i_target_scores) < self.eps)[0]
            final_scores[idx] = 0.0
        else:  
            (row,col) = np.nonzero( np.abs(i_target_scores) < self.eps)
            final_scores[row,col] = 0.0
        return final_scores
    
    ############################################ Axis limits 

    def __getAxisLimits(self,  i_n_std, i_gauss_std, i_x=np.array([0.0]), i_y=np.array([0.0])):
        """Get the axis limits:  the axis limits will bte at least i_n_std*i_gauss_std away from the min,max limits
          contained in i_x, i_y"""
        (min_x, max_x) = (- i_n_std*i_gauss_std + np.min(i_x), i_n_std*i_gauss_std + np.max(i_x) ) 
        #(min_x, max_x) = ( np.min(i_x), np.max(i_x)) 
        (min_y, max_y) = (np.min(i_y)- i_n_std*i_gauss_std, i_n_std*i_gauss_std + np.max(i_y)) 
        #(min_y, max_y) = (min_x, max_x)  
        return (min_x, max_x, min_y, max_y)

    def __getAxis(self, i_target_val, i_n_samples, i_n_std, i_gauss_std, i_x, i_y, i_T):
        """Get the axis limits and mesh such that i_target_val is included in the mesh:
           i_n_samples: The number of samples; see __getAxisLimits()"""
        (min_x, max_x, min_y, max_y) = (0.0, i_T, 0.0, i_T)
        x = np.linspace(min_x, max_x, i_n_samples)
        y = np.linspace(min_y, max_y, i_n_samples)
            
        """if i_target_val is None:
            x = np.linspace(min_x, max_x, i_n_samples)
            y = np.linspace(min_y, max_y, i_n_samples)
        else:
            step_size_x = (max_x - min_x) / np.float(i_n_samples-1)
            x= np.arange(min_x, i_target_val[0], step_size_x)  
            x= np.hstack( (x ,  np.arange(i_target_val[0], max_x, step_size_x)  ))
            step_size_y = (max_y - min_y) / np.float(i_n_samples-1)
            y= np.arange(min_y, i_target_val[1], step_size_y)  
            y = np.hstack( (y ,  np.arange(i_target_val[1], max_y, step_size_y)  ))"""
        (min_x, max_x, min_y, max_y) = ( np.min(x), np.max(x), np.min(y), np.max(y))
        print "AXIS LIMITS:", (min_x, max_x, min_y, max_y)
        [X,Y] = p.meshgrid(x, y )
        return (min_x, max_x, min_y, max_y, X, Y)
    
    ################################## Decision boundaries
    
    def contourSegs(self, i_x, i_y, i_z, i_level):
        """Extract the contours (from i_z) at level i_level. 
           Return the list of coordinates from i_x, i_y.
           This is used as an initial guess of the decision bounaries for a specific letter.""" 
        z = ma.asarray(i_z, dtype=np.float64)  # Import if want filled contours.
        c = cntr.Cntr(i_x, i_y, z)
        segs = c.trace(i_level, i_level, 0)
        for s in segs:
            s = np.vstack( (s, s[0,:]) )
        return segs    # x,y coords of contour points. 
        
    def reduceContourPoints(self, i_segs, i_thresh, i_max_dist, i_display):
        """Reduce the number of points by ignoring some points that fall on 
           a straight line""" 
        final_segs = []
        for s in i_segs:
            seg_idx = [0]
            offset = s[0,:]
            for n in range(1, s.shape[0]-1):
                (cost, delta_dist, delta_x, delta_y) = self.__curvatureCost( s, seg_idx, n)
                if cost > i_thresh:
                    """Choose the current point as break point if the error 
                       occurs between two successive indices"""
                    if (n - seg_idx[-1]) == 1:
                        seg_idx.append( n )
                    elif ( n - seg_idx[-1]) == 2:
                        """Difference of 2 between indices - append the previous index
                        as break point (the current one overshot the error threshold)
                        and compute the error between new break point and current index"""
                        seg_idx.append( n - 1 )
                        self.__thresholdCurvatureCost(s, seg_idx, n, i_thresh, i_max_dist)
                    else:
                        """"Compute the perpendicular distance between the current 
                        chord and the points on the curve between them. Choose the 
                        point where the distance is maximum as the breakpoint (the 
                        index will be smaller than n). Then compute the error between 
                        new break point and current index"""
                        (dist, idx_range) = self.__perpendicularDist(s, seg_idx, n)
                        idx_max = np.argmax(dist)
                        seg_idx.append( idx_range[idx_max] )
                        self.__thresholdCurvatureCost(s, seg_idx, n, i_thresh, i_max_dist)
                elif delta_dist >= i_max_dist:
                    seg_idx.append( n )   
            seg_idx.append(0)
            seg_idx = np.array(seg_idx) 
            final_segs.append( s[seg_idx,:])
            if i_display:
                print "Number of points reduced from ", s.shape[0], " to ", len(seg_idx)
        return final_segs
    
            
    def refineContourAccuracy(self, i_target_index, i_pos, i_gauss_std, i_thresh, i_priors,  i_segs, i_bounds, i_display, i_approx_grad):
        clip = False
        target_mean = i_pos[i_target_index,:]
        o_segs = []
        if i_approx_grad == 0:
            min_cost =  0.005
        else:
            min_cost = 1E-6      #The max distance allowed between points found and i_prob_thresh
       
        for segs in  i_segs:
            if i_display:
                print "SEG size = ", segs.shape, " all seg shape = ", len(i_segs)
            init_scores =  self.getNormalisedScore(segs[:,0], segs[:,1], i_pos, i_target_index, i_gauss_std, i_priors, clip)
            args = (i_target_index, i_pos, i_gauss_std, i_priors , i_thresh, clip, i_approx_grad )
            roots = []
            for n in range(0, segs.shape[0]-1):
                if np.abs( init_scores[n] ) < min_cost:
                    roots.append(segs[n,:] )
                    continue
                x0 = list(segs[n,:])
                params, f_val, results = solver.fmin_l_bfgs_b( func=self.getBoundaryCost, 
                                                    x0=x0, 
                                                    fprime=None, 
                                                    args=args,
                                                    approx_grad=i_approx_grad,
                                                    bounds=i_bounds, 
                                                    m=10, 
                                                    factr=10.0,
                                                    pgtol=1e-10,
                                                    epsilon=1e-10,
                                                    iprint=-1,
                                                    maxfun=1000 )
                prob = self.getNormalisedScore(params[0],params[1], i_pos, i_target_index, i_gauss_std, i_priors, i_clip=False ) 
                if (prob + min_cost) < i_thresh:
                    print "WHAM: cost = ", f_val, " prob = ", prob, " diff = ", prob-min_cost, " i_thresh = ", i_thresh
                    continue
                if len(roots) < 1:
                    roots.append(params)
                    continue
                tmp_roots = np.array(roots)
                c1 = np.abs( tmp_roots[:,0] - params[0] ) < self.eps
                c2 = np.abs( tmp_roots[:,1] - params[1] ) < self.eps
                idx = np.nonzero( np.logical_and( c1, c2 ))[0]
                if len(idx) > 0:
                    continue 
                roots.append(params)
                prob = self.getNormalisedScore(params[0],params[1], i_pos, i_target_index, i_gauss_std, i_priors, i_clip=False ) 
                if i_display:
                    print "x0 = ", x0, " root = ", roots[-1], " z0 = ", init_scores[n], " prob = ", prob, "  cost = " , f_val
            if len(roots) > 0:
                roots.append( roots[0] ) 
                o_segs.append(np.array(roots))
        return o_segs        

    def __letterDecisionBoundaries( self,  i_X, i_Y, i_scores, i_bounds, i_level, i_target_index, i_pos, i_gauss_std, i_priors, i_display, i_score_mask, i_plot):
        score_mask = np.bool8(np.ones( i_scores.shape))
        total_scores = np.array( i_scores)
        for n in range(0, i_pos.shape[0]):
            if not (n == i_target_index): 
                if i_priors is not None:
                    prior = i_priors[n] 
                else:
                    prior = None
                click_scores = self.getScores( i_X,  i_Y, i_pos[n,:], i_gauss_std, prior, i_clip=False )
                total_scores += click_scores
                if i_score_mask:
                    is_bigger = np.logical_and( i_scores >= (click_scores-self.eps), np.abs(i_scores) > self.eps )
                    score_mask = np.logical_and( score_mask, is_bigger)
        scores = self.__normaliseLikelihoods( i_scores,  total_scores)
        #if i_score_mask:
        #    scores *= np.int32(score_mask)
        #segs = self.__extractContours(i_X, i_Y, scores, i_bounds, i_level, i_target_index, i_pos, i_gauss_std, i_priors, i_display, i_plot) 
        segs = []
        return (scores, segs, score_mask)
    
    def __extractContours(self, i_X, i_Y, i_scores, i_bounds, i_level, i_target_index, i_pos, i_gauss_std, i_priors, i_display, i_plot):
        """* Extract the contours from i_scores, on grid defined by i_X, i_Y.
           * i_level: probability defining the contour
           * i_target_index: The letter we want to write (define means of the gaussians used in mixture model)
           * i_pos: The 2D positions associated with each letter (time of click 1 and 2). 
           * i_priors: Prior probability associated with each letter.
           * i_display: Write diagnostic messages to stdout or not. """
        
        curve_cost_thresh = 0.005         #Curvature threshoold to reduce the number of contour points
        curve_dist_thresh = 0.1           #Min distance allowed between successive curvature points
        init_segs = self.contourSegs(i_X, i_Y, i_scores, i_level)
        if not is_fast:
            reduced_segs = self.reduceContourPoints(init_segs, curve_cost_thresh, curve_dist_thresh, i_display)
            final_segs = self.refineContourAccuracy(i_target_index, i_pos, i_gauss_std, i_level ,
              i_priors, reduced_segs, i_bounds, i_display, i_approx_grad=True)
        if i_plot:
            #A diagnostic plot of the contours
            if plotContoursSeparate:    
                p.figure(facecolor='w'); p.grid('on');
                self.__plotContourSegs(init_segs, i_color='b',)
                self.__plotContourSegs(reduced_segs, i_color='k' )
                self.__plotContourSegs(final_segs, i_color='r' )
        if is_fast:
            return init_segs
        return final_segs 
    
    def __thresholdCurvatureCost(self, i_segs, io_seg_idx, i_cur_index, i_thresh, i_max_dist):
        """This function appends the index (i_cur_index) to io_seg_idx, if it is classified as 
        as a critical point. That is, if it is classified as a high curvature point (curvature 
        cost > i_thresh), or if it is more than i_max_dist pixels away from the previous selection."""
        (cost, delta_dist, delta_x, delta_y) = self.__curvatureCost( i_segs, io_seg_idx, i_cur_index )
        if cost >= i_thresh:
            io_seg_idx.append( i_cur_index )
        elif delta_dist >= i_max_dist:
            io_seg_idx.append( i_cur_index )
    
    def __curvatureCost(self,  i_segs, i_seg_idx, i_cur_index ):
        """Compute the cost associated with a curvature point, according to "Practical Algorithms
        for image analysis, by Seul, O'Gorman and Sammon, page 194."""
        (prev_xi, prev_yi, xi, yi, orig_xi, orig_yi) = self.__getCurvatureCostPoints(i_segs, i_cur_index, i_seg_idx)
        delta_x  = xi - prev_xi
        delta_y  = yi - prev_yi
        Li = np.sqrt( (xi-orig_xi)**2 + (yi-orig_yi)**2 )
        Ai = (xi-orig_xi) * delta_y - (yi-orig_yi)*delta_x
        cost = np.abs(Ai) / Li 
        delta_dist = np.sqrt( (xi - orig_xi)**2 + (yi - orig_yi)**2)
        return (cost, delta_dist, delta_x, delta_y)
    
    def __perpendicularDist(self, i_segs, i_seg_idx, i_cur_idx):
        """A point is classified as critical if its associated curvature cost exceeds a threshold. 
        To insert a better breakpoint, the contour point with the largest perpendicular cost to the 
        current chord is selected as the new breakpoint. We use homogoneous coordinate for this 
        computation; see curvatureCost for book reference to algorithm."""
        dist = []
        (prev_xi, prev_yi, xi, yi, orig_xi, orig_yi) = self.__getCurvatureCostPoints(i_segs, i_cur_idx, i_seg_idx)
        (delta_x, delta_y) = (xi - orig_xi, yi-orig_yi)
        if np.abs(delta_x) < self.eps:
            """Vertical line"""
            l1 = np.array([0.0, 1.0, -yi])
        else:
            m = delta_y / delta_x
            l1 = np.array([-m, 1.0, m*xi - yi ])
        idx_range = range(i_seg_idx[-1]+1, i_cur_idx) 
        for k in idx_range: 
            (pt_x, pt_y) =  (i_segs[k,0], i_segs[k,1])
            if np.abs(l1[0]) <= self.eps:
                #Vertical line
                (xc, yc) = (xi, pt_y) 
            else:
                l2 = np.array([1.0/m, 1.0, -(pt_y + (1.0/m)*pt_x)])
                crosspoint = np.cross(l1, l2)
                crosspoint /= crosspoint[2]
                if np.abs(crosspoint[2]) < self.eps:
                    raise ValueError("Lines are parallel can not cross!!!")    
                (xc, yc) = (crosspoint[0], crosspoint[1])
            dist.append( np.sqrt( (pt_x - xc)**2 + (pt_y - yc)**2) )
            if np.abs(l1[0]) > self.eps:
                skuins = np.sqrt( (orig_xi - pt_x)**2 + (orig_yi - pt_y)**2 )
                a1  = np.arcsin(dist[-1] / skuins) * 180 / np.pi
                teenoorstaand = np.sqrt( (xc - orig_xi)**2 + (yc - orig_yi)**2 )
                a2 = np.arcsin(teenoorstaand / skuins) * 180 / np.pi
                if  np.abs(a1+a2 - 90.0) > 1E-3:
                    print "a1 = ", a1, " a2 = ", a2, " sum  = ", a1 + a2 
                    raise ValueError('Error in angle calculations')
        return (np.array(dist), idx_range)
    
    def __getCurvatureCostPoints(self, i_segs, i_cur_index, i_seg_idx):
        """Retrieve the curvate points in the format to caluclate high curvature regions, 
        in order to reduce the number of contour points. Helper function for 
        reduceContour points."""
        (orig_xi, orig_yi) = (float(i_segs[i_seg_idx[-1], 0]), float(i_segs[i_seg_idx[-1], 1]))
        (xi, yi) = (float(i_segs[i_cur_index,0]), float(i_segs[i_cur_index,1]))
        (prev_xi, prev_yi) = (float(i_segs[i_cur_index-1,0]), float(i_segs[i_cur_index-1,1]))
        return (prev_xi, prev_yi, xi, yi, orig_xi, orig_yi )
    
    ##################################### Display
    
    def __plotAllLetterScores(self, i_alphabet, i_pos, i_scores, i_X, i_Y, i_color_bar_title, i_gray=False, i_target_index=None):
        """Image plot of i_scores. Also plot the letters where they occur"""
        fig = p.figure(facecolor='w'); 
        self.__plotLetters(i_alphabet, i_pos[:,0], i_pos[:,1], i_delta_x=0.00, i_alpha=0.5, i_color='white', i_marker=None, i_target_index=i_target_index)
        self.__dispLetterProbs( i_scores, i_X, i_Y,  i_color_bar_title)
 
        
    def __plotContourSegs(self, i_segs, i_color='k',  i_linewidth=3):
        """Plot the list of contours stored in i_segs"""
        for s in i_segs:
            p.plot(s[:,0], s[:,1], i_color, linewidth=i_linewidth)
    
    def __plotLetters(self, i_letters, i_x, i_y, i_delta_x=0.0, i_alpha=0.6, i_color='white', i_marker=None, i_target_index=None):
        """Plot the alphabet letter positions in 2D
           i_letters: The alphabet letters
           i_x, i_y: The positions in 2D
           i_delta_x: A small offset from the dot (position) to the text label (letter).
           i_alpha: Alpha value of dot
           i_color: Color of text and dots"""
        p.hold('on')
        if i_marker is not None:
            p.plot(i_x, i_y,  self.colors[i_color]+i_marker, alpha=i_alpha, linewidth=1)
        for n in range(0, len(i_x)):
            if self.disp.params['text.usetex'] and ( i_letters[n] == '_') :
                letter_str = "\_"
            else:
                letter_str = i_letters[n]
            a = p.text(i_x[n]+i_delta_x, i_y[n],  letter_str, fontsize=self.disp.params['text_font'],ha='center', va='center', alpha=i_alpha  )
            a.set_color(i_color)
        p.xlabel('$t_{1}$ $\\mathrm{(seconds)}$'); p.ylabel('$t_{2}$ $\\mathrm{(seconds)}$');
        #p.xlabel('t1'); p.ylabel('t2')
    
    def __dispLetterProbs(self, i_image, i_X, i_Y, i_color_bar_title, is_gray=False):
        #Compute the maximum
        idx_max = np.argmax(i_image)
        max_score =  i_image.flatten()[idx_max]
        (x_m,  y_m) = (i_X.flatten()[idx_max], i_Y.flatten()[idx_max])
        #Display
        (min_x, max_x, min_y, max_y ) = (np.min(i_X), np.max(i_X), np.min(i_Y), np.max(i_Y))
        if is_gray:
            p.imshow(i_image, origin='lower', extent=(min_x, max_x, min_y, max_y),  cmap = p.cm.binary) 
        else:
            p.imshow(i_image, interpolation='bilinear', origin='lower', extent=(min_x, max_x, min_y, max_y)) #, vmin=0, vmax=1.0)
        #p.plot( [x_m] , [y_m], 'ko', linewidth=3, label=("%.2f"%max_score))
        #print "MAX: score = ", max_score, " x_m=", x_m, " y_m=", y_m  
        p.axis('image');
        self.plt_utils.setColorBar(np.max(i_image), i_color_bar_title, i_min=0.0)
            
    ######################################################### Diagnostic
    
    def updateWordPriorsSlow(self, i_t1_x, m, target_letter_idx, gauss_var,target_word_idx, word_priors,letter_list, i_classify, i_thresh):
        """Update the priors if no decision has been made"""
        t= time.time()
        probs = np.zeros([len(i_t1_x), len(word_priors)])
        delta_t = np.abs( i_t1_x[1] - i_t1_x[0] )
        letter_scores =np.array( [self.__getScores1DSlow( m, gauss_var, t1, i_clip=False, i_log=False)  for t1 in i_t1_x])
        const =  delta_t / (np.sqrt(2* np.pi * gauss_var ) )
        target_scores = np.atleast_2d( const* np.array( letter_scores[:, target_letter_idx] )).transpose()        
        for n in range(0, len(i_t1_x)):
            tmp_letter_scores = letter_scores[n, :]
            probs[n,:] =  tmp_letter_scores.take(letter_list) 
        #print "for loop = ", time.time() - t, " s"
        probs *= word_priors 
        az =  np.sum( probs , axis=1 ) 
        idx = np.nonzero( np.abs(az) > self.eps )[0]
        factor = np.zeros(len(az))
        factor[idx] = 1.0 / az[idx] 
        factor = np.atleast_2d(factor).transpose()
        un_normalised = np.array(probs)
        probs *=  factor
        o_classify_probs = None
        if i_classify:
            (rows, cols) = np.nonzero( probs >= (i_thresh-0.2) )
            if len(rows) < 0:
                o_classify_probs = np.zeros(len(word_priors ))
            else:
                classify_probs = np.zeros(probs.shape)
                for k in range(0, len(rows)):
                    ay = un_normalised[rows[k], cols[k]]
                    azy = az[rows[k]] - ay
                    x = np.log(ay)-np.log(azy)
                    classify_probs[rows[k], cols[k]] = self.classifySlow(i_thresh,x , i_rate=30.0)
                classify_probs  *= target_scores
                o_classify_probs = np.sum(classify_probs, axis = 0) 
        o_priors = np.sum(probs*target_scores, axis = 0) 
        return (o_priors, o_classify_probs)  
            
    def classifySlow(self, i_T, i_x, i_rate):
        return 1.0 / (1.0 +np.exp(-i_rate*i_x + i_rate*np.log(i_T/(1-i_T))))
            
    def __getScores1DSlow(self, i_mean, i_gauss_var, i_x, i_clip, i_log ):
        """The unnormalised Gaussian scores for the click distribution (constants cancel out)"""
        dist = -((i_x - i_mean)**2)/ (2.0*i_gauss_var)
        if not i_log:
            dist = np.exp(dist)  
            zero_val = 0.0
        else:
            zero_val = -np.inf
        if i_clip:
            diff = i_y-i_x
            if np.ndim(diff) < 1:
                if diff <= self.recovery_time:
                    return zero_val
            elif np.ndim(diff) < 2:
                idx = np.nonzero( diff <= self.recovery_time)[0]
                dist[idx] = zero_val
            else:
                (row, col) = np.nonzero( diff <= self.recovery_time )
                dist[row, col] = zero_val
        return dist

class TickerClickNoise1D():
    def __init__(self):
        self.letters = ['a','b','c','d','e','f','g','h']
        self.sigma = .5
        self.k = 5.0
        self.n_letters  = len(self.letters)
        delta_letters = self.k*self.sigma 
        self.offset = 0
        self.um = np.linspace( self.k + self.offset,  (self.k+ self.offset)*(self.n_letters), self.n_letters )
        self.letter_index = 3
        self.um_input = self.um[self.letter_index]

        print "Input values"
        print " um              =", self.stringVector( self.um), " len = ", len(self.um), " n_letters = ", self.n_letters 
        
    def stringVector(self,i_vec, i_type="%.4f"):
            disp_str=""
            for val in i_vec:
                disp_str += ( " " + i_type % (val) )
            disp_str = list(disp_str)
            disp_str[0]="["
            disp_str.append("]")
            return ''.join(disp_str)
                
    def plotDistribution(self):
        t = list(np.hstack( (np.linspace(20-5, 20, 200) , np.linspace(20, 20+5,200)  )))
        y = [self.getPosterior(tc) for tc in t]
        print "mid = ", t[200], " y = ", y[200]
        print "SUM : " ,  np.sum(np.array(y))
        print "Len y = ", len(y), " len t = ", len(t)
        p.figure; p.hold(True); p.plot(t, y, 'r');
        p.xlabel('tc'); p.ylabel('p(xm | tc)')
        p.show()
        
    def getPosterior(self, tc, display=False):
        dist = (tc - self.um_input)**2 - (tc - self.um)**2 
        dist = np.hstack( (dist[0:self.letter_index], dist[self.letter_index+1:]) ) 
        normalised_dist = dist / (2.0 * ( self.sigma**2))
        exp_dist = np.exp(normalised_dist)
        dist_total = np.sum(exp_dist)
        posterior =  1.0 / (1.0 + dist_total)
        if display:
            print " letter input  = ", self.um_input
            print " dist            = ", self.stringVector(dist )
            print " normalised_dist = " ,  self.stringVector(normalised_dist)
            print "exp_dist         = ",  self.stringVector(exp_dist)
            print "dist total       = ",  dist_total 
            print "posterior        = ",  posterior 
        return posterior
    
if __name__ ==  "__main__":
    #g = TickerClickNoise1D()
    #g.getPosterior(20, True) 
    #g.plotDistribution()
    g = TickerClickNoise2D()
    #g.compute()
    #g.printSigmoid()

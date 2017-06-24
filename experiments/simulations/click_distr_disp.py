

import pylab as p
import numpy as np
import sys
sys.path.append('../../')
sys.path.append('../')
from scipy.misc import factorial 
from utils import Utils, DispSettings
import time
import matplotlib._cntr as cntr
import numpy.ma as ma
from click_distr import ClickDistribution
import scipy.stats.distributions as sd
from plot_utils import PlotUtils


class ClickDistributionDisplay():
    ###################################### Init 
    def __init__(self, i_click_distr = None):
        """Diagnostic settings:
        self.normalise:  * Work with normalised distributions, 
                         * If false, all factors that are common to all letters will not be included. 
        self.disp:       * pylab settings
        self.log:        * Set only to false when doing plots
        self.log_const_gauss * Gaussian normalisation constant, for simulations"""
        if not i_click_distr is None:
            self.distr = i_click_distr
        else:
            self.distr = ClickDistribution()
        self.distr.normalise = True
        self.disp = DispSettings()
        self.plt_utils = PlotUtils()
        self.utils = Utils()
        
    def init2DScores(self, i_nsamples):
        """Initialise the normalising constant associated with M=2, L=2, 
           and compute the set of t1, t2 on a grid of n_samples x n_samples,
           subject to the constraint that t1 < t2"""
        x = np.linspace(0.0, self.distr.T, i_nsamples)
        y = np.linspace(0.0, self.distr.T, i_nsamples)
        [X,Y] = p.meshgrid(x, y )
        (rows, cols) = np.nonzero( Y >= X )
        ZC = self.distr.labelProbs(i_M=2)
        return (X, Y, rows, cols, ZC)
    
    #################################### Set
    
    def setChannel(self, i_channel_config):
        self.distr.reset(i_channel_config)

    #################################### Main 
    
    def draw2DLetterConfig(self, i_nsamples):
        (X, Y, rows, cols, ZC) = self.init2DScores(i_nsamples)
        scores = 255*np.ones([i_nsamples, i_nsamples])
        cbar = self.dispScoresImage(scores, X, Y, None, is_gray=True)
        self.dispLetters2D(i_delta_x=0.0, i_alpha=0.3, i_color='k', i_marker=None , i_target_index = None)
        
    def draw1DClick(self, i_target_idx, i_nsamples, is_noise_simple, is_many, i_draw_all_means=True ):
        times = np.linspace(0.0, self.distr.T, i_nsamples)
        scores = []
        ZC = self.distr.labelProbs(i_M=1)
        obs = np.atleast_2d(times).transpose()
        (delay, std) = (self.distr.delay, self.distr.std)
        #Plot The scores
        t_eval = self.distr.loc[i_target_idx,:]
        true_scores = np.atleast_2d(self.distr.utils.likelihoodGauss(obs-t_eval, std, i_log=False))
        gauss_scores =  0.5*(true_scores[:,0] + true_scores[:,1])
        if is_noise_simple:
            final_scores = np.array(gauss_scores)
        else:
            final_scores = np.zeros(i_nsamples)
            for n in range(0, true_scores.shape[0]):
                #[p(t | L1) , p(t | L2), ...]  
                obs_scores = np.atleast_2d(true_scores[n,:])
                final_scores[n] = self.distr.sumLabels( np.atleast_2d(obs_scores), ZC )
        p.plot(times, final_scores, 'k')
        if not i_draw_all_means:
            return np.max(final_scores)
        if is_many:
            max_val =  np.max(gauss_scores)
        else:
            max_val = max(final_scores)
        #Plot the letter lines (gauss means) and label them
        y_line_start = -0.05*max_val
        text_start = y_line_start -0.1*max_val
        delta_text = -0.1*max_val
        text_end =  text_start + delta_text
        xmin = self.distr.loc[0,0]-0.1
        (xmax, ymin, ymax) = (self.distr.T, text_end,  max_val)
        if i_draw_all_means:
            for (n,m) in enumerate(self.distr.loc):
                if n == i_target_idx:
                    p.plot([m, m],[0.0, max_val], 'k--', linewidth=1)
                p.plot([m, m], [y_line_start, 0], linewidth=1, color='k')
                #Display alphabet
                for lm in m:
                    disp_str = str(self.distr.alphabet[n])
                    if disp_str == "_":
                        disp_str = "\_" 
                    #p.plot([lm], [text_end], 'w+')
                    p.text(lm, text_start,  disp_str,  fontsize=self.disp.params['alphabet_size'], 
                        ha='center', va='baseline', fontweight='bold')
            #Adjust the axis     
            p.axis([xmin, xmax, ymin, ymax])
            target_letter = self.distr.alphabet[i_target_idx]
            p.xlabel("$t_{1}$ $\\mathrm{(seconds)}$")
            #Adjust the ticks
            self.plt_utils.adjustYTicks(i_ymin=0.0)
        return max_val
        
    def draw2DClick(self, i_target_idx, i_letter, i_nsamples, i_is_gray, i_is_save, i_plot_num=0, i_is_many=False):
        """Draw a 2D image of p(t1, t2, M = | letter), where letter=self.alphabet[i_target_idx],
           and there are i_nsamples x i_nsamples (t1,t2) pairs uniformly spread in region (0, Ts)
        i_is_many: True if many figures are drawn and one should not add extra things like labelling 
                   the regions of interest (too cluttered)."""
        (X, Y, rows, cols, ZC) = self.init2DScores(i_nsamples)
        score_file = "./click_distr_plots/target_score2D_%.2d_%.2d.npy" % (i_target_idx, i_plot_num)
        if i_is_save:
            (scores, t_start) = self.target2DScores( i_target_idx, X, Y, rows, cols, ZC )
            np.save(score_file, scores)
        scores = np.load(score_file)    
        cbar_title = "$P(t_{1}, t_{2}, M=2 \mid  \\boldsymbol{\\theta}, \ell=\mathrm{%s})$" % i_letter
        cbar = self.dispScoresImage(scores, X, Y, cbar_title, is_gray=False)
        #self.dispLetters2D(i_delta_x=0.0, i_alpha=0.3, i_color='w', i_marker=None , i_target_index =i_target_idx)
        return cbar
        
    def drawAll2DClicks(self, i_letter_ids_file, i_scores_file,  i_nsamples, i_is_gray, i_is_save, i_draw_text=False, i_normalise=True ):
        """Draw 2D image of the results from allTarget2DScoresMax(i_nsamples)."""
        if i_is_save:
            (final_scores, letter_ids) = self.allTarget2DScoresMax(i_nsamples, i_normalise=i_normalise)
            np.save(i_scores_file, final_scores)
            np.save(i_letter_ids_file, letter_ids)
        final_scores = np.load(i_scores_file)
        letter_ids = np.load(i_letter_ids_file)
        (X, Y, rows, cols, ZC) = self.init2DScores(max(final_scores.shape))
        img_boundaries = self.letterRegionBoundaries(letter_ids)
        #contour_segs = self.contourSegs(X, Y, img_boundaries, i_level=0, i_min_contour_area=0.0)
        cbar_title = "$P( \ell^{*} \mid  \\boldsymbol{\\theta}, \mathbf{t}_{1}, M=2)$"
        cbar = self.dispScoresImage(final_scores, X, Y, cbar_title, is_gray=False)
        #self.plotContourSegs(contour_segs, i_alpha=0.6, i_color='w',  i_linewidth=1)
        if i_draw_text:
            self.dispLetters2D(i_delta_x=0.12, i_alpha=1.0, i_color='w', i_marker=None)
        return cbar

    def dispLetters2D(self, i_delta_x=0.05, i_alpha=0.6, i_color='w', i_marker='o',  i_target_index=None ):
        """Plot the alphabet letter positions in 2D (only useful when alphabet is repeated twice.
           i_letters: The alphabet letters
           i_x, i_y: The positions in 2D
           i_delta_x: A small offset from the dot (position) to the text label (letter).
           i_alpha: Alpha value of dot
           i_color: Color of text and dots"""
        if i_marker is not None:
            p.plot(self.distr.loc[:,0], self.distr.loc[:,1],  i_color+i_marker, alpha=i_alpha, linewidth=1)
        for n in range(0, self.distr.loc.shape[0]):
            if (self.distr.alphabet[n] == '_') and self.disp.params['text.usetex'] :
                letter_str = "\_"
            else:
                letter_str = self.distr.alphabet[n]
            (x,y) = (self.distr.loc[n,0], self.distr.loc[n,1])
            font = self.disp.params['text_font']
            a = p.text(x+i_delta_x, y,  letter_str, fontsize=font,ha='center', va='center', alpha=i_alpha )
            a.set_color(i_color)
        self.add2DAxisLabels()
        
    def dispScoresImage(self, i_image, i_X, i_Y, i_cbar_title, is_gray=False):
        """Display a 2D matrix of scores as a 2D image (associated with 2 repetitions of the alphabet"""
        (min_x, max_x, min_y, max_y ) = (np.min(i_X), np.max(i_X), np.min(i_Y), np.max(i_Y))
        if is_gray:
            p.imshow(i_image, origin='lower', extent=(min_x, max_x, min_y, max_y),  cmap = p.cm.binary) 
        else:
            p.imshow(i_image, interpolation='bilinear', origin='lower', extent=(min_x, max_x, min_y, max_y)) #, vmin=0, vmax=1.0)
        p.axis('image') 
        cbar = self.plt_utils.setColorBar(np.max(i_image), i_cbar_title, i_min=0.0)
        self.add2DAxisLabels()
        return cbar
            
    ###################################### Scores
    
    def allTarget2DScoresMax(self, i_nsamples, i_normalise=True):
        """Compute the maximum posterior scores over all letters 
           and i_nsamples x i_nsamples click times, 
           assuming uniform priors, i.e., max( p_1, p_2, ... p_A) where 
           p_a = P(letter_a | t1, t2, M=2). 
           Also return the letter ids 1...A associated with each score. """
        (X, Y, rows, cols, ZC) = self.init2DScores(i_nsamples)
        scores = np.zeros(X.shape)
        total_scores = np.zeros(X.shape)
        letter_ids = np.zeros(X.shape) 
        for n in range(0, len(self.distr.alphabet)):
            (target_scores, t_start) = self.target2DScores(n, X, Y, rows, cols, ZC)
            total_scores += target_scores
            print "n = ", n, " letter = ", self.distr.alphabet[n], " time = ", time.time() - t_start
            (rows2, cols2) = np.nonzero( target_scores - scores > self.distr.utils.eps)
            scores[rows2, cols2] = np.array( target_scores[rows2, cols2] )
            letter_ids[rows2, cols2] = n + 1
        final_scores = np.zeros(X.shape)
        (rows, cols) = np.nonzero(  total_scores > self.distr.utils.eps )
        final_scores[rows, cols] = scores[rows, cols]  
        if i_normalise:
            final_scores[rows, cols] /= total_scores[rows, cols]
        return (final_scores, letter_ids)

    def target2DScores(self, i_target_idx, i_X, i_Y, i_rows, i_cols, i_ZC):
        """Compute P(t1,t2 | letter) for a discrete set of samples, representing 
           all possible values of t1, t2 on a 2D grid"""
        t=time.time()
        times = np.zeros(2)
        obs_t1 = np.atleast_2d(i_X[i_rows, i_cols]).transpose() - self.distr.loc[i_target_idx,:]
        obs_t2 = np.atleast_2d(i_Y[i_rows, i_cols]).transpose() - self.distr.loc[i_target_idx,:]
        g1 = self.utils.likelihoodGauss( obs_t1, self.distr.std, False)
        g2 = self.utils.likelihoodGauss( obs_t2, self.distr.std, False)
        scores_tmp = np.zeros(len(i_rows))
        for nt in range(0, len(i_rows)):
            obs =  np.vstack( (g1[nt,:], g2[nt,:]) )
            scores_tmp[nt]  = self.distr.sumLabels(obs, i_ZC)
        scores = np.zeros(i_X.shape)
        scores[i_rows, i_cols] = np.array(scores_tmp)
        return (scores, t)

    ###################################### Contours
    
    def letterRegionBoundaries(self, letter_ids):
        """Compute the decision boundary regions as determined by letter_ids, where 
           letter_ids has been computed by allTarget2DScoresMax"""
        letter_diff = np.zeros(letter_ids.shape)
        letter_diff_v =  np.clip(letter_ids[1:,:] - letter_ids[0:-1,:], -1, 1)
        letter_diff[1:,:] += np.abs(letter_diff_v)
        letter_diff_h =  np.clip(letter_ids[:,1:] - letter_ids[:,0:-1], -1, 1)
        letter_diff[:,1:] = np.logical_or( letter_diff[:,1:], np.abs(letter_diff_h))    
        return letter_diff
            
    def polyArea(self, p):
        """Compute the area of a polygon/contour"""
        return 0.5 * abs(sum(x0*y1 - x1*y0 for ((x0, y0), (x1, y1)) in self.segments(p)))

    def segments(self, p):
        return zip(p, p[1:] + [p[0]])
    
    def contourSegs(self, i_x, i_y, i_z, i_level, i_min_contour_area=0.0):
        """Extract the contours (from i_z) at level i_level. 
           Return the list of coordinates from i_x, i_y.
           This is used as an initial guess of the decision bounaries for a specific letter.""" 
        z = ma.asarray(i_z, dtype=np.float64)  
        c = cntr.Cntr(i_x, i_y, z)
        segs = c.trace(i_level, i_level, 0)
        o_segs = []
        for s in segs:
            #s = np.vstack( (s, s[0,:]) )
            poly_area = self.polyArea(s)
            if poly_area >= i_min_contour_area:
                o_segs.append(np.array(s))
        return o_segs    
        
    def plotContourSegs(self, i_segs, i_alpha=0.3, i_color='k',  i_linewidth=3):
        """Plot the list of contours stored in i_segs
           i_add_label: Add the alphabet letter closest to the centroid of each contour"""
        for s in i_segs:
            p.plot(s[:,0], s[:,1], i_color, linewidth=i_linewidth, alpha=i_alpha)
        
    #################### Axis / Figure labels
    
    def letterLabel(self, i_target_idx, M):
        target_letter = self.distr.alphabet[i_target_idx]
        time_str = ""
        for m in range(0, M):
            time_str += ("t" + str(m+1))
            if m < (M-1):
                time_str += ","
        label = "p(" + time_str + ", M=" + str(M) + " | letter = " + target_letter + ")"
        return label 
    
    def add2DAxisLabels(self):
        if  self.disp.params['text.usetex']:
            p.xlabel('$t_{1}$ $\\mathrm{(seconds)}$')
            p.ylabel('$t_{2}$ $\\mathrm{(seconds)}$')
        else:
            p.xlabel('t1 (seconds)')
            p.ylabel('t2 (seconds)')    

    ######################################## Plot the priors
    
    def plotPriorBeta(self):
        x = np.linspace( 1.0/ ( 0.05**2), 1.0/(0.3**2), 200 )
        a_range  =  [2] # np.linspace(10, 20,  3)
        b_range =  [0.01] # [0.001, 0.01, 0.05, 0.1] #np.linspace(0.1, 10,  3)

        p.figure() 
        labels  = []
        for a in a_range:
            for b in b_range:
                scores =  np.log(  sd.gamma.pdf( x , a, scale=1.0/b) )
                labels.append( "a=%.2f, b=%.2f" % (a,b) )
                p.plot(x, scores)

        p.legend( tuple(labels) )
        p.grid('on')
        p.show()
    
    def plotPriorFpRate(self):
        x = np.linspace( 0.0 , 0.05, 200 )
        a_range  =  [2] # np.linspace(10, 20,  3)
        b_range =  [200.0] # [0.001, 0.01, 0.05, 0.1] #np.linspace(0.1, 10,  3)

        p.figure() 
        labels  = []
        for a in a_range:
            for b in b_range:
                scores =  np.log(  sd.gamma.pdf( x , a, scale=1.0/b) )
                labels.append( "a=%.2f, b=%.2f" % (a,b) )
                p.plot(x, scores)

        p.legend( tuple(labels) )
        p.grid('on')
        p.show()
        
    def plotPriorFr(self):
        x = np.linspace( 0.0 , 0.3, 200 )
        a_range  =  [2] # np.linspace(10, 20,  3)
        b_range =  [10.0] # [0.001, 0.01, 0.05, 0.1] #np.linspace(0.1, 10,  3)
        p.figure() 
        labels  = []
        for a in a_range:
            for b in b_range:
                scores =  np.log(  sd.beta.pdf(x, a, b) )
                labels.append( "a=%.2f, b=%.2f" % (a,b) )
                p.plot(x, scores)

        p.legend( tuple(labels) )
        p.grid('on')
        p.show()
    
if __name__ ==  "__main__":   
    c =  ClickDistributionDisplay()
    #c.plotPriorFpRate() 
    c.plotPriorFr()

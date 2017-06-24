
import sys,  cPickle, time, itertools
import scipy.stats.distributions as sd
import numpy as np
import pylab as p
sys.path.append("../../")
sys.path.append("../")
from plot_utils import PlotUtils
from normalise_tests import TestGauss
from ticker_simulation import TickerSimulation
from click_distr_disp import  ClickDistributionDisplay

class TickerSimulationExamples(TickerSimulation):
    def __init__(self):
        TickerSimulation.__init__(self)
        self.plot_utils = PlotUtils()
        
    ##################################### Main
    
    def examples(self):
        self.sentenceExample()
        #self.sampleExample()
        #self.updateExampleGauss()
        #self.testTruePositiveSamples()
        
    def sentenceExample(self):
        """
        * Use the default parameters or some selected parameters to simulate the results on a given sentence. 
        * Optionally plot all the generated samplesWORDS =  ['at', 'me', '.']
        """
        #Parameters
        sentence  = "at_me_."
        default_params = self.defaultParams()         
        nsamples_sentence = 10
        #Display
        (disp_sentences, disp_top_words, disp_word_errors, save_click_times) = (True, True, True, True) 
        disp_settings = (disp_sentences, disp_top_words, disp_word_errors, save_click_times)
        #The results
        r = self.sentenceResults(sentence, default_params, disp_settings)
        (tot_error, speed, nclicks, error_rate,  click_times, word_lengths) = r
        r_normalised = self.normaliseResults(r)
        #Display of click times
        print_times = True
        draw_click_times = False #True
        save_image = False #Part of draw click times- this should be true, only set to false to save some time
        self.dispClickTimes(click_times, sentence, print_times, draw_click_times, save_image)
        #Display the results
        self.dispSentenceResultsRaw(r)
        self.dispSentenceResults(r_normalised)
        p.show()

    def sampleExample(self):
        """ * Plot the density conditioned on C and N (helper from normalise_tests)
            * See if the correct samples are plotted (more or less)"""
        params = self.defaultTickerParams()
        (nchannels, file_length, delay, fp_rate , fr ,  overlap,  end_delay,  std, root_dir) = params
        (delay, std) = (0.1, 0.1)  
        (fr, fp_rate) = (0.5, 0.3) 
        letter = "f"
        params = (nchannels, file_length, delay, fp_rate , fr ,  overlap,  end_delay,  std, root_dir)
        #Initialise the noise parameters
        self.setNoise(params)
        #Display the parameters
        disp_params =  (delay, std, fr, fp_rate, self.click_distr.T) 
        self.utils.printParams(disp_params, i_heading_str="Parameters corresponding to density plot")
        #Print example sample counts (20 samples)
        L = self.click_distr.loc.shape[1]
        N = self.sampler.sampleNumberFalsePositives(self.click_distr, 20)
        C = self.sampler.sampleNumberTruePositives(self.click_distr, L, 20) 
        print "Sampling number of fp/tp results: L = ", L, " C = ", C, " N= ", N
        #Generate some samples
        (samples_1, gauss_mu_1, tp_samples_1, fp_samples_1) = self.sampler.samplesJoint(self.click_distr, letter, 
            L=2, C=2, N=0, n_samples=100000, i_return_labels=True, i_n_rejection_samples=1E3, i_display=False)
        (samples_2, gauss_mu_2, tp_samples_2, fp_samples_2) = self.sampler.samplesJoint(self.click_distr, letter, 
            L=2, C=1, N=1, n_samples=100000, i_return_labels=True, i_n_rejection_samples=1E3, i_display=False)
        (samples_3, gauss_mu_3, tp_samples_3, fp_samples_3) = self.sampler.samplesJoint(self.click_distr, letter, 
            L=2, C=0, N=2, n_samples=100000, i_return_labels=True, i_n_rejection_samples=1E3, i_display=False)
        samples = np.vstack( (np.vstack((samples_1, samples_2)), samples_3))
        gauss_mu = np.hstack( (np.hstack((gauss_mu_1, gauss_mu_2)), gauss_mu_3))
        tp_samples = np.hstack( (np.hstack((tp_samples_1, tp_samples_2)), tp_samples_3))
        fp_samples = np.hstack( (np.hstack((fp_samples_1, fp_samples_2)), fp_samples_3))
        #Make the plot using normalise tests
        gt = TestGauss()
        uniform = False
        gt.plotDensity(samples, gauss_mu, self.click_distr.T, self.click_distr.std, uniform ) 
        gt.plotSamples(samples, gauss_mu, self.click_distr.T, self.click_distr.std, uniform, tp_samples, fp_samples)
        p.show()
        
    def updateExampleGauss(self):
        #Learn the gaussian density
        sentence = "the_quick_brown_fox_jumps_over_the_lazy_dog_."
        params = self.defaultTickerParams()
        (nchannels, file_length, delay, fp_rate , fr ,  overlap,  end_delay,  std, root_dir) = params
        (delay, std) = (0.1, 0.1)  
        (fr, fp_rate) = (0.3, 0.1)
        params = (nchannels, file_length, delay, fp_rate , fr ,  overlap,  end_delay,  std, root_dir)
        self.setNoise(params)
        (is_train,learning_rate, end_delay) = (True, 0.3, 0.3)
        (learn_delay, learn_std, learn_fp, learn_fr) = (True, True, True, True)
        self.click_distr.train_gauss = True
        self.click_distr.setFixLearning(learn_delay, learn_std, learn_fp, learn_fr)
        self.click_distr.setParams(is_train, self.channel_config, delay, std, fp_rate, fr, learning_rate, end_delay)
        #Draw ground truth gauss
        self.disp_settings.newFigure(); p.grid('on')
        x_eval = np.linspace( -10*std, 10*std, 200) + delay
        y_eval = sd.norm.pdf(x_eval, loc=delay, scale=std)
        p.plot(x_eval, y_eval, 'r')
        #Generate samples and update pdf
        words = self.phrase_utils.wordsFromSentece(sentence)  
        for m in range(0, len(words)):
            word = self.phrase_utils.getWord(words[m])
            print "************************************************************"
            print "Word = ", word 
            print "************************************************************"
            for letter in word:
                (click_time, N, C) = self.sampler.sample(self.click_distr, letter, i_return_labels=False,
                    i_n_rejection_samples=1E3, i_display=True)
                self.click_distr.storeObservation(click_time)
            self.click_distr.train(word)
        print "grnd_truth: delay=%.4f, std=%.4f, fr=%.4f, fp=%.4f" % (delay,std, fr,fp_rate)
        (delay, std, fr, fp_rate) = self.click_distr.getParams()
        print "learned params: delay=%.4f, std=%.4f, fr=%.4f, fp=%.4f" % (delay,std, fr,fp_rate)

    def testTruePositiveSamples(self):
        letter = "f"
        params = self.defaultTickerParams()
        (nchannels, file_length, delay, fp_rate , fr ,  overlap,  end_delay,  std, root_dir) = params
        (delay, std) = (0.2, 0.1)  
        (fr, fp_rate) = (0.0, 0.0)
        params = (nchannels, file_length, delay, fp_rate , fr ,  overlap,  end_delay,  std, root_dir)
        self.setNoise(params)
        n_samples = 200
        samples = []
        for k in range(0, n_samples):
            (sample, N, C) = self.sampler.sample(self.click_distr, letter, i_n_rejection_samples=1E3)
            samples.append( np.array(sample) )
        gauss_mean = self.click_distr.getLetterLocation( letter, i_with_delay=False)
        samples = np.array(samples) 
        print "****************************************************************"
        print "Values without subtracting the means"
        print "****************************************************************"
        print "sample mean : ", np.mean(samples, axis=0)
        print "sample std :", np.std(samples, axis=0)
        print "gauss_mean :", gauss_mean
        print "****************************************************************"
        print "Values after subtracting the means"
        print "****************************************************************"
        samples -=  np.atleast_2d(gauss_mean)
        print "sample mean : ", np.mean(samples, axis=0), " total = ", np.mean(samples.flatten())
        print "sample std :", np.std(samples, axis=0), " total = ", np.std(samples.flatten())
        print "grnd truth mean :", delay
        print "grnd truth std :",  std
        print "****************************************************************"
    
    ########################################################## Display
    
    def dispClickTimes(self, i_click_times, i_sentence, i_print_times, i_draw_click_times, i_save_image):
        """Only part of example in toy example - display the click time (sample) stats"""
        if i_draw_click_times:
            self.disp_settings.newFigure() 
        """Plot the results if 2 clicks were received, along with the average and std of the click times associated with 
        each letter in the alphabet"""
        (means, stds, nsamples_used) = self.clickTimesStats(i_click_times, i_sentence, i_print_times, i_draw_click_times)
        if not i_draw_click_times:
            return
        #Draw the samples for this examples on the 2D Gaussian distribution
        click_distr_disp = ClickDistributionDisplay(self.click_distr)
        click_distr_disp.disp = self.disp_settings
        scores_file = "./click_distr_plots/simul_scores.npy"
        letter_ids_file =  "./click_distr_plots/simul_ids.npy"
        (nsamples, is_gray, is_save, draw_text, i_normalise) = (300, False, i_save_image, True, False)
        click_distr_disp.drawAll2DClicks(letter_ids_file, scores_file, nsamples, is_gray, is_save, draw_text, i_normalise)
        self.plotClickStats(i_sentence, means, stds, nsamples_used)
        
    def plotClickStats(self, i_sentence, i_means, i_stds, i_nsamples):
        """* Plot the mean, std of the 2-sample pairs, as well as the gaussian means and stds of the actual letters
           * Only part of example in toy example 
        """
        #Draw the means and std (two samples)
        idx = np.nonzero( np.abs(i_nsamples) > 1E-3 )[0]
        for letter_idx in idx:
            (m0, m1, r) = ( i_means[letter_idx,0], i_means[letter_idx,1], np.mean( i_stds[letter_idx,:]) )  
            print "letter_idx = ", letter_idx, " m0 = ", m0, " m1 = ", m1, " r = ", r, " stds = ",  i_stds[letter_idx,:]
            self.plot_utils.plot2DCircle(m0, m1, r,  i_color="w", i_plot_r=True,  i_marker='o')
            p.plot(m0, m1, 'yo')
            letter = i_sentence[letter_idx]
            loc_idx =  self.ticker.letter_indices[letter]
            (m0, m1) = ( self.ticker.click_distr.loc[loc_idx,0],  self.ticker.click_distr.loc[loc_idx,1])
            r = self.ticker.click_distr.std
            #self.plot_utils.plot2DCircle(m0, m1, r, i_color="r", i_plot_r=True) 
        
    def clickTimesStats(self, i_click_times, i_sentence, i_print_times, i_draw_click_times):
        """
        * Print samples to std, and plot click times. 
        * Compute the average and std of samples per letter (where 2 samples were taken)
        * Only part of example in toy example 
        """
        if i_print_times:
            print "click times : "
        (means, stds , nsamples) = ( np.zeros([len(i_sentence), 2]), np.zeros([len(i_sentence), 2]) , np.zeros( [len(i_sentence )]))
        for n in range(0,  len(i_click_times)):
            self.dispMsg(i_disp_dots=True, i_msg=( "sample  = "+ str(n)), i_disp_msg=True)
            words = self.phrase_utils.wordsFromSentece( i_sentence )
            cur_idx = -1
            for m in range(0, len(words)):
                word = self.phrase_utils.getWord(words[m])
                for letter_idx in range(0, len(word)):
                    cur_idx += 1
                    letter = word[letter_idx]
                    loc_idx =  self.ticker.letter_indices[letter]
                    loc = np.array(self.ticker.click_distr.loc[loc_idx,:]).flatten()
                    print letter, " cur_idx = ", cur_idx, " letter idx  = ", letter_idx + 1, " of ", len(i_click_times[n][m]),
                    print self.utils.stringVector(loc, i_disp_str=(" loc=" )), " ", 
                    for t in range(0, len( i_click_times[n][m][letter_idx] )):
                        #Load the saved click times without the delay for display purposes only
                        click_times = i_click_times[n][m][letter_idx][t].flatten() - self.click_distr.delay
                        print self.utils.stringVector(click_times, i_disp_str=(" t%d=" % (t+1))),   
                        if len(click_times) == 2: 
                            if i_draw_click_times:
                                p.plot( [click_times[0]], [click_times[1]], 'k+', linewidth=4 ) 
                            means[cur_idx,:] += click_times
                            stds[cur_idx,:] += (click_times**2)
                            nsamples[cur_idx] += 1
                    print " "
        #Normalise the sample stats
        idx = np.nonzero( np.abs(nsamples) > 1E-3 )[0]
        nsamples = nsamples.reshape([len(nsamples), 1])
        means[idx,:] /=  nsamples[idx]
        stds = np.sqrt(  (stds[idx,:] / nsamples[idx]) - (means[idx,:]**2) )
        return (means, stds, nsamples)
    
           
if __name__ ==  "__main__":   
    g = TickerSimulationExamples()
    g.examples()
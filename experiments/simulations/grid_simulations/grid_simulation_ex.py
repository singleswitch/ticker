import numpy as np
import pylab as p
from grid_simulation import GridSimulation
from scipy.special import erfinv, erf

#Should be in grid simulation
from utils import Utils, DispSettings, PhraseUtils
from plot_utils import PlotUtils

#Example of grid simulation
class GridSimulationExamples(GridSimulation):
    ############################################### Init
    def __init__(self):
        GridSimulation.__init__(self)
        #Put back in grid simulation
        self.disp_settings = DispSettings()
        self.disp_settings.setGrid2ProbPlots()
        self.plot_utils =  PlotUtils()
       
    def initWordExampleParams(self):
        n_errors=2
        n_undo=2 
        fr=0.1
        fp_rate=0.01
        scan_delay=1.0
        click_time_delay=0.1 + 0.5*scan_delay
        std=0.1
        n_max=10 #Simulate for T=n_max*(n_rows+n_cols)*len(input_word) scans
        draw_tikz=False
        display=True
        self.debug = True
        self.init(i_scan_delay=scan_delay, i_click_time_delay=click_time_delay) 
        return (n_errors, n_undo, fr, fp_rate, scan_delay, click_time_delay, std, n_max, draw_tikz, display)
    
    def initSentenceExampleParams(self):
        n_errors=2
        n_undo=2 
        fr=0.1
        fp_rate=0.005
        scan_delay=1.0
        click_time_delay=0.1 + 0.5*scan_delay
        std=0.1
        n_max=10 #Simulate for T=n_max*(n_rows+n_cols)*len(input_word) scans
        draw_tikz=False
        display=False
        self.debug = False
        self.init(i_scan_delay=scan_delay, i_click_time_delay=click_time_delay) 
        return (n_errors, n_undo, fr, fp_rate, scan_delay, click_time_delay, std, n_max, draw_tikz, display)
    
    
    ############################################## Main
    
    def examples(self):
        self.simpleGridExample()
        #self.wordExample("standing_")
        ##self.sentenceExample("at_me_.")        
        p.show()
        
    ################################################## Sentence example
       
    def sentenceExample(self, i_sentence):
        params = self.initSentenceExampleParams()
        self.compute(i_sentence, params)
        
    ##################################################### Word example
            
    def wordExample(self, i_input_word):
        """
        * Draw distributions for the number of scans in correct state for a single word
          - On distribution plot: Min number of scans, average and std scan values too.
        * Display the stats for that word
        * Display ball park figures for no noise case 
        """
        self.add_tick = True
        draw_probs = True #Draw the prob picture 
        params = self.initWordExampleParams()
        (results, scan_probs, click_probs, err_probs, wpm_probs)  = self.wordResults(i_input_word, params)
        print "Ball park figures, average values, no noise, letter priors"
        self.getLetterPriors()
        self.avgTextEntryRate(i_uniform_priors=False, i_disp_letter_times=True)
        if not draw_probs:
            return results
        self.utils.savePickle( (scan_probs, click_probs, err_probs, wpm_probs, results),  "tmp.cPickle")
        (scan_probs, click_probs, err_probs, wpm_probs, results) = self.utils.loadPickle("tmp.cPickle") 
        (min_scans, avg_scans, std_scans, min_wpm, avg_wpm, std_wpm, avg_chr_err, std_chr_err, avg_clicks, std_clicks) = results 
        (scans,scans_wpm, clicks, errors) = self.getPossibleResultValues(i_input_word,
            len(scan_probs), len(click_probs), len(err_probs)) 
        #self.displayProbs(scan_probs, click_probs, err_probs)
        self.displayResults(results, True)
        self.plotProbs(scan_probs, scans, avg_scans, std_scans, min_scans, 0.0, '$ \#scans$', '$P( \#scans \mid  \\boldsymbol{\\theta}_{\mathrm{G}})$', "scan_probs")
        self.plotProbs(scan_probs, scans, avg_scans, std_scans, min_scans, 0.0, '$ \#scans$', '$P( \#scans \mid  \\boldsymbol{\\theta}_{\mathrm{G}})$', "scan_probs")
        self.plotProbs(wpm_probs, scans_wpm, avg_wpm, std_wpm, min_wpm, 0.0, '$wpm$', '$P( wpm \mid  \\boldsymbol{\\theta}_{\mathrm{G}})$', "wpm_probs")
        self.plotProbs(click_probs, clicks, avg_clicks, std_clicks, 2.0, 0.0, '$\#clicks$ $\\mathrm{(cpc)}$', '$P( \#clicks \mid \\boldsymbol{\\theta}_{\mathrm{G}})$', "click_probs")
        self.plotProbs(err_probs, errors, avg_chr_err, std_chr_err, 0.0, 0.0, '$\#errors$ $(\%)$',  '$P( \#errors \mid \\boldsymbol{\\theta}_{\mathrm{G}})$', "error_probs")
        return results
   
    def plotProbs(self, i_y, i_x, i_avg, i_std, i_no_err, i_min, i_xlabel, i_ylabel, i_file_name ):
        """Plot the scan probabilities, as an example"""
        #Cumulative prob
        start_index = np.nonzero( i_y > 1E-3 )[0][0] 
        if start_index < 0:
            start_index = 0 
        end_index =  np.nonzero( i_y > 1E-3 )[0][-1]
        y = i_y[start_index:end_index]
        x = i_x[start_index:end_index]
        self.disp_settings.newFigure()
        self.disp_settings.setTexTrue()
        p.plot(x, y, 'k')
        self.plot_utils.plotErrorBar(i_no_err, y, color="r", width=4)
        self.plot_utils.plotErrorBar(i_avg, y, i_std, 'b', 2.0, 25.0,width=4,i_min_x=i_min)
        p.xlabel(i_xlabel)
        p.ylabel(i_ylabel)
        p.axis('tight')
        file_name = self.output_dir + i_file_name 
        self.disp_settings.saveFig(file_name)
        
    def avgTextEntryRate(self, i_uniform_priors=False, i_disp_letter_times=True):
        """Get some idea of the text-entry rate for the no noise case by looking one char at a time, and 
        taking the average over all the characters."""
        expected_time_per_char = 0.0
        grid_config = np.array([['a','b','c','d','_','*'],['e','f','g','h','.','*'],['i','j','k','l','m','n'],['o','p','q','r','s','t'],['u','v','w','x','y','z']])
        for letter in  grid_config.flatten():
            if letter == '*':
                continue
            (row, col) = np.nonzero(grid_config == letter)
            letter_time =  (row + col+ 2)*self.scan_delay 
            if self.add_tick:
                letter_time += (2.0*self.scan_delay)
            if not i_uniform_priors:
                prior = np.float(self.letter_probs[letter])
            else:
                prior = 1.0 / ( len(self.letter_probs.keys()) )
            if i_disp_letter_times:
                print "letter = ",  letter, " row = " , row, " col = ", col, " letter time = ", letter_time,  " seconds, prior = ", prior
            expected_time_per_char += (letter_time*prior)
        print "Total scan_delay per character = %.2f seconds " % self.scan_delay
        print "Expected time per character = %0.3f seconds " % expected_time_per_char 
        print "Text entry rate = %0.3f wpm " %  (60./ (expected_time_per_char * 5.0 ))
        
    def getLetterPriors(self):
        """The letter frequencies from David's book to get some ball part figures of the text entry rates"""
        self.letter_probs = {}
        self.letter_probs['a'] = 0.0575
        self.letter_probs['b'] = 0.0128
        self.letter_probs['c'] = 0.0263
        self.letter_probs['d'] = 0.0285
        self.letter_probs['e']=  0.0913
        self.letter_probs['f'] = 0.0173
        self.letter_probs['g'] = 0.0133
        self.letter_probs['h'] = 0.0313
        self.letter_probs['i'] = 0.0599
        self.letter_probs['j'] = 0.0006
        self.letter_probs['k'] = 0.0084
        self.letter_probs['l'] = 0.0335
        self.letter_probs['m'] = 0.0235
        self.letter_probs['n'] = 0.0596
        self.letter_probs['o'] = 0.0689
        self.letter_probs['p'] = 0.0192
        self.letter_probs['q'] = 0.0008
        self.letter_probs['r'] = 0.0508
        self.letter_probs['s'] = 0.0567
        self.letter_probs['t'] = 0.0706
        self.letter_probs['u'] = 0.0334
        self.letter_probs['v'] = 0.0069
        self.letter_probs['w'] = 0.0119
        self.letter_probs['x'] = 0.0073
        self.letter_probs['y'] = 0.0164
        self.letter_probs['z'] = 0.0007
        self.letter_probs['_'] = 0.1928
        self.letter_probs['.'] = 0.1928/7.0        
        prob_sum = np.sum(self.letter_probs.values())
        for key in self.letter_probs.keys():
            self.letter_probs[key] /= prob_sum
        print "Probs sum = ", np.sum( self.letter_probs.values() )
        
    #################################################### 2x2 Grid Example
    def simpleGridExample(self):
        """A toy example.
        * Construct a very simple 2x2 grid
        * Generate only the probabilities that can be used to compute the error rate and speed with
        * It seems that to compute the cpc would be computationally too expensive.
        * Generate tikz graph of the Markov chain stored as results/simulations/grid1.tikz
        * Compute a predefined sequence to a ground-truth example (also used in paper as a hyothetical seq example).
        """
        config=[['a', '_'],['t', 'D']]
        input_word ="a_" #The word to write
        (n_errors, n_undo, fr, fp_rate, scan_delay, click_time_delay, std, n_max, draw_tikz, display) = self.initWordExampleParams()
        draw_tikz = True
        self.add_tick=True
        click_time_delay = 0.5*scan_delay
        overlap = 0.05 #Use this instead of the default std
        T=50 #The maximum number of scans to use
        test_grnd=True #Test against grnd truth
        #Test seq to evaluate markove chain (example in paper).
        seq = [1,2,5,11,13,14,17,18,11,12,15,16,1,3,21,23,24,43]
        #Compute the standard deviation of the Gaussian, assuming that the click is in the middle of the block
        k = 0.5*scan_delay/np.sqrt(2)
        std = k / erfinv(1.0 -  overlap)
        print "Gaussian std = ", std, " with delay of ", click_time_delay, " and scan delay of " , scan_delay
        #Initialise the parameters and the configuration  
        params = (n_errors, n_undo, fr, fp_rate, scan_delay, click_time_delay, std, n_max, draw_tikz, display)
        self.init(scan_delay, click_time_delay, config)
        max_scans = n_max*len(input_word)*self.n_rows*self.n_cols
        print "Grnd truth word = ", input_word, " max scans = ", max_scans
        #Get the Markov chain structure
        (states, transitions, transition_probs ) = self.getStates(input_word,  params, self.letter_info)
        #Compute the pdfs for the stats
        (scan_probs, click_probs, err_probs, wpm_probs) = self.stateProbs(states, transitions, transition_probs, max_scans,
                    input_word, n_errors, display)
        print "Min scans = ", self.getMinScans(input_word)
        self.displayProbs(scan_probs, click_probs, err_probs)
        print "********************************************************"
        print "testing a simple example, compute the probability of a state sequence"
        self.testSimpleExample(scan_delay, fp_rate, overlap, states, transitions, transition_probs, seq, test_grnd, display)
        
    def testSimpleExample(self, i_scan_delay, i_fp_rate, i_overlap, i_states, i_transitions, i_transition_probs, i_seq, i_test_grnd, i_display):
        if not i_test_grnd:
            return
        (prob, prob_seq) = self.getProbStateSeq(i_states, i_transitions, i_transition_probs, i_seq, i_display)
        print "prob_seq = [", prob_seq
        print "=========================================="
        print "prob of sequence = ", prob
        print "=========================================="
        (p010, p110, p000, p100, p011, p111) = self.getSimpleExampleGrndTruthProbs(i_scan_delay, i_overlap) 
        #Compare prob seq against ground truth
        print "fp rate  = ", self.fp_rate, " fr = ", self.fr, " overlap = ", i_overlap
        grnd_truth = np.array([p010,p100,p111,p100,p011,p011,p011,p011,p000,p110,p000,p110,p110,p110,p110,p000,p110])
        print "len grnd_truth = ", len(grnd_truth), " len prob_seq = ", len(prob_seq)
        dist = grnd_truth - prob_seq
        print "len(grnd_truth) = ", len(grnd_truth), " len(prob_seq) = ", len(prob_seq)
        for n in range(0, len(grnd_truth)):
            print "src = ", i_seq[n], " dest = ", i_seq[n+1], " n = ", n, " ", grnd_truth[n]," ", prob_seq[n], " dist  ", dist[n]
        (states, dest) = self.convertDest(i_states, i_transitions )
        idx = np.nonzero( np.abs(dist) > 1E-5 )[0]
        if len(idx) > 0:
            print "******************************************"
            print "ERRORS" 
            print "******************************************"
            for n in range(0,len(idx)):
                src = i_seq[idx[n]]
                print "n = ", n , " src = ", src,  " dest :", dest[src-1]+1,
                print " probs : ", i_transition_probs[src-1]

    def getSimpleExampleGrndTruthProbs(self, i_scan_delay, i_overlap):
        fp = self.getFalsePositiveProb(self.fp_rate, i_scan_delay)
        p010 = fp*(1- ((1.0-self.fr)*(1.0-i_overlap)))  #intentional click but missed
        p110 = 1- p010                                     #intentional click
        p000 = fp*(1- (0.5*(1-self.fr)*i_overlap))      #intentional miss (did not aim for anything)
        p100 = 1 - p000                                    #non-intentional click
        p011 = fp
        p111 = 1.0 - p011
        print "p010: click, src  eq dest  : ", p010, " p110 = 1-p010 = ", p110
        print "p000: click, src neq dest  : ", p000, " p100 = 1-p000 = ", p100
        print "p011: click, src     dest=0: ", p011, " p111 = 1-p011 = ", p111 
        return (p010, p110, p000, p100, p011, p111)
    
 
        
    
if __name__ ==  "__main__":   
    g = GridSimulationExamples()
    g.examples()

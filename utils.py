
import numpy as np
import time, cPickle, sys
import scipy.stats.distributions as sd
import pylab as p

class WordDict(object):
    ################################ Init Functions    
    def __init__(self, i_file_name):
        self.loadDict(i_file_name)
        self.utils = Utils()
             
    def loadDict(self, i_file_name):
        t_start = time.time()
        file_name = file(i_file_name)
        d = file_name.read()
        file_name.close()
        b = d.split('\n')
        d = np.array([[bb.split()[0]+"_", int(bb.split()[1])] for bb in b if (len(bb)>0) & (bb.find("_")==-1)])
        d = np.vstack((d, np.array(['.', '0']))) 
        self.words = np.array(d[:,0])
        self.word_lengths = np.array([len(word) for word in self.words])
        idx = np.nonzero(self.words == 'the_')[0]  
        probs =  np.float32(np.array(d[:,1]))
        probs[-1] = probs[idx]
        self.log_probs = np.log(probs) - np.log(np.sum(probs))
        print "Loading from file = ", (time.time() - t_start )*1000, "ms"
        print "the_ = ", np.exp(self.log_probs[idx])
        print "sum = ", np.sum(np.exp(self.log_probs))

    def normalise(self, i_log_probs):
        log_max = np.max(i_log_probs)
        log_sum = log_max + np.log(np.sum(np.exp(i_log_probs - log_max)))
        return i_log_probs- log_sum  
          
    def listsToDict(self, i_labels,  i_vals):
            wi = [ [i_labels[n], i_vals[n]] for n in range(0, len(i_vals))]
            return dict(wi) 
    
class Utils():
    def __init__(self):
        self.eps = 1E-6
  
    def expTrick(self, i_log_probs):
            """Compute log(sum(exp(log(x)))
            Input:
            ====== 
                   i_log_probs: * N x D matrix
                                * N is the number of examples
                                * D is the dimensions
                                * All computations are done over D (column space)
            ======
            Output:
            ====== 
                   o_log_sum: * max(i_log_probs) + log(sum(exp(i_log_probs - log_max))
                              * This will be an N x 1 matrix"""
            o_log_sum = np.max( i_log_probs, axis=1).flatten()
            data = (np.exp( i_log_probs.transpose() - o_log_sum ) ).transpose()
            (rows,cols) = np.nonzero(np.isnan(data)  )
            data[rows,cols] = 0
            exp_sum = np.sum(data, axis = 1)
            idx = np.nonzero( exp_sum > 0.)[0]
            if len(idx) > 0:
                exp_sum[idx] = np.log(exp_sum[idx])
                o_log_sum[idx] += exp_sum[idx]
            return o_log_sum    
        
    def loadText(self, i_file):
        file_name = file(i_file)
        data = file_name.read()
        file_name.close()
        return data 
    
    def saveText(self, i_data, i_file):
        file_name = file(i_file, "w")
        file_name.write(i_data)
        file_name.close() 
        
    def loadPickle(self,i_file):
        f = open(i_file, 'rb')
        data = cPickle.load(f)
        f.close()
        return data
    
    def savePickle(self, i_data, i_file):
        f = open(i_file, 'wb')
        data = cPickle.dump(i_data, f)
        f.close()
        
    def likelihoodGauss(self, i_observations, i_std, i_log ):
        if np.abs(i_std) < 1E-3:
            raise ValueError("i_std = %.6f, too small for this application!" % i_std )
        y =  -0.5*(i_observations**2) / (i_std**2)
        log_const =   -0.5*(np.log(2) + np.log(np.pi)) - np.log(i_std) 
        y += log_const
        if not i_log:
            y = np.exp(y)
        return y
    
    def xcombinations(self, items, n):
        if n==0: yield []
        else:
            for i in xrange(len(items)):
                for cc in self.xcombinations(items[:i]+items[i+1:],n-1):
                    yield [items[i]]+cc
    
    def xpermutations(self, items):
        return self.xcombinations(items, len(items))

    ################################ Slower version of exp trick - can sort first before doing it

    def elnsum(self, elnx, elny):
        if np.isinf(elnx) and np.isinf(elny):
            return -np.inf
        if np.isinf(elnx):
            return elny
        if np.isinf(elny):
            return elnx
        if elnx > elny:
            return elnx + np.log( 1.0 + np.exp( elny - elnx ))
        return elny + np.log( 1.0 + np.exp( elnx - elny ))

    def elnprod(self, elnx, elny):
        if np.isinf(elnx) or np.isinf(elny):
            return -np.inf
        return elnx + elny
    
    ################################### Display

    def stringVector(self, i_vec, i_type="%.4f", i_disp_str=""):
        disp_str=str(i_disp_str)
        if len(i_vec) < 1:
            return disp_str + "[]"
        disp_str += "["
        for (n, val) in enumerate(i_vec):
            if n > 0:
                disp_str += " "
            disp_str += (i_type % (val) )
        disp_str = list(disp_str)
        disp_str.append("]")
        return ''.join(disp_str)
    
    def printMatrix(self, i_data, i_var_name=None, i_file_name=None, i_precision="%.7f" ):
        if i_file_name is not None:
            fout = open( i_file_name, "w")
        else:
            fout = sys.stdout 
        spaces_str = "" 
        if i_var_name is not None:
            fout.write( "%s=" %i_var_name )
            for n in range(0, len(i_var_name)+2):
                spaces_str += " "
        for t in range(0, len(i_data)):
            if t > 0:
                fout.write(  spaces_str )
                fout.write( "[" )
            else:
                fout.write( "[[" )
            for c in range(0, len(i_data[t])):
                fout.write( i_precision % i_data[t][c] )
                if c == (len(i_data[t])-1):
                    if t == (len(i_data) - 1):
                        fout.write( "]]")
                    else:
                        fout.write( "]")
                else:
                    fout.write( ",")
            fout.write("\n")
        if i_file_name is not None:
            fout.close()
            
    def printParams(self, i_params, i_heading_str=""):
        (delay, std, fr, fp_rate, T) = i_params
        fp_min = fp_rate * 60.0
        disp_str = "delay=%.4f, std=%.4f, fr=%.4f, fp=%.4f/s = %.4f/min, T=%.4f" % (delay,std,fr,fp_rate,fp_min, T)
        print i_heading_str + disp_str 
        
    def getPrintParams(self, i_params, i_T):
        print_params = list(i_params)
        print_params.append(i_T)
        return tuple(print_params)
    
    def dispMsg(self,  i_disp_dots, i_msg=None, i_disp_msg=True):
        if i_disp_dots and i_disp_msg:
            print "*******************************************************************************************************"
        if (i_msg is not None) and (i_disp_msg):
            print i_msg
            
    def dispResults(self, i_r):
        (avg_cpc, std_cpc, avg_wpm, std_wpm, avg_err_rate, std_err_rate) = i_r
        if avg_cpc is not None:
            print "avg cpc = %1.2f, std cpc = %1.2f,  " % (avg_cpc, std_cpc),        
        print " avg wpm = %1.2f, std wpm = %1.2f, " % (avg_wpm, std_wpm),
        print " avg err = %3.2f, std err = %3.2f" % (avg_err_rate, std_err_rate)


class PylabSettings():
    
    def __init__(self):
        self.params = { u'axes.labelsize':10, u'xtick.labelsize': 10,u'ytick.labelsize': 10, 
                u'axes.titlesize':10, u'text.usetex': False,  u'font.size': 10 }
        p.rcParams.update(self.params)
        
    ####################### Generic
               
    def newFigure(self):
        fig = p.figure(facecolor='w')
        self.newSubFigure()
        return fig
    
    def newSubFigure(self):
        p.hold(True)
        p.grid('on')
        
    def getString(self, i_str):
        if self.params['text.usetex']:
            o_str = "$" + i_str + "$"
            #_str.replace("\", "\\")
            o_str.replace("_", "\_")
        else:
            o_str = str(i_str)
            o_str.replace("_", "")
            o_str.replace("{", "")
            o_str.replace("}", "")
        #print "i_str = ", i_str, " o_str = ", o_str
        return o_str
    
    def saveFig(self, i_file_name, i_save_eps=True):
        pdf_fname = i_file_name + ".pdf"
        eps_fname = i_file_name + ".eps"
        #pdf_crop_fname = i_file_name + 
        p.savefig(pdf_fname, dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', papertype='a4', format=None,
            transparent=False, bbox_inches="tight", pad_inches=0.05)
        if i_save_eps:
            p.savefig(eps_fname, dpi=None, facecolor='w', edgecolor='w',
                orientation='portrait', papertype='a4', format=None,
                transparent=False, bbox_inches="tight", pad_inches=0.05)

#The settings for different experiments
class DispSettings(PylabSettings):
    def __init__(self):
        PylabSettings.__init__(self)


    ################################ Set
    
    def setTexTrue(self):
        self.params[u'text.usetex'] = True
        p.rcParams.update(self.params)
        p.rcParams[u'text.latex.preamble']=[r"\usepackage{amsmath}"]
        
    def setTexFalse(self):
        self.params[u'text.usetex'] = False
        p.rcParams.update(self.params)
            
    ################################ Settings for different plots

 
    def settingsNormaliseTests(self): 
        self.params = { u'axes.labelsize':10, u'xtick.labelsize': 10,u'ytick.labelsize': 10, 
                    u'axes.titlesize':10, u'text.usetex': False }
        p.rcParams.update(self.params)
        
    def setClickDistributionDisplay1D(self):
        self.params = {u'axes.labelsize':17,u'xtick.labelsize': 17,u'ytick.labelsize': 17,
            u'text.usetex': True, u'font.size': 17, u'alphabet_size':15 }
        p.rcParams.update(self.params)
        p.rcParams[u'text.latex.preamble']=[r"\usepackage{amsmath}"]
        
    def setClickDistributionSeqDisplay1D(self):
        self.params = {'axes.labelsize':10,'xtick.labelsize': 10,'ytick.labelsize': 8, 'text.usetex': True,
                  'font.size': 10}
        p.rcParams.update(self.params)
        p.rcParams[u'text.latex.preamble']=[r"\usepackage{amsmath}"]
        
    def setClickDistributionDisplay2DTarget(self):
        """This should be the same as in ticker_click_noise_numerical"""
        self.params = {u'axes.labelsize':35, u'xtick.labelsize': 35,u'ytick.labelsize': 35, 
            u'text.usetex': True, u'font.size': 35, u'text_font':30,
            u'font.family': u'sans-serif'}
        p.rcParams.update(self.params)
        p.rcParams[u'text.latex.preamble']=[r"\usepackage{amsmath}"]
        
    def setLetterConfigDisp(self):
        self.params = {u'axes.labelsize':12,u'xtick.labelsize': 12,u'ytick.labelsize': 12, u'text.usetex': True,
                  u'font.size': 12}
        p.rcParams.update(self.params)
        p.rcParams[u'text.latex.preamble']=[r"\usepackage{amsmath}"]

    def setGrid2ProbPlots(self):
        self.params = {u'axes.labelsize':40, u'xtick.labelsize': 35, u'ytick.labelsize': 35, u'text.usetex': False,
                  u'font.size': 35}
        p.rcParams.update(self.params)
        
    def setBoxPlotDisp(self):
        self.params = { u'axes.labelsize':12, u'xtick.labelsize': 12, u'ytick.labelsize': 12, u'text.usetex': False,
                  u'font.size': 12}
        p.rcParams.update(self.params)
        
    def setClickDataDisp(self):
        self.params = { u'axes.labelsize':8, u'xtick.labelsize':8, u'ytick.labelsize':8, u'text.usetex': False,
                  u'font.size': 8}
        p.rcParams.update(self.params) 
        
    def setAudioUserTrialBoxPlotDisp(self): 
        self.params = { u'axes.labelsize':12, u'xtick.labelsize':12,
            u'ytick.labelsize':12, u'text.usetex': False, u'font.size': 12}
        p.rcParams.update(self.params) 
    
    def setDrawBinarySequences(self):
        self.params = { u'axes.labelsize':10, u'xtick.labelsize': 10, u'ytick.labelsize': 10, 
                u'axes.titlesize':10, u'text.usetex': False,  u'font.size': 8, u'font.family':u'monospace',
                 u'font.weight':u'bold' }
        p.rcParams.update(self.params) 
        
    def setDispSoundAmplitudes(self):
        self.params = {'font.size': 16, 
                  'axes.labelsize': 14, 
                 'xtick.labelsize': 14,
                 'ytick.labelsize': 14,
                }
        p.rcParams.update(self.params) 
        
        
    def setAlphabetConfigDisplay(self):
        """This should be the same as in ticker_click_noise_numerical"""
        self.params = {u'axes.labelsize':45,u'xtick.labelsize': 40, u'ytick.labelsize': 40, 
            u'text.usetex': True, u'font.size': 40, u'text_font':40,
            u'font.family': u'sans-serif', u'font.weight' : 'bold'}
        p.rcParams.update(self.params)
        p.rcParams[u'text.latex.preamble']=[r"\usepackage{amsmath}"]


    def setSimulationPlotsDisp(self):
        self.params = {u'axes.labelsize':45, u'xtick.labelsize': 45, u'ytick.labelsize': 45, u'text.usetex': True,
                  u'font.size': 45}
        p.rcParams.update(self.params)
        
    def setAudioUserTrialBoxPlotDispGrid(self): 
        self.params = { u'axes.labelsize':20, u'xtick.labelsize':20,
            u'ytick.labelsize':20, u'text.usetex': False, u'font.size': 20}
        p.rcParams.update(self.params) 
        
class PhraseUtils():    
    def getWord(self, i_word):
        word = str(i_word)
        if (not (word == ".")) and (not (word == "")):
            word += "_"
        return word

    def getDispVal(self, i_val, i_format, i_spaces):
        if i_val is None:
            return "{0:{1}}".format( "-",  i_spaces ) + "|"
        return "{0:{1}}".format( i_format % i_val,  i_spaces ) + "|"
    
    def wordsFromSentece(self, i_sentence):
        if i_sentence[-1] == ".":
            words = i_sentence.split('_')
        else:
            words = i_sentence.split('_')[0:-1]
        return words
    
if __name__ ==  "__main__":
    #wd = WordDict("dictionaries/nomon_dict.txt") 
    u = Utils()
    seq = ['0','0','1','1','1']
    a = np.unique( np.array([''.join(p) for p in u.xpermutations(seq)]) )
    print a
    print "Selections of two indices, with no repetitions"
    seq = ['0','1','2','3']
    L = 2 
    a = np.unique(np.array([ ''.join(s[-L:]) for s in u.xpermutations(seq)]))
    print a

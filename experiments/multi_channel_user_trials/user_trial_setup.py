import os, cPickle, sys
sys.path.append("../../")
import numpy as np
from utils import Utils
from click_distr import ClickDistribution

class ExperimentSettingsStudents():
    ########################################################## INIT
    def __init__(self):
        #self.speeds = {'0.12' : 9.0, '0.1': 7.2, '0.08':6.0}
        a0 =[12.5998840332,12.5056018829 ,12.4253900051 ,12.5753979683 ,12.5774991512 ,12.4622721672 , 12.5928509235 , 
             12.4742190838 ,12.5771420002 , 12.704282999 , 12.6643178463 , 12.6442830563 , 12.6142480373 , 12.664236784 , 12.4642250538 , 12.5343000889 , 12.5742650032]
        a1 =[11.1842639446,11.1642529964 ,11.4249501228 ,11.204267025 ,11.1642839909 ,11.3642849922 ,11.2543110847 ,11.2355501652 ,11.255931139 ,11.1446011066 ,11.2042608261 ,11.2955827713 ,11.2342340946 ,11.1246528625 ,11.3155350685 ,11.2342720032 ,11.2775220871 ,11.3642799854 ,
             11.2042489052 ,11.1942811012 ,11.2943029404 ,11.3842530251 ,11.2342700958 ,11.2046570778 ,11.1642529964 ,11.2342498302 ,11.2742788792 ,11.064260006]
        a2= [9.01424598694 , 8.95424795151 ,8.92933797836 ,8.96424818039 ,9.01428890228 ,9.01427698135 ,8.96426820755 ,9.03428697586 ,8.99591588974 ,
             8.92554092407 ,8.99434494972 ,9.05429410934,8.99431991577 ,8.99429893494 ,9.03427886963 ,9.01427912712 ,8.96477913857 ,8.94618606567 ,8.97427511215 ,
             9.01513314247 ,9.03471398354 ,8.96580410004 ,8.96427297592 ,9.03432798386 ,9.01427221298 ,8.96425485611 ,9.07428503036 ,8.99432301521 ,8.99429893494 ,
             8.99427700043 ,9.03635907173 ,8.94426417351 ,8.96965193748 ,8.9955201149 ,8.93485689163 ,9.01787114143 ,9.05427789688]
        print "0.12: mean = ", np.mean(np.array(a0)), " std = ", np.std(np.array(a0))
        print "0.1: mean = ", np.mean(np.array(a1)), " std = ", np.std(np.array(a1))
        print "0.08: mean = ", np.mean(np.array(a2)), " std = ", np.std(np.array(a2))
        self.speeds = {'0.12' : 12.57, '0.1': 11.24, '0.08':  9.0 }
        self.speeds_ids = {'0.12' : "S", '0.10': "M", '0.08': "F" }
        self.speed_order = ['0.12','0.10','0.08']
        self.wpm = {}
        self.word_length = 5.0
        for key in self.speeds:
            self.wpm[key] = 1.0 / (self.speeds[key]*self.word_length / 60.0)
        print "Theoretical words per minute = ", self.wpm
        #Delay = -0.2, std=0.1, thresh=0.9 were used in old experiment 
        self.std = 0.1
        self.delay = -0.2 
        self.learning_rate = 0.3
        self.thresh = 0.9
        self.fp_rate = 0.008
        self.fr = 0.05
        self.utils = Utils()
        self.click_distr = ClickDistribution()
        self.n_repeat = 2#Number of times a word can be repeated 
        #Debug
        self.disp=False
    
    def resetClickDistr(self, i_n_channels, i_letter_group, i_overlap, i_params, i_new_letter_loc):
        self.click_distr.clear()
        self.click_distr.click_times = np.array(i_new_letter_loc)
        (alphabet, letter_idx) = ([],[])
        self.long_alphabet = []
        for (n, letter) in enumerate(i_letter_group):
            if len(alphabet) < 1:
                idx = []
            else:
                idx = np.nonzero( np.array(alphabet) == letter)[0]
            if len(idx) < 1:
                alphabet.append(letter)
                letter_idx.append([n])
            else:
                letter_idx[idx].append(n)
            self.long_alphabet.append(letter)
        self.click_distr.alphabet = list(alphabet) 
        self.click_distr.letter_idx = np.array(letter_idx)
        self.click_distr.loc = np.array( i_new_letter_loc[self.click_distr.letter_idx ] )
        self.click_distr.boundary_delay = 0.0
        self.click_distr.T =  self.click_distr.loc[-1, 1]
        (self.click_distr.delay, self.click_distr.std, self.click_distr.fr, self.click_distr.fp_rate) = i_params
        self.click_distr.learning_rate = self.learning_rate
        self.click_distr.initHistogram() 
        
    ############################################################### Get
             
    def getParams(self):
        return (self.delay, self.std, self.fr, self.fp_rate)
 

if __name__ ==  "__main__":
    s = ExperimentSettingsStudents()
    
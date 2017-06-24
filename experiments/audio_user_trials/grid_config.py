
import cPickle, pylab, os, sys
import numpy as np
from PyQt4 import QtCore, QtGui
sys.path.append("../../")   
from utils import Utils
 
class FileLoader():
    def __init__(self , i_root_dir):
        self.root_dir =  i_root_dir
         
    def setRootDir( self,  i_root_dir):
        self.root_dir =  i_root_dir
        
    def getRootDir(self):
        return self.root_dir
    
class ChannelConfig(FileLoader):
    """Input:
        * i_root_dir: Root dir is the root directory of all the lookup tables
          the directories voice_recordings and config are in this directory
        * i_scan_delay: scan delay in seconds"""
                         
    def __init__(self , i_scan_delay, i_root_dir="./", i_nticks=1, i_display=False):
        self.utils = Utils()
        self.display = i_display
        self.nchannels = 1 
        FileLoader.__init__(self, i_root_dir)
        self.__setRowDir()
        self.alphabet = AlphabetLoader(self.getConfigDir())
        self.letter_times = AlphabetTimes(i_nticks, self.getConfigDir(), i_display=i_display)
        self.alphabet.load(self.nchannels)
        alphabet = self.alphabet.getAlphabet( i_with_spaces=False)
        alphabet_group =  self.alphabet.getAlphabet( i_with_spaces=True, i_group=True)
        self.letter_times.load( alphabet, i_scan_delay)
        self.scan_delay = i_scan_delay
        self.config = {}
        self.config_seq = np.array(['a','e','i','o','u'])
        self.config['a'] = 'abcd_$'
        self.config['e'] = 'efgh.'
        self.config['i'] = 'ijklmn'
        self.config['o'] = 'opqrst'
        self.config['u'] = 'uvwxyz'
        
    ########################## Main functions 
    
    def setLetterTimes(self, i_times):
        #The times at which the user is expected to click can be set manually
        self.letter_times = i_times
    
    def setRowScan(self):  
        self.__setRowDir()
        self.__setConfigDir()
        self.row_mode = True
        
    def setColScan(self, i_id ):
        #Configuration directories - change all the time with Grid 
        idx = np.nonzero(self.config_seq == i_id)[0][0] + 1
        self.config_dir = "config/col_scan_" + str(idx) + "/channels"
        self.alphabet_dir = "alphabet_col_scan_" + str(idx) + "/"
        self.row_id = i_id
        self.row_mode = False
        self.__setConfigDir()
        
    def __setRowDir(self):
        self.config_dir = "config/row_scan/channels"  #Alphabet configuration file        
        self.alphabet_dir = "alphabet_row_scan/"      #Voice recordings of alphabet file x
        
    def __setConfigDir(self):
        self.alphabet.setRootDir(self.getConfigDir())
        self.alphabet.load(self.nchannels)
        self.setScanDelay(self.scan_delay)
    
    def setScanDelay(self, i_scan_delay):
        self.scan_delay = i_scan_delay 
        self.letter_times.setRootDir(self.getConfigDir())
        alphabet = self.alphabet.getAlphabet( i_with_spaces=False)
        alphabet_group =  self.alphabet.getAlphabet( i_with_spaces=True, i_group=True) 
        self.letter_times.load(alphabet, self.scan_delay)
        
    ######################### Get functions
     
    def getReadDelay(self, i_extra_wait_time=0):
        return self.letter_times.sound_times[-1,-1] + i_extra_wait_time
      
    def getFileLength(self):
        return self.scan_delay
 
    def getConfigDir(self):
        return self.getRootDir() + self.config_dir + str(self.nchannels) + "/"
    
    def getChannels(self):
        return self.nchannels
    
    def getAlphabetLoader(self):
        """The alphabet loader contains many representations of the same alphabet"""
        return self.alphabet
    
    def getNumberOfPreTicks(self):
        """The number of ticks before starting to read the alphabet sequence"""
        return self.letter_times.n_ticks
    
    def getSoundTimes(self):
        """Return the start/end times of all the letters in the alphabet"""
        return self.letter_times.sound_times
    
    def getClickTimes(self):
        """Return the times where the user is expected to click"""
        return self.letter_times.click_times
  
        
class AlphabetTimes(FileLoader):
    def __init__(self, i_nticks=2, i_dir="./", i_display=False):
        self.utils = Utils()
        FileLoader.__init__(self, i_dir)
        #The start and end times of the sound recordings
        self.sound_times = None
        #The estimated times of where someone clicks 
        self.display = i_display
        #The number of tick sounds before sounds file
        self.n_ticks = i_nticks
    
    def load(self, i_alphabet, i_scan_delay):
        #Initialise click times to be equally spaced in long sequence (should stay like this for 1 channel)
        self.initClickTimes(i_alphabet, i_scan_delay)
        #Add the times of the click sounds
        avg_scan_delay = i_scan_delay
        tmp_sound_times = np.array(self.sound_times)
        self.sound_times = np.zeros([self.sound_times.shape[0]+self.n_ticks, 2])
        (start_time, end_time) = (0.0, 0.0)
        for n in range(0, self.n_ticks):
            end_time = start_time + avg_scan_delay
            self.sound_times[n,0] = start_time
            self.sound_times[n,1] = end_time
            start_time = end_time
        self.sound_times[self.n_ticks:, :] = np.array(tmp_sound_times + end_time)
        if self.display:
            for n in range(0, self.n_ticks):
                print "n = ", n, " letter= *  times=", self.sound_times[n,:],
                print " file length=",  i_scan_delay 
            for n in range(self.n_ticks, self.sound_times.shape[0]):
                print "n = ", n, " letter=", i_alphabet[n-self.n_ticks], " times=", self.sound_times[n,:],
                print " scan delay=",  i_scan_delay
                 
    def initClickTimes(self, i_alphabet, i_scan_delay):
        self.sound_times = np.zeros([len(i_alphabet), 2])
        self.click_times = np.zeros(len(i_alphabet))
        start_time = 0.0 
        for n in range(0, len(i_alphabet)):
            end_time =  start_time + i_scan_delay
            self.sound_times[n,0] = start_time
            self.sound_times[n,1] = end_time 
            start_time = end_time   
            
class AlphabetLoader(FileLoader):
    """This class contains all the loading functions associated with loading the alphabet, and configuring it for multiple channels usage
       Input: 
            * The setChannels functions is expected to be called to change the configuration
            * Otherwise the get functions should be called for different representations of the same alphabet."""
    
    ###################################### Init functions
    def __init__(self, i_dir="./"):
        FileLoader.__init__(self, i_dir)

    ##################################### Load the alphabet
    
    def load(self, i_nchannels):
        self.nchannels = i_nchannels
        file_name = self.getRootDir() + "alphabet.txt"
        file_name = file(file_name)
        alphabet = file_name.read()
        file_name.close()
        alphabet = alphabet.split('\n')[0]
        alphabet = alphabet.split(" ")[0]
        alphabet = [letter for letter in alphabet if not (letter == '') ]
        array_alphabet = np.array(alphabet)
        repeat = np.array([len(np.nonzero(array_alphabet == letter)[0]) for letter in alphabet if not( letter == '*') ])
        idx = np.nonzero(repeat == repeat[0])[0]
        if not ( len(idx) == len(repeat) ):
            print "Repeat = ", repeat
            raise ValueError("Error in alphabet, all letters should repeat the same number of times")
        self.repeat = repeat[0]
        self.alphabet = list(alphabet)
        self.alphabet_no_spaces =  self.__getSequenceAlphabet(self.alphabet)
        alphabet_len = len(self.alphabet) /self.repeat
        self.unique_alphabet = list( self.alphabet[0:alphabet_len])
        self.unique_alphabet_no_spaces =  self.__getSequenceAlphabet(self.unique_alphabet)
        
    ##################################### Get functions 
    
    def getLetterPositions(self):
        """Extract the indices where each letter in unique alphabet occur in long alphabet sequence"""
        alphabet = np.array(self.alphabet_no_spaces )
        return np.array([ np.nonzero(alphabet == letter )[0] for letter in  self.unique_alphabet_no_spaces ])
    
    def getAlphabetRepeat(self):
        """Return the number of times the alphabet is repeated"""
        return self.repeat
    
    def getAlphabet(self, i_with_spaces=True, i_group=False):
        if  i_group:
            return self.__getGrouping(self.alphabet, i_with_spaces)
        if i_with_spaces: 
            return self.alphabet
        return self.alphabet_no_spaces
 
    def getUniqueAlphabet(self, i_with_spaces=True, i_group=False):
        if  i_group:            
             return self.__getGrouping( self.unique_alphabet, i_with_spaces)
        if i_with_spaces: 
            return  self.unique_alphabet
        return self.unique_alphabet_no_spaces
    
    ##################################### Private functions
    
    def __getGrouping(self, i_alphabet, i_with_spaces):
        alphabet = np.array(i_alphabet)
        o_alphabet = [ alphabet[range(n,len(alphabet),self.nchannels)] for n in range(0,self.nchannels)]
        if i_with_spaces:
            return o_alphabet
        for n in range(0, len(o_alphabet)):
            idx = np.nonzero( np.logical_not( o_alphabet[n] =='*'))[0]
            o_alphabet[n] = np.array(o_alphabet[n][idx])
        return o_alphabet
    
    def __getSequenceAlphabet(self, i_alphabet):
        #Return the alphabet in sequence without the spaces
        return [letter for letter in i_alphabet if  not letter == '*']
 
class ConfigExamples():
     
    def __init__(self):
        cur_dir = os.path.dirname(os.path.abspath(__file__)) + "/grid_config/"
        self.channel_config = ChannelConfig(i_scan_delay=1.0, i_root_dir=cur_dir, i_display=True)
        for letter_id in self.channel_config.config_seq:
            print "---------------------------------------------------------------"
            self.channel_config.setColScan(letter_id)

    def minScans(self, i_word):
        print "Computing min scans for ", i_word
        ch_seq = np.array(self.channel_config.config_seq)
        (n_scans, n_total) = (0,0)
        max_row_skips = len(ch_seq)+self.channel_config.getNumberOfPreTicks()
        for letter in i_word:
            for (row, row_letter) in enumerate(ch_seq):
                col_seq = np.array(list(self.channel_config.config[row_letter]))
                idx = np.nonzero(col_seq == letter)[0]
                if len(idx) > 0:
                    n_ticks = self.channel_config.getNumberOfPreTicks()
                    n_rows = n_ticks + row + 1
                    n_cols = n_ticks + idx + 1
                    print "letter = ", letter, " n_rows = ", n_rows, " n_cols = ", n_cols
                    n_total += (n_rows + n_cols) 
        print "================================================="
        print "Total scans = ", n_total
        print "Max row skips = ", max_row_skips

if __name__ ==  "__main__":
    ex = ConfigExamples()
    ex.minScans("enjoyed_")

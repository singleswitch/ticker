
import numpy as np
import cPickle, pylab
from PyQt4 import QtCore, QtGui
import pylab as p #Used for plotting
from utils import Utils
"""* This file contains everything that has to do with the stereo audio configurations of ticker 
    that can be loaded from a file/lookup table, including:
    (a) The alphabet
    (b) The estimated click times.
    (c) Letter start/end times
    (d) Channel names to make it easier for the user to direct his/her attention"""

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
        * i_nchannels: The number of channels
        * i_file_length: TODO - document this parameter
        * i_sound_overlap: A value between 0 and 1, specifying how much successive
                           sounds should overlap"""
                         
    def __init__(self, i_nchannels, i_sound_overlap , i_file_length, i_root_dir="./", i_display=False):
        self.utils = Utils()
        self.display = i_display
        self.config_dir =  "config/channels" 
        self.nchannels = i_nchannels
        self.overlap = i_sound_overlap
        FileLoader.__init__(self, i_root_dir)
        self.alphabet = AlphabetLoader(self.getConfigDir())
        self.letter_times = AlphabetTimes(self.getConfigDir(), i_display=i_display)
        self.channel_names  = ChannelNames()
        self.alphabet.load(self.nchannels)
        alphabet = self.alphabet.getAlphabet( i_with_spaces=False)
        alphabet_group =  self.alphabet.getAlphabet( i_with_spaces=True, i_group=True)
        self.letter_times.load(self.nchannels, self.overlap, alphabet, alphabet_group, i_file_length)
        self.file_length = i_file_length
       
    ########################## Main functions 
    
    def setLetterTimes(self, i_times):
        #The times at which the user is expected to click can be set manually
        self.letter_times = i_times

    def setConfig( self,  i_nchannels, i_sound_overlap, i_file_length):
        self.overlap = i_sound_overlap
        self.setChannels(i_nchannels, i_file_length)
            
    def setChannels(self, i_nchannels,  i_file_length):
        if (i_nchannels == self.nchannels) and (i_file_length == self.file_length):
            return
        self.nchannels = i_nchannels
        self.alphabet.setRootDir(self.getConfigDir())
        self.alphabet.load(self.nchannels)
        self.setOverlap(self.overlap, i_file_length)
    
    def setOverlap(self, i_sound_overlap, i_file_length):
        self.file_length = i_file_length
        self.overlap = i_sound_overlap
        self.letter_times.setRootDir(self.getConfigDir())
        alphabet = self.alphabet.getAlphabet( i_with_spaces=False)
        alphabet_group =  self.alphabet.getAlphabet( i_with_spaces=True, i_group=True)
        self.letter_times.load(self.nchannels, self.overlap, alphabet, alphabet_group, self.file_length)
    
    ######################### Get functions
    
    def getFileLength(self):
        return self.file_length
 
    def getConfigDir(self):
        return self.getRootDir() + self.config_dir + str(self.nchannels) + "/"
    
    def getOverlap(self):
        return self.overlap
    
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
     
    def getChannelNames(self):
        """Return the names associated with the current channel"""
        return self.channel_names.channel_names[ self.nchannels -1]
   
    ######################  Display functions
    """2D Plot functions for the current configuration.
       The configuration is specified by the number of channels and the sound overlap.
       This plot is only for a letter sequence where each leter is repeated exactly twice"""
    def plotClickTimes(self):
        """* Plot how far the click times are from each other."""
        params = {'axes.labelsize':12,'xtick.labelsize': 12,'ytick.labelsize': 12 }
        p.rcParams.update(params)
        alphabet = np.array(self.alphabet.getAlphabet( i_with_spaces=False))
        alphabet_unique = self.alphabet.getUniqueAlphabet( i_with_spaces=False)
        click_times = self.getClickTimes()
        sound_times = self.getSoundTimes()
        idx = self.alphabet.getLetterPositions()
        pos = click_times[idx]
        p.plot(pos[:,0], pos[:,1], 'ko', alpha=0.3, linewidth=2)
        p.grid('on')
        for n in range(0, pos.shape[0]):
            p.text(pos[n,0], pos[n,1], alphabet[n], fontsize=15)
        p.xlabel('Letter click times, first repetition (seconds)')
        p.ylabel('Letter click times, second repetition (seconds)') 
      
class AlphabetTimes(FileLoader):
    def __init__(self, i_dir="./", i_display=False):
        self.utils = Utils()
        FileLoader.__init__(self, i_dir)
        #The start and end times of the sound recordings
        self.sound_times = None
        #The estimated times of where someone clicks
        self.click_times = None
        self.display = i_display
        #The number of tick sounds before sounds file
        self.n_ticks = 2
    
    def load(self, i_nchannels, i_overlap , i_alphabet, i_alphabet_group, i_file_length):
        #Initialise click times to be equally spaced in long sequence (should stay like this for 1 channel)
        self.initClickTimes(i_alphabet_group, i_alphabet, i_nchannels, i_file_length, i_overlap)
        #For more than one channel, adjust so that times are equally spaced within the channel
        #Only implemented for 5 channels at this point
        if self.display:
            print "Alphabet groups:"
            print np.array(i_alphabet_group)
            print "Init deltas"
            self.printDeltas(i_nchannels, i_alphabet_group )
        avg_file_length = i_file_length*(1.0 - i_overlap)*i_nchannels
        #How much should be added to each letter
        if i_nchannels == 5:
            file_lengths = self.getNewFileLengths(np.array(i_alphabet_group), i_file_length)
            #Adjust the click times
            self.initClickTimes(i_alphabet_group, i_alphabet, i_nchannels, file_lengths, i_overlap)
        elif i_nchannels == 3:
            skipped_length = 2.0*i_file_length*(1.0 - i_overlap)
            self.sound_times[-1] += skipped_length 
            self.click_times[-1] += skipped_length
                    
        #Add the times of the click sounds
        tmp_sound_times = np.array(self.sound_times)
        self.sound_times = np.zeros([self.sound_times.shape[0]+self.n_ticks, 2])
        (start_time, end_time) = (0.0, 0.0)
        for n in range(0, self.n_ticks):
            end_time = start_time + avg_file_length
            self.sound_times[n,0] = start_time
            self.sound_times[n,1] = end_time
            start_time = end_time
        self.sound_times[self.n_ticks:, :] = np.array(tmp_sound_times + end_time)
        self.click_times += end_time
        if self.display:
            print "Final deltas"
            self.printDeltas(i_nchannels, i_alphabet_group )
            print "Final letter times"
            print self.sound_times
            print "Click time deltas"
            print self.click_times[1:] - self.click_times[0:-1]
            print "Total times = ", np.sum(self.click_times[1:] - self.click_times[0:-1])

    def initClickTimes(self,  i_alphabet_group, i_alphabet, i_nchannels, i_file_lengths, i_overlap):
        #letter_offsets = getattr(self, "channel" + str(i_nchannels) )()
        #start_time + letter_offsets[letter]
        self.sound_times = np.zeros([len(i_alphabet), 2])
        self.click_times = np.zeros(len(i_alphabet))
        cnt = 0
        start_time = 0.0
        max_cols = len(i_alphabet_group[0])
        if self.display:
            print "Calling adjust timings "
        for n in range(0,  max_cols ):
            for channel in range(0, i_nchannels):
                if i_alphabet_group[channel][n] == "*":
                    continue
                if np.ndim(i_file_lengths) < 1:
                    file_length = i_file_lengths
                else:
                    if n > (len(i_file_lengths[channel]) - 1):
                        continue
                    file_length = i_file_lengths[channel][n]
                end_time =  start_time + file_length*(1.0 - i_overlap)
                self.sound_times[cnt,0] = start_time
                self.sound_times[cnt,1] = end_time
                self.click_times[cnt] = start_time
                if self.display:
                    print i_alphabet[cnt], " ", cnt, " sound=", self.sound_times[cnt,:], " click=",  self.click_times[cnt], 
                    print " channel = ", channel, " n = ", n, " new file length = ",  file_length*(1.0 - i_overlap)
                cnt += 1
                start_time = end_time   
     
    ################################################### Hand Crafted timings

    def getNewFileLengths(self, i_alphabet_group, i_file_length ):
        #Calculate how much time should be added to each letter in group format 
        #(with no "*" in order to make clicks sound equally distributed within the channel
        file_lengths = i_file_length * np.ones(i_alphabet_group.shape)
        [row, col] = np.nonzero( i_alphabet_group == "*")
        file_lengths[row, col] = 0.0
        n_channels = i_alphabet_group.shape[0]
        for n in range(0,i_alphabet_group.shape[1]): 
            idx = np.nonzero( np.logical_not(i_alphabet_group[:,n] == "*") )[0]
            if len(idx) < 1:
                continue
            idx = np.sort(idx)
            n_col = float(len(idx))
            #There are gaps
            if not (n_col == n_channels):
                #The gaps at the end of the sequences
                if n == (i_alphabet_group.shape[1] - 1):
                    #If only ones at the end are skipped continue
                    #As all deltas should be correct, this include an utterance of one in the last column
                    if (max(idx)+1) == n_col:
                        continue
                    #Absorb the delay before the first letter in last column by most of the previous letters in the alphabet
                    if min(idx) > 0:
                        gap_length =  float(i_file_length * min(idx)) 
                        idx_prev_channels = min(idx) + 2
                        prev_channels = n_channels - idx_prev_channels 
                        file_lengths[range(idx_prev_channels,n_channels-1),-2] += (0.35*gap_length / prev_channels)
                        #Let the last elemet in the fore last column absorb half of the delay, as the max per 
                        #cell error is lower
                        file_lengths[-1,-2] += (0.65*gap_length)
                        #If there is only one gap then only the previous case (min(idx) > 0) has to be covered
                        if len(idx) == (n_channels -1):
                            continue
                        #If the remaining utterances are all consecutive then all the delays are complete
                        if (max(idx) - min(idx) + 1) == n_col:
                            continue
                    #Don't take skipped gaps at the end of the sequence into account as well as the last uttered letter
                    #Go through the remaining utterances and average the mistakes out
                    all_file_lengths = file_lengths.transpose().flatten()
                    total_lengths = []
                    flattened_idx = (i_alphabet_group.shape[1]-1)*n_channels + idx[-1]
                    prev_lengths = all_file_lengths[(flattened_idx - n_channels):flattened_idx]
                    total_lengths= n_channels*i_file_length - np.sum(prev_lengths)
                    file_lengths[idx[0:-1],-1] += (0.05 + float(total_lengths) / len(idx[0:-1]))
                #The gaps at the beginning of the sequence
                elif n==0:
                    max_sum = float( (n_channels-min(idx))*i_file_length) 
                    channel_sum = max_sum - len(idx)*i_file_length
                    file_lengths[idx[0], n] += (0.6 * channel_sum) 
                    file_lengths[idx[1], n] += (0.3 * channel_sum)
                    file_lengths[idx[2], n] += (0.1 * channel_sum)
                #else:
                #    print "n_col = ", n_col, " n = ", n
                #    raise ValueError("Waiting period not at begin or end!")
        return file_lengths

    #The following functions contain precomputed values of where we expect the person to click
    #For alphabet long (the short alphabet is 400ms per alphabet file)
    def channel1(self):
        g = {}
        g['a']  = 0.54; g['b']  = 0.672 ; g['c']  = 0.574 ; g['d']  = 0.473; g['e']  = 0.559 ;g['f']  = 0.639; g['g']  = 0.672; g['h']  = 0.806 ; g['i']  = 0.55; g['j']  = 0.657;
        g['k']  = 0.664; g['l']  = 0.657; g['m']  = 0.561; g['n']  = 0.74; g['o']  = 0.551; g['p']  = 0.649; g['q']  = 0.75; g['r']  = 0.783; g['s']  = 0.543; 
        g['t']  = 0.523;g['u']  = 0.659; g['v']  = 0.816; g['w']  = 0.779; g['x']  = 0.794; g['y']  = 0.71;g['z']  = 0.71; g['_']  = 0.9; g['.'] = 0.7
        return g 
    
    def channel2(self):
        g = {}
        g['a']  = 0.54; g['b']  = 0.672 ; g['c']  = 0.574 ; g['d']  = 0.473; g['e']  = 0.559 ;g['f']  = 0.639; g['g']  = 0.672; g['h']  = 0.806 ; g['i']  = 0.55; g['j']  = 0.657;
        g['k']  = 0.664; g['l']  = 0.657; g['m']  = 0.561; g['n']  = 0.74; g['o']  = 0.551; g['p']  = 0.649; g['q']  = 0.75; g['r']  = 0.783; g['s']  = 0.543; 
        g['t']  = 0.523;g['u']  = 0.659; g['v']  = 0.816; g['w']  = 0.779; g['x']  = 0.794; g['y']  = 0.71;g['z']  = 0.71; g['_']  = 0.9; g['.'] = 0.7
        return g 
    
    def channel3(self):
        #This is Patrick Tracey Alan
        g = {}
        g['a']  = 0.43; g['b'] =0.42;g['c']= 0.57;g['d']=0.47;g['e']  = 0.56;g['f']=0.64;g['g']=0.672; g['h']=0.785 ; g['i']  = 0.55; 
        g['j']  = 0.52; g['k']=0.56;g['l']  = 0.411; g['m']  = 0.471; g['n']  = 0.58; g['o']=0.43; g['p']=0.6; g['q']= 0.55; g['r']=0.554; 
        g['s']  = 0.5;g['t']  = 0.58; g['u']  = 0.6;  g['v']  = 0.4; g['w']  = 0.6; g['x']  = 0.75; g['y']  = 0.7;g['z']  = 0.7; g['_']  = 0.72; g['.'] = 0.7
        return g 
    
    def channel4(self):
        g = {}
        g['a']  = 0.43; g['b'] =0.42;g['c']= 0.57;g['d']=0.47;g['e']  = 0.56;g['f']=0.64;g['g']=0.672; 
        g['h']  = 0.54 ; g['i']  = 0.4; g['j']  = 0.5; g['k']  = 0.52; g['l']  = 0.4; g['m']  = 0.4; g['n']  = 0.5;
        g['o']=0.43; g['p']=0.6; g['q']= 0.471; g['r']=0.55; g['s']  = 0.56; g['t']  = 0.42; g['u']  = 0.47; 
        g['v']  = 0.4; g['w']  = 0.6; g['x']  = 0.75; g['y']  = 0.7;g['z']  = 0.7; g['_']  = 0.72; g['.'] = 0.7
        return g

    def channel5(self):
        g = {}
        g['a'] = 0.43; g['b'] = 0.42; g['c'] = 0.57; g['d'] = 0.47; g['e'] = 0.34;
        g['f'] = 0.6 ; g['g'] = 0.52; g['h'] = 0.73; g['i'] = 0.56; g['j'] = 0.59; g['k']  = 0.47; 
        g['l'] = 0.4;  g['m'] = 0.4;  g['n'] = 0.5;  g['o'] = 0.5;  g['p'] = 0.45;
        g['q'] = 0.47; g['r'] = 0.55; g['s'] = 0.56; g['t'] = 0.42; g['u'] = 0.47; g['v']  = 0.61;  
        g['w'] = 0.6;  g['x'] = 0.75; g['y'] = 0.7;  g['z'] = 0.7;  g['_'] = 0.72; g['.'] = 0.7
        return g
    
    #################################################################### Display 
    
    def getGroupTimes(self, i_nchannels, i_alphabet_group):
        #Get the click times in channel group format
        group_times = []
        for channel in range(0, i_nchannels):
            group_times.append([])
        (cnt, max_delta) = (0, 0.0)
        for n_letter in range(0, len(i_alphabet_group[0])):
            for channel in range(0, i_nchannels):
                letter = i_alphabet_group[channel][n_letter]
                if letter == '*':
                    group_times[channel].append(0.0)
                else:
                    group_times[channel].append(self.click_times[cnt])
                    cnt += 1
        return np.array(group_times)
    
    def printDeltas(self, i_nchannels, i_alphabet_group ):
        if not self.display:
            return
        group_times = self.getGroupTimes(i_nchannels, i_alphabet_group)
        for channel in range(0, i_nchannels):
            col_cnt = 0
            for (n, letter) in enumerate(i_alphabet_group[channel]):
                if letter == "*":
                    print "-.-- ", 
                    continue
                if col_cnt == 0:
                    col_cnt += 1
                    prev_time = group_times[channel,n]
                    continue
                cur_time = group_times[channel,n] 
                delta = cur_time - prev_time
                prev_time = cur_time
                print ("%.2f" % delta), " ",  
            print ""
            
class AlphabetLoader(FileLoader):
    """This class contains all the loading functions associated with loading the alphabet, and configuring it for multiple channels usage
       Input: 
            * The setChannels functions is expected to be called to change the configuration
            * Otherwise the get functions should be called for different representations of the same alphabet."""
    
    ###################################### Init functions
    def __init__(self, i_dir="./"):
        FileLoader.__init__(self, i_dir)
        self.utils = Utils()

    ##################################### Load the alphabet
    
    def load(self, i_nchannels):
        self.nchannels = i_nchannels
        alphabet = self.utils.loadText( self.getRootDir() + "alphabet.txt")
        self.setAlphabet(alphabet)
    
    ##################################### Set
    
    def setAlphabet(self, i_alphabet):
        alphabet = i_alphabet.split('\n')[0]
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
            return  self.alphabet
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
 
class ChannelNames():
    def __init__(self):
        self.channel_names= [  [["Cheerful Charlie"]],  
                               [["Cheerful Charlie"],["Cheerful Charlie"]],
                               [["Cheerful Charlie"],["Melodic Mary"], ["Precise Pete"]],
                               [["Cheerful Charlie"],["Baritone Bob"], ["Melodic Mary"], ["Precise Pete"]],
                               [["Cheerful Charlie"],["Sad Sandy"], ["Baritone Bob"], ["Melodic Mary"],["Precise Pete"]] ] 
  
if __name__ ==  "__main__":
    channel_config = ChannelConfig(i_nchannels=5, i_sound_overlap=0.8, i_file_length=0.21, i_display=True)
    print "LETTER TIME SIZE = ", channel_config.getSoundTimes().shape
    p.figure(facecolor='w');  
    channel_config.plotClickTimes(); p.axis('equal')
    p.show()  
    

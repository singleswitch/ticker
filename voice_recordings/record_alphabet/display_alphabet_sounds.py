import numpy as np
import ogg.vorbis
import pylab as p
import scikits.openopt
from scikits import audiolab
import cPickle
import os
import sys
sys.path.append('../../')
from channel_config import ChannelConfig
from PyQt4 import QtCore, QtGui
from utils import Utils
from pylab_settings import DispSettings

class BatchSoundAnalyser(object):
    def __init__(self, i_display=False):
        self.sound_analyser = SoundAnalyser()
        self.times = None
        self.display = i_display       
        self.utils = Utils()
        self.disp = DispSettings()
        self.disp.setDispSoundAmplitudes()
    
    def computeEndOfSounds(self, i_alphabet, i_ground_truth=None, o_file_name=None  ):
        self.times = {}
        for letter in i_alphabet:
            self.times [letter] =  self.sound_analyser.getEndOfSound()
        if o_file_name is not None:
            filename =  i_sound_dir + o_file_name
            print "Saving time information to ", filename 
            self.disp.savePickle(self.times, filename)
    
    def displayChannelSounds(self, i_sound_dir, i_channel_config, i_timing_info, i_plot_axis=True):
        #Get the channel colors (assume 5 for now)
        channel_colours = [ QtGui.QColor("red"), 
                                                    QtGui.QColor("green"),
                                                    QtGui.QColor("blue"), 
                                                    QtGui.QColor("Brown"),
                                                    QtGui.QColor("Purple")]
        colours = [tuple( list( np.array(c.getRgb()[0:-1])/255. )) for c in channel_colours] 
        #Other plot parameters
        self.disp.newFigure()
        n_channels =  len(i_channel_config)
        n_letters_group = len(i_channel_config[0])
        n_letters = n_channels*n_letters_group
        (tmax, tmin, ymax,ymin) = (max(i_timing_info),  0, 1.5, -1.5)
        p.axis( [tmin, tmax, ymin, ymax])
        
        #Initialise all counters
        channel_cnt = -1
        letter_cnt = 0
        time_cnt = 0
        yticks = []
        ytick_labels = []
        for n in range(0, n_letters):
            channel_cnt += 1
            if channel_cnt == n_channels:
                channel_cnt = 0
                letter_cnt += 1
            letter = i_channel_config[channel_cnt][letter_cnt]
            if letter == "*": 
                continue;
            print "channel_cnt = ", channel_cnt , " letter_cnt = ", letter_cnt, 
            print " letter = ", letter 
            sound_file = i_sound_dir + letter + ".wav"
            self.sound_analyser.setSoundFile(sound_file)
            x = i_timing_info[time_cnt]
            y = -(channel_cnt+1) * (ymax - ymin)
            ytick_labels.append(1.0)
            ytick_labels.append(-1.0)
            yticks.append(y+1.0)
            yticks.append(y-1.0)
            time_cnt += 1
            self.sound_analyser.plotSound(  x, y, colours[channel_cnt])
            p.text(  x, y+ymax-0.4, str(letter) )
            
        #Some final plots parameters
        if i_plot_axis:
            p.axis('on')
        else:
            p.axis('off')
        p.grid('on')
        ax = p.gca()
        ax.set_yticks( tuple(yticks) )
        ax.set_yticklabels(  ytick_labels )
        p.xlabel("Time (seconds)")
        p.ylabel("Normalised Amplitude Range (Per Channel)")
        p.axis('tight')
        self.disp.saveFig("alphabet_sounds", i_save_eps=True)
        p.show()
    
    def displaySounds(self,  i_sound_dir,  i_alphabet, i_labels=True, i_ground_truth=None, i_plot_axis=False):
        nrows = np.int32(np.round(np.sqrt(len(i_alphabet))))
        ncols = 1
        while (1):
            if nrows*ncols >= len(i_alphabet):
                break
            ncols += 1
        p.figure(facecolor='w')
        total_diff = 0.0
        for n in range(0, len(i_alphabet)):
            letter = i_alphabet[n]
            sound_file = i_sound_dir + letter + ".ogg"
            self.sound_analyser.setSoundFile(sound_file)
            if self.times is not None:
                self.sound_analyser.setEndOfSound(self.times[letter])
                if self.display:
                    diff = np.abs( self.times[letter] -  i_ground_truth[letter] )
                    total_diff += diff
                    print "Letter: %s, time  = %0.2f, grnd_truth = %0.2f, diff=%0.2f" % (letter, self.times[letter],  i_ground_truth[letter], diff)
            p.subplot(nrows, ncols, n+1)
            self.sound_analyser. plotAmplitude( i_ground_truth=i_ground_truth, i_labels= i_labels)
            p.title(letter)
            if i_plot_axis:
                p.axis('on')
            else:
                p.axis('off')
        if self.display:
            print "Average diff=0.2f" % (total_diff / len(i_alphabet))
        p.show()
        
class SoundAnalyser(object):
    def __init__(self, i_sound_file=None, i_display=False):
        self.rms_thresh = 0.001
        self.display = i_display
        if i_sound_file is not None:
            self.setSoundFile(i_sound_file)
        
    def setSoundFile(self, i_sound_file):
        self.sound_file = i_sound_file
        (   self.sound, self.n_channels, self.sampling_freq, 
            self.file_duration, self.nsamples, self.letter ) = self.soundInfo(i_sound_file)
        self.rms_vals = None
        self.time_end = None#Time when the sound ends
       
    def soundInfo(self, i_sound_file):
        print "Sound file = ", i_sound_file
        (snd, sampling_freq, n_bits) = audiolab.wavread(i_sound_file)
        if snd.ndim == 1:
            n_channels = 1
            channel1 = np.array(snd)
        else:
            n_channels = snd.shape[1]
            channel1 = np.array(snd[:,0])
        n_samples = snd.shape[0]
        file_duration = float(n_samples) / float( sampling_freq )
        letter = str(i_sound_file).split('/')[-1]
        if letter == "..ogg":
            letter = "."
        else:
            letter = letter.split('.')[0]
    
        if self.display:
            print "File info: ", i_sound_file
            print "Number of samples:  " , n_samples
            print "Number of channels: ", n_channels
            print "File duration: ", file_duration, " seconds" 
            print "Currrent letter: ", letter
        return (channel1, n_channels, sampling_freq, file_duration, n_samples, letter)
    
    def rms(self, i_samples=None, i_fast=True):
        if i_fast:
            return  np.sqrt(np.mean(i_samples**2))
        (fourier, n_unique_points) = self.fft(i_samples) 
        return np.sqrt(np.sum(fourier))
        
    def fft(self, i_samples=None):
        if i_samples is None:
            fourier = p.fft(self.sound)  
        else:
            fourier = p.fft(i_samples)
        n_unique_pts = np.ceil((self.nsamples+1)/2.0)
        fourier= np.abs(fourier[0:n_unique_pts])
        fourier = fourier / float(self.nsamples)# scale by the number of points so that
                                            #the magnitude does not depend on the length 
                                            # of the signal or on its sampling frequency  
        fourier= fourier**2  # square it to get the power 

        # multiply by two (see technical document for details)
        # odd nfft excludes Nyquist point
        if (self.nsamples % 2) > 0: # we've got odd number of points fft
            fourier[1:self.nsamples] = fourier[1:self.nsamples] * 2
        else:
            fourier[1:self.nsamples-1] = fourier[1:self.nsamples- 1] * 2 # we've got even number of points fft
        return (fourier, n_unique_pts)
    
    def setEndOfSound(self, i_time):
        self.time_end  = float(i_time)
    
    def getEndOfSound(self):
        if self.time_end  is None: 
            if self.rms_vals is None:
                rms_vals = [self.rms(i_samples=self.sound[-n:], i_fast=True) for n in range(1,self.nsamples+1)]
                rms_vals.reverse()
                self.rms_vals = np.array(rms_vals)
            idx = np.nonzero( np.array( self.rms_vals ) >= self.rms_thresh )[0]
            idx = idx[-1]
            t =    np.arange(0.0, float(self.nsamples), 1.0) / float(self.sampling_freq)
            self.time_end = t[idx]
        return self.time_end
      
####################################### Display functions

    def plotRms(self, i_labels=False):
        t_end =self.getEndOfSound()
        t = np.arange(0.0, self.nsamples, 1.0) / self.sampling_freq
        max_y = max(self.rms_vals)
        min_y = min(self.rms_vals)
        p.plot( t, self.rms_vals, color='k' ) 
        p.hold('True')
        p.plot([t_end, t_end], [min_y, max_y],'r')   
        if i_labels:
            p.ylabel('Accummulating RMS')
            p.xlabel('Time (ms)')
            
    def plotSound(self, x, y, i_color='k'):
        t =  np.arange(0.0, self.nsamples , 1.0) / self.sampling_freq
        t += x
        max_y = max(self.sound)
        min_y = min(self.sound)
        p.plot(t, self.sound + y, color=i_color)
        p.plot([x+0.15, x+0.15], [-1.2+y, 1.5+y ],'k', linewidth=3)   
        
    def plotAmplitude(self, i_ground_truth=None, i_labels=False, i_time_offset=0.0, i_plot_sound_end=True):
        t_end =self.getEndOfSound() + i_time_offset
        t =  np.arange(0.0, self.nsamples , 1.0) / self.sampling_freq
        t += i_time_offset
        max_y = max(self.sound)
        min_y = min(self.sound)
        p.plot(t, self.sound, color='k')
        p.hold('True')
        if i_plot_sound_end:
            p.plot([t_end, t_end], [min_y, max_y],'r', linewidth=3)   
        if i_ground_truth is not None:
            t_end_grnd_truth =  i_ground_truth[self.letter] + i_time_offset
            p.plot([t_end_grnd_truth, t_end_grnd_truth], [min_y, max_y],'b')   
        if i_labels:
            p.ylabel('Amplitude')
            p.xlabel('Time (ms)')
     
    def plotPowerSpectrum(self,i_labels=False ):   
        (fourier, n_unique_pts)=  self.fft()
        freq = np.arange(0, n_unique_pts, 1.0) * (self.sampling_freq / self.nsamples)
        print "RMS from amplitude: ",  np.sqrt(np.mean(self.sound**2))
        print "RMS from power spectrum: ",  np.sqrt(np.sum(fourier))
        p.plot(freq , 10.*np.log10(fourier), color='k')
        if i_labels:
            p.xlabel('Frequency (kHz)')
            p.ylabel('Power (dB)')
        
    def displaySound(self, i_sound_file=None, i_labels=True, i_ground_truth=None):
        if i_sound_file is not None:
            self.setSoundFile(i_sound_file)
        p.figure()
        p.subplot(2,2,1)
        sound_analyser.plotAmplitude( i_ground_truth=i_ground_truth, i_labels=i_labels)
        p.subplot(2,2,2)
        sound_analyser.plotPowerSpectrum(i_labels=i_labels)
        p.subplot(2,2,3)
        sound_analyser.plotRms(i_labels=i_labels)
        p.show()
   
  
class BatchExtractor(object):
    def __init__(self, i_alphabet_sequence, i_file_name, out_dir,  i_start=0.1, i_period=0.4):
        self.start = i_start
        self.file_name = i_file_name
        self.period = i_period
        self.seq = np.array(i_alphabet_sequence)
        self.out_dir = out_dir

    def compute(self):
        print "Sound file = ", self.file_name
        #can also  call oggread
        (snd, sampling_freq, n_bits) =  audiolab.wavread(self.file_name)
        n_samples = snd.shape[0]
        file_duration = float(n_samples) / float( sampling_freq )
        print "n_samples = ", n_samples, " file_duration = ", file_duration, " seconds"
    
        #The number of samples in a file - assuming all files have the same length = self.period
        letter_samples = np.int32(self.period* float( sampling_freq ))
        #The sample number to start from
        n_start = np.int32( self.start * float(sampling_freq))
        print "letter_duration = ", letter_samples, " m_start = ", n_start
        p.figure()
        for n in range(0, len(self.seq)):
            letter = self.seq[n]
            filename =  self.out_dir + letter+ ".ogg"
            amplitude = snd[n_start:n_start+letter_samples]
            audiolab.oggwrite(amplitude, filename, fs=sampling_freq, enc='vorbis')
            n_start += letter_samples
            #Display
            p.subplot(7,4,n+1)
            t =  1000.0 * np.arange(0.0, letter_samples , 1.0) / sampling_freq
            print "len t = ", len(t), " letter_samples = ", letter_samples, " lend(Data) = ", len(amplitude)
            p.plot(t, amplitude, color='k')
            max_y = max(amplitude)
            min_y = min(amplitude)
            p.plot([0.5*t[-1], 0.5*t[-1]], [min_y, max_y],'r', linewidth=3)   
            p.ylabel(letter)
            #t += i_time_offset
            #n_end -=letter_samples
            #p.ylabel('Amplitude')
            #p.xlabel('Time (s)')
        p.show()
      
if __name__ ==  "__main__":
    #print "Run letter_offset_display for some examples, and saving of all end of alphabet sounds"
    c = ChannelConfig(i_nchannels=5,i_sound_overlap=0.65, i_file_length=0.21, i_root_dir='../../')
    timing_info = c.getClickTimes()
    alphabet_loader = c.getAlphabetLoader()
    alphabet = alphabet_loader.getAlphabet(i_with_spaces=True, i_group=True)
      
    sound_dir  = "../alphabet_fast/channels5/"
    print "sound dir = ", sound_dir
    print "timing_info = ", timing_info
    print "alphabet = ", alphabet
   
    a = BatchSoundAnalyser()
    a.displayChannelSounds( sound_dir ,  alphabet,  timing_info, i_plot_axis=True)
   
    #alphabet = ['a','b','c','d','e','f','g','h','i','j','k', 'l', 'm','n','o','p','q','r','s','t','u','v','w','x','y','z','_','.']
    #extractor = BatchExtractor(alphabet, file_name, output_dir,  i_start=0.77, i_period=0.5)
    #extractor.compute()
    

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


class BatchSoundAnalyser(object):
    def __init__(self, i_display=False):
        self.__sound_analyser = SoundAnalyser()
        self.__times = None
        self.__display = i_display       
    
    def computeEndOfSounds(self, i_alphabet, i_ground_truth=None, o_file_name=None  ):
        self.__times = {}
        for letter in i_alphabet:
            self.__times [letter] =  self.__sound_analyser.getEndOfSound()
        if o_file_name is not None:
            filename =  i_sound_dir + o_file_name
            print "Saving time information to ", filename 
            f = open(filename, 'w')
            cPickle.dump(self.__times, f)
            f.close()
    
    def displayChannelSounds(self, i_sound_dir, i_channel_config, i_timing_info, i_plot_axis=True):
        p.figure(facecolor='w')
        n_channels =  len(i_channel_config)
        (tmax, tmin, ymax,ymin) = ( max( i_timing_info.values())[-1] ,  0, 1.5, -1.5)
        for channel in range(0,n_channels):
            p.subplot(n_channels, 1, channel+1); p.hold('on')
            p.axis( [tmin, tmax, ymin, ymax])
            for letter in  i_channel_config[channel]:
                sound_file = i_sound_dir + letter + ".ogg"
                self.__sound_analyser.setSoundFile(sound_file)
                time_offset = i_timing_info[letter][0]
                if self.__times is not None:
                    self.__sound_analyser.setEndOfSound(self.__times[letter])         
                self.__sound_analyser. plotAmplitude( i_ground_truth=None, i_labels=False,  i_time_offset=time_offset,  i_plot_sound_end=True)
                if i_plot_axis:
                    p.axis('on')
                else:
                    p.axis('off')
 
                p.text(  time_offset + self.__sound_analyser.getEndOfSound(), ymax-0.4, str(letter) )
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
            self.__sound_analyser.setSoundFile(sound_file)
            if self.__times is not None:
                self.__sound_analyser.setEndOfSound(self.__times[letter])
                if self.__display:
                    diff = np.abs( self.__times[letter] -  i_ground_truth[letter] )
                    total_diff += diff
                    print "Letter: %s, time  = %0.2f, grnd_truth = %0.2f, diff=%0.2f" % (letter, self.__times[letter],  i_ground_truth[letter], diff)
            p.subplot(nrows, ncols, n+1)
            self.__sound_analyser. plotAmplitude( i_ground_truth=i_ground_truth, i_labels= i_labels)
            p.title(letter)
            if i_plot_axis:
                p.axis('on')
            else:
                p.axis('off')
        if self.__display:
            print "Average diff=0.2f" % (total_diff / len(i_alphabet))
        p.show()
        
class SoundAnalyser(object):
    def __init__(self, i_sound_file=None, i_display=False):
        self.__rms_thresh = 0.001
        self.__display = i_display
        if i_sound_file is not None:
            self.setSoundFile(i_sound_file)
        
    def setSoundFile(self, i_sound_file):
        self.__sound_file = i_sound_file
        (   self.__sound, self.__n_channels, self.__sampling_freq, 
            self.__file_duration, self.__nsamples, self.__letter ) = self.soundInfo(i_sound_file)
        self.__rms_vals = None
        self.__time_end = None#Time when the sound ends
       
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
    
        if self.__display:
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
            fourier = p.fft(self.__sound)  
        else:
            fourier = p.fft(i_samples)
        n_unique_pts = np.ceil((self.__nsamples+1)/2.0)
        fourier= np.abs(fourier[0:n_unique_pts])
        fourier = fourier / float(self.__nsamples)# scale by the number of points so that
                                            #the magnitude does not depend on the length 
                                            # of the signal or on its sampling frequency  
        fourier= fourier**2  # square it to get the power 

        # multiply by two (see technical document for details)
        # odd nfft excludes Nyquist point
        if (self.__nsamples % 2) > 0: # we've got odd number of points fft
            fourier[1:self.__nsamples] = fourier[1:self.__nsamples] * 2
        else:
            fourier[1:self.__nsamples-1] = fourier[1:self.__nsamples- 1] * 2 # we've got even number of points fft
        return (fourier, n_unique_pts)
    
    def setEndOfSound(self, i_time):
        self.__time_end  = float(i_time)
    
    def getEndOfSound(self):
        if self.__time_end  is None: 
            if self.__rms_vals is None:
                rms_vals = [self.rms(i_samples=self.__sound[-n:], i_fast=True) for n in range(1,self.__nsamples+1)]
                rms_vals.reverse()
                self.__rms_vals = np.array(rms_vals)
            idx = np.nonzero( np.array( self.__rms_vals ) >= self.__rms_thresh )[0]
            idx = idx[-1]
            t =    np.arange(0.0, float(self.__nsamples), 1.0) / float(self.__sampling_freq)
            self.__time_end = t[idx]
        return self.__time_end
      
####################################### Display functions

    def plotRms(self, i_labels=False):
        t_end =self.getEndOfSound()
        t = np.arange(0.0, self.__nsamples, 1.0) / self.__sampling_freq
        max_y = max(self.__rms_vals)
        min_y = min(self.__rms_vals)
        p.plot( t, self.__rms_vals, color='k' ) 
        p.hold('True')
        p.plot([t_end, t_end], [min_y, max_y],'r')   
        if i_labels:
            p.ylabel('Accummulating RMS')
            p.xlabel('Time (ms)')
        
    def plotAmplitude(self, i_ground_truth=None, i_labels=False, i_time_offset=0.0, i_plot_sound_end=True):
        t_end =self.getEndOfSound() + i_time_offset
        t =  np.arange(0.0, self.__nsamples , 1.0) / self.__sampling_freq
        t += i_time_offset
        max_y = max(self.__sound)
        min_y = min(self.__sound)
        p.plot(t, self.__sound, color='k')
        p.hold('True')
        if i_plot_sound_end:
            p.plot([t_end, t_end], [min_y, max_y],'r', linewidth=3)   
        if i_ground_truth is not None:
            t_end_grnd_truth =  i_ground_truth[self.__letter] + i_time_offset
            p.plot([t_end_grnd_truth, t_end_grnd_truth], [min_y, max_y],'b')   
        if i_labels:
            p.ylabel('Amplitude')
            p.xlabel('Time (ms)')
     
    def plotPowerSpectrum(self,i_labels=False ):   
        (fourier, n_unique_pts)=  self.fft()
        freq = np.arange(0, n_unique_pts, 1.0) * (self.__sampling_freq / self.__nsamples)
        print "RMS from amplitude: ",  np.sqrt(np.mean(self.__sound**2))
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
        self.__start = i_start
        self.__file_name = i_file_name
        self.__period = i_period
        self.__seq = np.array(i_alphabet_sequence)
        self.__out_dir = out_dir

    def compute(self, i_ogg):
        print "Sound file = ", self.__file_name
        #can also  call oggread
        (snd, sampling_freq, n_bits) =  audiolab.wavread(self.__file_name)
        n_samples = snd.shape[0]
        file_duration = float(n_samples) / float( sampling_freq )
        print "n_samples = ", n_samples, " file_duration = ", file_duration, " seconds"
        #The number of samples in a file - assuming all files have the same length = self.__period
        letter_samples = np.int32(self.__period* float( sampling_freq ))
        #The sample number to start from
        n_start = np.int32( self.__start * float(sampling_freq))
        print "letter_duration = ", letter_samples, " m_start = ", n_start 
        (n_rows, n_cols) = (7, 4)
        n_plots = n_rows * n_cols
        for n in range(0, len(self.__seq)):
            letter = self.__seq[n]
            filename =  self.__out_dir + letter
            if i_ogg:
                filename += ".ogg"
            else:
                filename += ".wav"
            amplitude = snd[n_start:n_start+letter_samples]
            if i_ogg:
                audiolab.oggwrite(amplitude, filename, fs=sampling_freq, enc='vorbis')
            else:
                audiolab.wavwrite(amplitude, filename, fs=sampling_freq)
            n_start += letter_samples
            #Display
            cur_plot = (n % n_plots) + 1
            if cur_plot == 1:
                p.figure()
            t =  1000.0 * np.arange(0.0, letter_samples , 1.0) / sampling_freq
            print "n = ", n, " cur_plot = ", cur_plot, " n_plots = ", n_plots, " file_name = ", filename, 
            print " len t = ", len(t), " letter_samples = ", letter_samples, " len(Data) = ", len(amplitude)
            p.subplot(n_rows, n_cols, cur_plot)
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
    file_name = "../new.wav"
    output_dir = './tmp/'
    ogg = False
    #current_config = ChannelConfig(i_nchannels=5,i_sound_overlap=1.0, i_root_dir='../')
    #cmds = ['a','b','c','d','e','f','g','h','i','j','k', 'l', 'm','n','o','p','q','r','s','t','u','v','w','x','y','z','_','.']
    #cmds = ["first", "second","third","fourth","fifth","sixth","seventh","eighth","ninth","tenth", 
    #            "eleventh","twelfth","thirteenth","fourteenth","fifteenth","sixteenth","seventeenth","eighteenth",
    #            "nineteenth","twentieth","twentyfirst","twentysecond","twentythird","twentyfourth","twentyfifth",
    #            "twentysixth","twentyseventh","twentyeighth","twentyninth","thirtieth",
    #            "letter", "alphabet", "repetition", "reached", "shutting", "down", "top", "three","words"]
    cmds = ["row", "column","scan", "one","two","three","four","five","delete","selected"]
    extractor = BatchExtractor(cmds, file_name, output_dir,  i_start=1.6, i_period=0.5)
    #extractor = BatchExtractor(cmds, file_name, output_dir,  i_start=2.865, i_period=0.2)
    extractor.compute(i_ogg = ogg)
    
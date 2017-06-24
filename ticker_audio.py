 
import audio, time, os, subprocess
import numpy as np
 
class Audio():
    """This is the main audio module of Ticker:
        1) The C++ fmod wrapper is called to play the alphabet in multiple channels,
          i.e., at different stereo positions. 
        2) The alphabet to play and the stereo positions of each channel can be 
           found in the config directory, where e.g., 5 channels correspond to 5 
           stereo positions (between -1 (left) and 1 (right)). The alphabet is 
           loaded from the config files by the ChannelConfig class.
        3) The overlap between different channels can also be set (in the channel config). 
          With an overlap of  0.1, the next sound will start if 10% of the current sound is complete.
          The smaller the overlap the more it will sound as if the sounds are being 
          played simultaneously, and therefore increasing the speed.
        4) The restart function has to be called to start playing the alphabet. 
           After the whole alphabet has been played the audio will be stopped 
           automatically, and update will return True.
        5) The test functions for this class are in module_tests/test_ticker_audio, where
           test_alphabet_player.cpp interacts with the underlying C++ code and 
           test_alphabet_player.py demostrates how this class works with a small gui."""
    
    ##############################################Initialisation functions
    
    def __init__(self,  i_nchannels=None,  i_parent=0, i_root_dir=None):
        """ * Each ticker widget consists of a widget and a title associated with it - this title is also a widget, typically a QLabel
           * The title can be none """
        if i_nchannels is not None:
            self.setChannels(i_nchannels)        
        self.initDirectories(i_root_dir=i_root_dir)
        #FIXME: include dirs
        operating_system = 'Linux' #os.uname()[0]
        if operating_system == 'Windows':
            self.festival = ['festival.exe', '--tts', self.talk_file, '--libdir', 'C:\\Users\\en256\\Desktop\\libs\\festival\\festival\\lib']
        else:
            self.festival = ['text2wave', self.talk_file, '-o', self.synthesis_file]
            #self.festival = ['/usr/bin/festival', '--tts', self.talk_file]
         
    def initDirectories(self, i_root_dir=None):
        if i_root_dir is None:
            self.cur_dir = os.path.dirname(os.path.abspath(__file__)) + "/"
        else:
            self.cur_dir = str(i_root_dir)
        self.talk_file = self.cur_dir + 'talk.txt'
        self.synthesis_cmd = "synthesise"
        self.synthesis_file = self.cur_dir + "voice_recordings/commands/" + self.synthesis_cmd  + ".wav"
        audio.setRootDir(self.cur_dir)
        
    def resetInstructions(self): 
        self.instructions = []
        self.file_types = []
        self.instruct_idx = 0
         
    ##################################### Set functions
 
    def setChannels(self, nchannels):
        audio.setChannels(nchannels)
        self.clear()
        self.resetInstructions()
       
    def setTicks(self, i_nticks):
        audio.setTicks(i_nticks)
    
    def setPause(self, i_pause):
        self.pause = i_pause
        
    def setVolume(self, i_val, i_channel):
        audio.setVolume(np.float64(i_val), np.int32(i_channel))
    
    def setAlphabetDir(self, i_alphabet_dir):
        audio.setAlphabetDir(i_alphabet_dir)
        
    def setConfigDir(self, i_config_dir):
        audio.setConfigDir(i_config_dir)
        
    ##################################### Get functions
    
    def isPlayingInstructions(self):
        return  len(self.instructions) > 0
    
    def isReady(self):
        return audio.isReady()
    
    def getSoundIndex(self, i_channel_config):
        n_ticks = i_channel_config.getNumberOfPreTicks()
        idx = self.sound_index - n_ticks
        if idx < 0:
            idx = 0
        return idx

    def getTime(self,i_channel_config):
        self.updateCurTime(i_channel_config)
        return self.cur_time
    
    #####################################' Main functions
    
    def utteranceFromWord(self, i_word):
        utterance = str(i_word) 
        if utterance[-1] == "_":
            utterance = utterance[0:-1] 
        if utterance == '.':
            utterance = "fullstop"  
        elif(utterance == 'a'):
            utterance = 'ay'
        elif(utterance == 'the'):
            utterance = 'thee'
        elif(utterance == 'of'):
            utterance = 'of'
        elif(utterance == 'on'):
            utterance = 'on'
        elif(utterance == 'or'):
            utterance = 'oar'
        elif(utterance == 'to'):
            utterance = 'two'
        return utterance
    
    def saveUtterance(self, i_utterance):
        print "SAVING UTTERENCE : ", i_utterance
        fid = open(self.talk_file,'w') 
        fid.write(i_utterance)
        fid.close()
        retcode = subprocess.call(self.festival) 
    
    def synthesise(self, i_utterance): 
        self.saveUtterance(i_utterance)
        self.playInstructions([self.synthesis_cmd], ".wav") 
        
    def synthesiseWord(self, i_word, i_is_word_selected=True):
        utterance = self.utteranceFromWord(i_word)
        if i_is_word_selected:
            utterance = ".. you have written " + utterance + " ?"
        self.synthesise(utterance)
        
    def synthesisePhrase(self, i_phrase):
        phrase = ' '.join(i_phrase.split('_'))
        utterance = "New phrase ... "+ phrase + "..."  
        self.synthesise(utterance)
        
    def playNext(self, i_loop=True):
        if self.isPlayingInstructions():
            if not self.isReady():
                return
            self.pause = True
            if self.instruct_idx >= len(self.instructions):
                self.resetInstructions()
                return
            audio.playInstruction(self.instructions[self.instruct_idx], self.file_types[self.instruct_idx])
            print "PLAYING INSTRUCTION : ", self.instructions[self.instruct_idx]
            self.instruct_idx += 1 
            return
        if i_loop:
            self.pause = False 
            audio.playNext()
        
    def restart(self):
        self.clear()
        self.resetInstructions()
        self.pause = True        

    def clear(self):
        self.sound_index = 0
        self.cur_time = 0.0 
        self.clock_time = 0.0
        self.letter_start = time.time()
        self.pause = True
        
    def stop(self):
        self.clear()
        audio.stop()
        
    def playInstructions(self, i_str_list, i_file_type=".ogg"):
        """Play the .ogg files that start with the entries in the input list of string."""
        self.instructions.extend(list(i_str_list))
        for cmd in i_str_list:
            self.file_types.append(i_file_type)
        self.playNext( i_loop=True)
        
        print "STORED INSTRUCTIONS = ", self.instructions, "NEW INSTRUCT = ", i_str_list
    
    def playClick(self):
        audio.playClick()
    
    def update(self, i_channel_config, i_loop=True):
        """Return true when ready to play new sequence of letters/instruction"""
        (is_read_next, is_update_time, is_first_letter) = (False, False, False)
        if np.abs(self.cur_time) < 1E-6: 
            self.letter_start = time.time()
        if self.isPlayingInstructions(): 
            if audio.isReady(): 
                self.playNext(i_loop)
            return (is_read_next, is_update_time, is_first_letter)
        if self.pause:
            is_first_letter = True
            if audio.isReady():
                self.playNext(i_loop)
            return (is_read_next, is_update_time, is_first_letter)
        letter_times = i_channel_config.getSoundTimes() 
        self.updateCurTime(i_channel_config)
        is_update_time = True
        if self.clock_time >= letter_times[self.sound_index][1]:
            if self.sound_index >= (len(letter_times) - 1):
                is_read_next = True
                return (is_read_next, is_update_time, is_first_letter)
            else:
                self.playNext()
                self.sound_index += 1
                self.dispNewSound(i_channel_config, i_disp_new_sound=False, i_channel=None)
        return (is_read_next, is_update_time, is_first_letter)
    
    def readTime(self, i_channel_config):
        est_read_time = i_channel_config.getSoundTimes()[-1,-1]
        self.read_time = self.clock_time
        print " Cur time = ", self.cur_time, " Clock time = ", self.clock_time, " s", " should be approx ", est_read_time, " s"
        
    def updateCurTime(self, i_channel_config): 
        self.clock_time = time.time() - self.letter_start
        letter_times = i_channel_config.getSoundTimes() 
        cur_letter_times = np.array(audio.getCurLetterTimes()) / 1000.0 
        idx = np.nonzero(cur_letter_times > -1E-12)[0]
        if len(idx) > 0: 
            self.cur_time = np.mean(letter_times[idx,0] + cur_letter_times[idx]) 
            sound_idx = idx[-1]
            if not idx[-1] == self.sound_index:
                self.dispCurTime(i_channel_config, cur_letter_times, i_disp_cur_time=True)
                print "sound_index = ", self.sound_index, " index = ", idx[-1]     
                raise ValueError("Idx not equal!!!")
        else:  
            self.cur_time = self.clock_time
        if np.abs(self.cur_time - self.clock_time) > 0.1:
            self.dispCurTime(i_channel_config, cur_letter_times, i_disp_cur_time=True) 
            raise ValueError("Clock time and current time out of sync!!!")
        self.dispCurTime(i_channel_config, cur_letter_times, i_disp_cur_time=False)    
            
    ######################################### Display diagnostic

    def dispNewSound(self, i_channel_config, i_disp_new_sound=False, i_channel=None):
        if not i_disp_new_sound:
            return
        letter_times = i_channel_config.getSoundTimes() 
        if self.sound_index >= letter_times.shape[0]:
            return
        alphabet = i_channel_config.alphabet.getAlphabet(i_with_spaces=False, i_group=False)
        letter = alphabet[self.getSoundIndex(i_channel_config)]
        if i_channel is not None:
            alphabet = np.array(i_channel_config.alphabet.getUniqueAlphabet(i_with_spaces=True, i_group=True))
            (row,col) = np.nonzero(alphabet == letter)
            if not row==i_channel:
                return    
        click_times = i_channel_config.getClickTimes()
        n_ticks = i_channel_config.getNumberOfPreTicks()
        click_time_idx = self.sound_index - n_ticks
        if click_time_idx < 0:
            click_time_idx = 0
        #print "***************************************************************************"
        print "Finished with: sound index=%.2d, letter=%s " % (self.sound_index, letter), 
        print " cur time=%2.3f, clock_time%2.3f, diff=%2.3f" % (self.cur_time, self.clock_time, self.clock_time-self.cur_time),
        print " start_time=%2.3f, end_time=%2.3f" % (letter_times[self.sound_index][0],letter_times[self.sound_index][1]),
        print " click_time=%2.3f" % click_times[click_time_idx]
         
    def dispCurTime(self, i_channel_config, i_cur_letter_times, i_disp_cur_time=False):
        if not i_disp_cur_time:
            return 
        letter_times = i_channel_config.getSoundTimes() 
        alphabet = i_channel_config.alphabet.getAlphabet(i_with_spaces=False, i_group=False)
        letter = alphabet[self.getSoundIndex(i_channel_config)]
        disp_str = "index=%d of %d, letter=%s, start time=%2.4f, end time=%2.4f," % (self.sound_index,
            len(letter_times), letter, letter_times[self.sound_index][0], letter_times[self.sound_index][1])
        disp_str += (" cur_time=%.3f, clock_time=%.3f" % (self.cur_time, self.clock_time))
        print disp_str, 
        print " ", i_cur_letter_times
       
 
from PyQt4 import QtCore, QtGui
import sys, os, copy
sys.path.append("../../")
from grid import GridGui
from utils import Utils 
import numpy as np
import time, cPickle, shutil
from ticker_widgets import InstructionsDisplay
from synthetic_noise import SyntheticNoise

global phrase_start, user, session, sub_session, synthetic_noise
phrase_start = 0
user=4
session=5 #The user will write in various sessions
sub_session=2
synthetic_noise=True#Add some synthetic noise to the keyboard event.

class GridGuiExperiment(GridGui):
    
    ########################################### Init 
    def __init__(self):
        self.initialised = False
        self.max_word_repeat = 2 #The maximum number of times any word can be repeated (number of clicks) 
        self.initDirectories() 
        self.utils = Utils()
        self.phrases = self.utils.loadText(self.phrase_file)
        self.phrases = self.phrases.split('\n')
        self.phrase_cnt = phrase_start
        GridGui.__init__(self)
        self.phrase_display = InstructionsDisplay(self.phrase_disp, self.centralwidget, i_title=None)
        self.action_tutorial.setVisible(True)
        self.action_inc_phrases.setVisible(True)
        self.showRecordingWidgets()
        #Synthetic noise 
        if synthetic_noise:
            self.noise = SyntheticNoise(sub_session)
            self.setNoiseParameters()
        self.delay_noise_timer = QtCore.QTimer()
        self.delay_noise_timer.setSingleShot(True)
        QtCore.QObject.connect( self.delay_noise_timer, QtCore.SIGNAL("timeout()"), self.processTruePositive)
        #Start the experiment
        self.loadPhrase() 
        self.initialised = True
    
    def initDirectories(self):
        self.root_dir = "../../../user_trials/audio_experiment/grid/"
        self.phrase_file = "phrases.txt"
        if not os.path.exists(self.root_dir):
            os.mkdir(self.root_dir)
        self.user_dir = "%suser_%.2d/" % (self.root_dir, user)
        if not os.path.exists(self.user_dir):
            os.mkdir(self.user_dir)
        self.user_dir = "%ssession_%.2d/"% (self.user_dir, session)
        if not os.path.exists(self.user_dir):
            os.mkdir(self.user_dir)
        self.user_dir = "%ssub_session_%.2d/"% (self.user_dir, sub_session)
        if not os.path.exists(self.user_dir):
            os.mkdir(self.user_dir)    
        self.output_dir = self.user_dir
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        shutil.copyfile(self.phrase_file, self.output_dir + self.phrase_file)
        self.tutorial_dir = self.output_dir + "tutorial/" 
        if not os.path.exists(self.tutorial_dir):
            os.mkdir(self.tutorial_dir)        
            
    def initNewPhraseVariables(self):
        """* Letter offset for whole phrase 
           * For the letter offset in current word see ticker_core.letter_index""" 
        self.letter_offset = 0 
        self.word_cnt = 0
        self.sentence_display.clear()
        
    def loadPhrase(self):
        self.initNewPhraseVariables()
        cur_phrase = self.phrases[self.phrase_cnt]
        self.phrase_display.update(cur_phrase)
        self.phrase_display.highLight(0)
        self.updateLetterDisplay() 
        if self.initialised:
            if self.phrases[self.phrase_cnt] == ".":
                utterance = "fullstop"
            else:
                utterance = str(self.phrases[self.phrase_cnt])
            self.audio.synthesisePhrase(utterance)
            self.waitAudioReady()
            self.initDisplayForNewWord()
            self.updateLetterDisplay() 
          
    ########################################## Main
    
    def update(self):
        (is_read_next, is_update_time, is_first_letter) = GridGui.update(self)
        if is_first_letter:
            print "***********************************************************"
            self.generateFalsePositives()
        if is_update_time:
            self.processFalsePositive()
            
    def pauseSlot(self, i_checked):
        if i_checked: 
            if self.restart:
                self.pauseTrue(False)
                self.restart = False
                if self.action_inc_phrases.isChecked():
                    self.phrase_cnt += 1  
                self.startSoundTrue()
                if self.initialised: 
                    self.loadPhrase()
            else:
                self.pauseTrue(True)
            self.main_timer.start(10)
        else:
            self.pauseFalse()
    
    def updateNextLetter(self, i_letter):
        print "-------------------------------------------------------------------"
        click_stats = self.getCurClickStats()
        self.letter_idx += 1        
        self.updateLetterDisplay(i_new_word=False)
        #No word selected but clicks received - if maximum word repetition reached emit error, 
        #otherwise return
        if self.isMaxWordRepetionReaced():
            self.processTimeoutError(i_letter, click_stats)
        else:
            self.sentence_display.update(i_letter, i_adjust_stop=False, i_add_space=False)
            self.newLetter(['written', i_letter, "next"])
                
    def processTimeoutError(self, i_letter, i_click_stats):
        click_stats = dict(i_click_stats)
        self.sentence_display.update(i_letter + "_", i_adjust_stop=False, i_add_space=False)
        utterance =  ".. time out error ?" 
        self.audio.synthesise(utterance)
        self.waitAudioReady()
        click_stats['selected_word'] = self.sentence_display.lastWord()
        click_stats['word_error'] = True
        self.saveWordStatistics( click_stats )
        self.newWord(i_is_word_selected=False, i_play_new_word=False)
        self.waitAudioReady()
            
    def processWord(self, i_letter):
        print "*****************************************************************"
        click_stats = self.getCurClickStats()
        click_stats['selected_word'] = GridGui.processWord(self, i_letter, i_play_new_word=False)
        if (not click_stats['selected_word']  == ".") and (not click_stats['selected_word']  == ""):
            click_stats['selected_word'] += "_"
        self.saveWordStatistics(click_stats) 
     
    def playCurrentWord(self):
        cur_word = self.getCurWord()
        print "PLAYING CURRENT WORD: ", cur_word
        if not cur_word == ".":
            cur_word = cur_word[0:-1]
        cur_word = self.audio.utteranceFromWord(cur_word)
        cur_word += " ?" 
        self.audio.synthesise(cur_word)
        self.waitAudioReady()
        
    def newLetter(self, i_cmd_str):
        GridGui.newLetter(self, i_cmd_str)
        self.updateLetterDisplay()
        
    ################################################## Display
    
    def updateLetterDisplay(self, i_new_word=False):
        print "IN UPDATE LETTER DIS new_word = " , i_new_word, " word_cnt = ", self.word_cnt
        if i_new_word:
            phrase_words = self.getCurPhrase()
            print "phrase_words = ", phrase_words, " len = ", len(phrase_words)
            #Load a new phrase if this on is finished
            if self.word_cnt >= len(phrase_words):
                self.phrase_cnt += 1
                #Multiple phrases not currently allowed with calibration
                self.loadPhrase() 
            else:
                self.letter_offset=0
                for n in range(0, self.word_cnt):
                    self.letter_offset += ( len(phrase_words[n]) + 1 )
        self.adjustCursorToBeginWord()
        cur_word = self.getCurWord()
        letter_index = self.getWarpIndex()
        if letter_index > 0:
            self.phrase_display.highLight(letter_index) 
        cur_letter = cur_word[letter_index]   
        print "IN UPDATE LETTER DIS new_word: cur_word = ", cur_word, " letter_index = ", letter_index, " current letter = " , cur_letter, 
        print " letter_offset = ", self.letter_offset
         
    def adjustCursorToBeginWord(self):
        print "In update letter display calling highlight, letter offset = ", self.letter_offset
        self.phrase_display.highLight(0) 
        self.phrase_display.highLight(self.letter_offset) 
        
    ##################################### Save
 
    def saveWordStatistics(self, i_click_stats ):
        #Store click stats running up to selection
        click_stats = dict(i_click_stats)
        #If the maximum number of word repetions were reach there will be a key already
        if click_stats['grnd_truth'] == click_stats['selected_word']:
            click_stats['word_error'] = False
        else:
            click_stats['word_error'] = True 
        self.utils.savePickle(click_stats, self.getSaveFileName(click_stats))
        #Update After saving
        self.word_cnt += 1
        self.updateLetterDisplay(i_new_word=True)
        if not self.word_cnt == 0:
            self.audio.playInstructions(self.newWordStr()) 
            self.playCurrentWord()
            self.waitAudioReady()
        #Display
        print "******************************************************************"
        print "SAVED THE CLICK STATS TO FILE = ",self.getSaveFileName(click_stats)   
        self.dispClickStats(click_stats, i_disp_settings=True)
        
    def dispClickStats(self, i_click_stats, i_disp_settings):
        print "CLICK STATS:"
        for key in i_click_stats.keys():
            if key == "settings":
                continue
            print "------------------"
            print key, " :" , i_click_stats[key]
        if not i_disp_settings:
            return
        print "---------------------------------------"
        print "CLICK STATS SETTINGS:"
        for key in i_click_stats.keys():
            if not key == "settings":
                continue
            print "---------------------------------------"
            for sub_key in i_click_stats['settings'].keys():
                print sub_key, " :" , i_click_stats['settings'][sub_key]
            
   
    ################################################### Get
    
    def isMaxWordRepetionReaced(self):
        print "Testing for max repetition reached"
        cur_word = self.getCurWord()
        print "CUR WORD = ", cur_word
        print "letter_idx = ", self.letter_idx
        word_repeat = int(float(self.letter_idx) / len(cur_word))
        print "WORD REPEAT = ", word_repeat
        #Clicks were re
        if word_repeat < self.max_word_repeat:
            return False
        return True
    
    def getCurClickStats(self):
        """* This function is processed before a word is selected
           * If a word is selected then save these click stats (that's why there's an letter index offset of 1)""" 
        click_stats = {} 
        #Setttings
        click_stats['settings'] = dict(self.getSettings())
        #Stats
        click_stats['nclicks'] = self.nclicks
        click_stats['click_times'] = np.array(self.click_times)
        click_stats['undo_last_action_cnt'] = self.undo_last_action_cnt
        click_stats['delete_cnt'] = self.delete_cnt
        click_stats['nscans'] = np.array(self.nscans)
        click_stats['word_cnt'] = self.word_cnt     
        click_stats['letter_index'] = self.letter_idx
        click_stats['word_error'] = False
        click_stats['grnd_truth'] = str(self.getCurWord()) 
        return click_stats 
    
    def getSettings(self):
        #Settings
        settings = {}
        settings['scan_delay'] = self.scan_delay 
        settings['n_prog_status'] = self.n_prog_status
        settings['n_undo_last'] =  self.n_undo_last
        settings['tutorial'] = self.tutorial
        settings['config_seq'] = list(self.channel_config.config_seq)
        settings['config'] = dict(self.channel_config.config)  
        #Noise settings
        settings['synthetic_noise'] = synthetic_noise
        if synthetic_noise:
            settings['gauss_delay'] = self.noise.gauss_delay
            settings['gauss_std'] = self.noise.gauss_std
            settings['fp_rate'] = self.noise.fp_rate
            settings['fr'] = self.noise.fr
            settings['noise_delay'] = self.noise.delay
        return settings
    
    #File to save all results in
    def getSaveFileName(self, i_click_stats):
        if self.action_tutorial.isChecked():
            dir_name = "%s" % (self.tutorial_dir) 
        else:
            dir_name = "%s" % (self.output_dir) 
        file_name = "%sclick_stats_%.2d_%.2d.cPickle" % (dir_name, self.phrase_cnt, self.word_cnt) 
        return file_name
    
    def getCurPhrase(self):
        cur_phrase = self.phrases[self.phrase_cnt]
        phrase_words = cur_phrase.split("_")[0:-1]
        phrase_words.append('.') 
        return phrase_words
    
    def getCurWord(self):
        phrase_words = self.getCurPhrase()
        if phrase_words[self.word_cnt]  == '.':
            cur_word = '.' 
        else:
            cur_word = phrase_words[self.word_cnt] + "_"
        return cur_word

    def getWarpIndex(self):
        cur_word = self.getCurWord()
        print "IN get WARP IDX cur_word=  ", cur_word, " len = ", len(cur_word), " letter idx = ", self.letter_idx
        if self.letter_idx < len(cur_word):
            print "RETURNING with letter idx = ", self.letter_idx
            return self.letter_idx
        letter_idx = -self.letter_idx / len(cur_word)
        letter_idx *= len(cur_word)
        letter_idx += self.letter_idx
        if letter_idx < 0:
            letter_idx += len(cur_word)
        print "FINAL LETTER IDX = ", letter_idx
        return letter_idx
    
    ######################################################## Noise
         
    def generateFalsePositives(self):
        if not synthetic_noise:
            return
        period = self.channel_config.getSoundTimes()[-1,-1]
        self.false_positives = list(self.noise.sampleFalsePositives(period)) 
        print "Sampled false positives = ", self.false_positives, " period = ", period
        
    def processFalsePositive(self):
        if not synthetic_noise:
            return
        if len(self.false_positives) < 1:
            return
        cur_time = self.audio.getTime(self.channel_config)
        if cur_time < self.false_positives[0]:
            return
        fp_time = self.false_positives.pop(0)
        print "Generating false positive! at time", fp_time
        self.processClick()
        
    def processTruePositive(self):
        print "Processing true positive"
        self.processClick()
            
    def keyPressEvent(self, event):   
        if (event.key() == QtCore.Qt.Key_Space) and (not self.button_pause.isChecked()):
            self.pauseSlot(True)
        if event.key() == QtCore.Qt.Key_Space:
            if synthetic_noise:
                if self.noise.isFalseRejection():
                    print "Got a click at time " , self.audio.getTime(self.channel_config), 
                    print " but switch decided to ignore it"
                    return
                delay = self.noise.sampleGaussOffset()
                print "KeyPressEvent Got click at time : " , self.audio.getTime(self.channel_config),
                print  "  offset if with ", delay
                self.delay_noise_timer.setInterval(delay*1000)
                self.delay_noise_timer.start()
            else:
                self.processClick()
            
    def stopTimers(self): 
        GridGui.stopTimers(self)
        if not self.initialised:
            return
        self.delay_noise_timer.stop()
        
    def setNoiseParameters(self):
        if not synthetic_noise:
            return
        if not session == 4:
            raise ValueError("Noise should only be tested in session 4")
        scan_delay = 0.4
        scan_delay += self.noise.delay  
        scrollbar_val = int(10*scan_delay+0.5)
        print "Set new value : ", scrollbar_val
        self.scrollbar_letter_speed.setValue(scrollbar_val)
        print "AFTER SET: ", self.scrollbar_letter_speed.value()
        self.setScanDelay()
        print "Set noise params: gaus_delay = ", self.noise.gauss_delay, " std = ", self.noise.gauss_std, 
        print " fp rate = ", self.noise.fp_rate, " fr = " , self.noise.fr,
        print " noise delay = ", self.noise.delay, " new scan delay = ", scan_delay 
        
if __name__ ==  "__main__":
    app = QtGui.QApplication(sys.argv)
    gui = GridGuiExperiment()
    gui.show()
    sys.exit( app.exec_())
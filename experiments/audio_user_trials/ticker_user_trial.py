#! /usr/bin/env python 
from PyQt4 import QtCore, QtGui
import sys, os, copy
sys.path.append("../../")
from ticker import TickerGui
from utils import Utils 
import numpy as np
import time, cPickle, shutil
from ticker_widgets import InstructionsDisplay
from synthetic_noise import SyntheticNoise

global user, session, sub_session, synthetic_noise

user=10
session=1 #The user will write in various sessions
sub_session=1
synthetic_noise=False #Add some synthetic noise to the keyboard event.

class TickerGuiExperiment(TickerGui):
    
    ########################################### Init 
    def __init__(self):
        self.initialised = False
        self.max_word_repeat = 2 #The maximum number of times any word can be repeated 
        (settings_dir, settings_file) = self.initDirectories() 
        self.utils = Utils()
        self.phrases = self.utils.loadText(self.phrase_file)
        self.phrases = self.phrases.split('\n')
        self.phrase_cnt = 0
        TickerGui.__init__(self, settings_dir, settings_file)
        self.phrase_display = InstructionsDisplay(self.phrase_disp, self.centralwidget, i_title=None)
        self.action_tutorial.setVisible(True)
        self.action_inc_phrases.setVisible(True)
        #Synthetic noise 
        self.delay_noise_timers = []
        self.delay_timer_count = 0
        if synthetic_noise:
            self.noise = SyntheticNoise(sub_session, i_extra_wait_time=0.3)
            self.setNoiseParameters()
            for n in range(0,15):
                self.delay_noise_timers.append(QtCore.QTimer())
                self.delay_noise_timers[-1].setSingleShot(True)
                QtCore.QObject.connect( self.delay_noise_timers[-1], QtCore.SIGNAL("timeout()"), self.processTruePositive)
        #Start the experiment
        self.loadPhrase(self.isBusyCalibrating())
        self.showPhrases()
        self.initialised = True
        
    def initDirectories(self):
        self.root_dir = "../../../user_trials/audio_experiment/ticker/"
        self.phrase_file = "phrases.txt"
        if not os.path.exists(self.root_dir):
            os.mkdir(self.root_dir)
        self.user_dir = "%suser_%.2d/" % (self.root_dir, user)
        if not os.path.exists(self.user_dir):
            os.mkdir(self.user_dir)
        settings_dir = self.user_dir
        settings_file = "settings.cPickle"
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
        return (settings_dir, settings_file)
    
    def initNewWordVariables(self):
        self.false_positives = []
        self.ticker_core.newWord()
        TickerGui.initDisplayForNewWord(self)  
        self.word_repeat = 0
        self.niterations = 0 #Total number of alphabet sequence iterations
        self.nclick_iterations = 0 #Total number of click iterations
        self.nclicks = 0  
        self.avg_alphabet_read_time = 0.0
        self.nundo = 0
        self.total_time = time.time()
        self.inc_time = 0.0 #If pause is pressed this should be added
        self.click_distr_before = copy.deepcopy(self.ticker_core.click_distr)
         
    def initNewPhraseVariables(self):
        """* Letter offset for whole phrase 
           * For the letter offset in current word see ticker_core.letter_index"""
        self.initNewWordVariables() 
        self.letter_offset = 0 
        self.word_cnt = 0 
    
    def loadPhrase(self, i_is_calibrating):
        self.initNewPhraseVariables()
        if i_is_calibrating:
            cur_phrase = self.calibration_word
        else:
            cur_phrase = self.phrases[self.phrase_cnt]
        self.phrase_display.update(cur_phrase)
        self.phrase_display.highLight(0)
        self.updateLetterDisplay()
        self.read_phrase = True
        if self.initialised:
            if not i_is_calibrating:
                self.audio.synthesisePhrase(self.phrases[self.phrase_cnt])
                self.read_phrase = False
            else:
                self.audio.playInstructions(['calibrating']) 
        
    ########################################## Main
    
    def keyPressEvent(self, event):
        if (event.key() == QtCore.Qt.Key_Space) and self.enable_clicks and (not self.button_pause.isChecked()):
            self.pauseSlot(True)
        if (not self.enable_clicks) or (not self.action_space_bar.isChecked()):
            return  
        if event.key() == QtCore.Qt.Key_Space:
            if synthetic_noise:
                if self.noise.isFalseRejection():
                    print "Got a click at time ", self.audio.getTime(self.channel_config), 
                    print " but switch decided to ignore it"
                    return
                delay = self.noise.sampleGaussOffset()
                print "KeyPressEvent Got click at time : " , self.audio.getTime(self.channel_config),
                print  "  offset if with ", delay, " Total time = ",  self.audio.getTime(self.channel_config)+delay, 
                print " END boundary = ", self.channel_config.getSoundTimes()[-1,1] + 0.001*self.best_letters_timer.interval(),
                print " BEST LETTER = ",  0.001*self.best_letters_timer.interval()
                self.delay_noise_timers[self.delay_timer_count].setInterval(delay*1000)
                self.delay_noise_timers[self.delay_timer_count].start()
                self.delay_timer_count += 1
                if self.delay_timer_count >= len(self.delay_noise_timers):
                    self.delay_timer_count = 0
            else:
                self.processClick()
            
    def readPendingDatagrams(self):
        TickerGui.readPendingDatagrams(self)
        if self.enable_clicks:
            self.nclicks += 1
    
    def updateAudio(self):
        (is_read_next, is_update_time, is_first_letter) = TickerGui.updateAudio(self)
        if is_first_letter:
            self.generateFalsePositives()
            return
        if is_update_time:
            self.processFalsePositive()
        if not self.read_phrase:
            if not self.audio.isPlayingInstructions():
                self.read_phrase = True
                #After a new phrases is read, set the total time variable
                self.total_time = time.time()
            
    def pauseSlot(self, i_checked, i_play_cur_letter_idx=True):
        if i_checked: 
            self.total_time = time.time() 
            if self.restart:
                self.pauseTrue(False)
                self.restart = False
                if self.action_inc_phrases.isChecked():
                    self.phrase_cnt += 1  
                self.startSoundTrue()
                if self.initialised:    
                    self.loadPhrase(self.isBusyCalibrating())   
            else:
                self.pauseTrue(i_play_cur_letter_idx)
            self.main_timer.start(10)
        else:
            self.pauseFalse()
            #If the pause button is pressed update the total time that should be added
            self.inc_time += (time.time()-self.total_time) 
          
    def processClick(self):
        if self.enable_clicks:
            self.nclicks += 1
        is_undo = TickerGui.processClick(self)
        if is_undo:
            self.waitAudioReady()
            self.pauseSlot(False, i_play_cur_letter_idx=False)
            self.nundo += 1
            self.audio.setChannels(self.settings_editor.getCurrentChannel())
            self.button_pause.setChecked(True)
            self.button_pause.setText("Pause") 
            self.main_timer.start(10)
        
    def newWord(self, i_selected_word=None, i_grnd_truth_word=None, i_play_new_word=True): 
        if (i_selected_word is not None) and (not i_selected_word == ""):
            self.audio.synthesiseWord(i_selected_word) 
        if (i_grnd_truth_word is not None) and (not i_grnd_truth_word == ""):
            utterance =  ".. time out error ?" 
            self.audio.synthesise(utterance)
        if i_play_new_word:
            self.playCurrentWord()
        self.initNewWordVariables()
        
    def processClicks(self):
        print "*****************************************************************"
        print " processCLICKS word_cnt = ", self.word_cnt
        print "*****************************************************************" 
        self.false_positives = []
        self.niterations += 1
        #Avg alphabet read time does not include box end delay
        self.avg_alphabet_read_time +=  self.audio.read_time 
        print " CALLING TICKER USER TRIALS process clicks", " BEST TIME ACTIVE = ", self.best_letters_timer.isActive(), 
        print " Audio read time = ", self.audio.read_time, " end delay = ", self.settings_editor.box_end_delay.value()
        self.audio.clear()
        #No clicks received
        if not self.ticker_core.clicksReceived():
            print "PROCESS ALPHABET REPETIONS"
            self.processAlphabetRepetions()
            return
        #Clicks were received - process it
        self.nclick_iterations += 1
        self.repeat_count = 0
        selected_word = self.processWordSelections()
        print "IN PROCESS CLICKS selected_word = ", selected_word
        if selected_word is not None:
            return
        self.updateNextLetter()
        print "*****************************************************************"
        print " END OF PRCESS CLICKS = ", self.word_cnt
        print "*****************************************************************"
    
    def processAlphabetRepetions(self):
        #Process the number of times the alphabet sequence has been repeated when no clicks were received 
        self.repeat_count += 1
        word_repeat = self.settings_editor.box_restart_word.value()
        prog_repeat = self.settings_editor.box_prog_status.value()
        is_new_word =  self.repeat_count % word_repeat
        is_prog_status = self.repeat_count % prog_repeat
        if is_new_word is 0: 
            self.newWord(i_play_new_word=False)
            self.audio.playInstructions(["restart", "word"])
            self.updateLetterDisplay(i_new_word=True) 
        elif is_prog_status is 0:
            self.playCurrentLetterIndex()
            
    def playCurrentWord(self):
        cur_word = self.getCurWord()
        if not cur_word == ".":
            cur_word = cur_word[0:-1]
        cur_word = self.audio.utteranceFromWord(cur_word)
        cur_word += " ?" 
        self.audio.synthesise(cur_word)
         
    def processWordSelections(self):
        is_calibrated = (not self.action_calibrate.isChecked()) or self.calibrated
        click_stats = self.getCurClickStats()
        selected_word =  self.ticker_core.newLetter(i_process_word_selections=False) 
        if self.action_practise.isChecked():
            return
        (click_stats['top_ten_words'], click_stats['top_probs']) = self.ticker_core.getBestWordProbs(i_n=10)
        selected_word = self.__processWordSelection(selected_word, is_calibrated)
        print "in word selections TICKER USER TRIAL: selected word = ", selected_word 
        print " letter_index = ", click_stats['letter_index']
        print "====================================================================="
        #Selected word can change to "" if no word was selected but max repetions was reached - click stats will then be saved
        (selected_word, click_stats) = self.__updateWordSelectDisplay(selected_word, click_stats)
        if selected_word is None:
            return
        #Update values in settings editor
        self.settings_editor.setClickParams(self.settings_editor.clickPdfToSettingsParams(self.click_pdf.getParams()))
        self.saveWordStatistics(selected_word, click_stats) 
        self.__updateAfterWordSelect(selected_word,is_calibrated)
        return selected_word
    
    def __processWordSelection(self, i_selected_word, i_is_calibrated):
        #No word was selected - see if max word repetions was reached
        #Calibrating if process_word_selections = False
        selected_word = None
        if not i_is_calibrated: 
            selected_word = self.processWordSelectCalibrating()
        elif i_selected_word is not None:
            selected_word  = str(i_selected_word)
            if selected_word == ".":
                train_word = "."
            else:
                train_word = selected_word + "_"
            #Compensate for the fact that one has been added to the letter index (because processing was disabled)
            self.ticker_core.letter_idx -= 1
            self.ticker_core.train(train_word)
            self.ticker_core.newWord()
        return selected_word
    
    def __updateWordSelectDisplay(self, i_selected_word, i_click_stats):
        #Update the display if a word was selected, if no word was selected set selected word to ""
        #and update the display with it
        click_stats = dict(i_click_stats)
        if i_selected_word is not None:
            selected_word = str(i_selected_word)
        else:
            selected_word = None
        if selected_word is not None:
            self.sentence_display.update(i_selected_word) 
        else:
            #No word selected but clicks received - if maximum word repetition reached emit error, 
            #otherwise return
            if self.isMaxWordRepetionReaced(i_click_stats['letter_index']):
                print "WORD ERROR"
                click_stats['word_error'] = True
                selected_word = ""
            else:
                self.updateLetterDisplay(i_new_word=False)
        return (selected_word, click_stats)
    
    def __updateAfterWordSelect(self,  i_selected_word, i_is_calibrated):
        """Updating after a word was selected (not None)"""
        if i_selected_word == "":
            self.newWord(i_grnd_truth_word=self.getCurPhrase()[self.word_cnt], i_play_new_word=False)
        else:
            self.newWord(i_selected_word, i_play_new_word=False)
        if not i_is_calibrated: 
            self.manual_calibration = True
            self.action_calibrate.setChecked(False) 
            self.loadPhrase(i_is_calibrating =False)
        else:
            self.word_cnt += 1
            self.updateLetterDisplay(i_new_word=True)
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
                self.loadPhrase(i_is_calibrating=False) 
            else:
                self.audio.playInstructions(self.newWordStr())
                self.playCurrentWord()
                self.letter_offset=0
                for n in range(0, self.word_cnt):
                    self.letter_offset += ( len(phrase_words[n]) + 1 )
        self.adjustCursorToBeginWord()
        cur_word = self.getCurWord()
        letter_index = self.ticker_core.warpIndices(self.ticker_core.getLetterIndex(), len(cur_word))
        if letter_index < 0:
            letter_index += len(cur_word)
        if letter_index > 0:
            self.phrase_display.highLight(letter_index) 
        cur_letter = cur_word[letter_index]   
        print "IN UPDATE LETTER DIS new_word: cur_word = ", cur_word, " letter_index = ", letter_index, " current letter = " , cur_letter, 
        print " letter_offset = ", self.letter_offset
        cur_string = ("Letter likelihoods:  '" + str(cur_letter) + "'  ")
        self.label_letter_likelihoods.setText(cur_string)
    
    def adjustCursorToBeginWord(self):
        print "In update letter display calling highlight, letter offset = ", self.letter_offset
        self.phrase_display.highLight(0) 
        self.phrase_display.highLight(self.letter_offset) 
        
    ##################################### Save
 
    def saveWordStatistics(self, i_selected_word,  i_click_stats ):
        #Store click stats running up to selection
        click_stats = dict(i_click_stats)
        click_stats['total_time'] = time.time() - self.total_time  + self.inc_time
        click_stats['click_times'] = []
        if self.nclick_iterations > 0:
            click_stats['click_times'] = list(self.ticker_core.click_distr.train_obs[-self.nclick_iterations:])
        #Time in seconds
        click_stats['selected_word'] = str(i_selected_word)
        #If the maximum number of word repetions were reach there will be a key already
        if not click_stats.has_key('word_error'):
            if click_stats['grnd_truth'] == select_word:
                click_stats['word_error'] = False
            else:
                click_stats['word_error'] = True 
        click_stats['click_distr_before'] = copy.deepcopy( self.click_distr_before )
        click_stats['click_distr_after']  = copy.deepcopy(self.ticker_core.click_distr)
        self.utils.savePickle(click_stats, self.getSaveFileName(click_stats))
        self.saveSettings(self.settings_file) 
        print "******************************************************************"
        print "SAVED THE CLICK STATS TO FILE = ",self.getSaveFileName(click_stats)  
        print "SAVED SETTINGS FILE = ",self.settings_file   
        self.dispClickStats(click_stats)
        self.dispClickDistr(click_stats)
        self.dispSettings(click_stats)
       
    def dispClickStats(self, i_click_stats):
        print "CLICK STATS:"
        for key in i_click_stats.keys():
            if key == "settings":
                continue
            if key == "click_distr_before":
                continue
            if key == "click_distr_after":
                continue
            print "------------------"
            print key, " :" , i_click_stats[key]
    
    def dispClickDistr(self, i_click_stats):
        print "CLICK DISTR:"
        c_bef = i_click_stats['click_distr_before']
        c_aft = i_click_stats['click_distr_after']     
        print "Bef: (Delay, learn)=(%2.5f,%d), Aft: (Delay, learn)=(%2.5f,%d)" % (c_bef.delay,c_bef.learn_delay, c_aft.delay, c_aft.learn_delay)
        print "Bef: (Std  , learn)=(%2.5f,%d), Aft: (Std  , learn)=(%2.5f,%d)" % (c_bef.std,c_bef.learn_std,c_aft.std, c_aft.learn_std)
        print "Bef: (Fp   , learn)=(%2.5f,%d), Aft: (Fp   , learn)=(%2.5f,%d)" % (c_bef.fp_rate,c_bef.learn_fp, c_aft.fp_rate, c_aft.learn_fp)
        print "Bef: (Fr   , learn)=(%2.5f,%d), Aft: (Fr   , learn)=(%2.5f,%d)" % (c_bef.fr,c_bef.learn_fr, c_aft.fr, c_aft.learn_fr)
        print "Bef: (Learn, learn)=(%2.5f,%d), Aft: (Learn, learn)=(%2.5f,%d)" % (c_bef.learning_rate, c_bef.is_train, c_aft.learning_rate, c_aft.is_train) 
    
    def dispSettings(self, i_click_stats):
        print "EXPERIMENTAL SETTINGS:"
        for key in i_click_stats['settings']:
            print "------------------"
            print key, " :" , i_click_stats['settings'][key] 
        
    ################################################### Get
    
    def isMaxWordRepetionReaced(self, i_letter_index):
        """ * Test if the maximum word repetion is reached, in which case an error is submitted 
            * A user is only allowed to repeat any particular word twice. 
            * This function is called if clicks were received but not word was selected. 
            * So if clicks were received, no words selected and  self.word_repeat >= self.max_word_repeat,
               a word error will be generated.""" 
        letter_index = int(i_letter_index) + 1
        cur_word = self.getCurWord()
        self.word_repeat = int(float(letter_index) / len(cur_word))
        #Clicks were re
        if self.word_repeat < self.max_word_repeat:
            return False
        return True
    
    def getCurClickStats(self):
        """* This function is processed before a word is selected
           * If a word is selected then save these click stats (that's why there's an letter index offset of 1)""" 
        click_stats = {}
        #Settings editor
        click_stats = self.getClickStatsSettings(click_stats)
        self.obs_letters = [] 
        click_stats['nundo'] = self.nundo
        click_stats['word_cnt'] = self.word_cnt     
        click_stats['letter_index'] = self.ticker_core.getLetterIndex()
        click_stats['niterations'] = self.niterations + self.nundo
        click_stats['nclick_iterations'] = self.nclick_iterations
        click_stats['nclicks'] = self.nclicks
        click_stats['is_calibrated'] = (not self.action_calibrate.isChecked()) or self.calibrated
        click_stats['word_error'] = False
        click_stats['alphabet_read_time'] = 0.0
        if self.nclick_iterations > 0:
            click_stats['alphabet_read_time'] =  self.avg_alphabet_read_time / self.nclick_iterations
        phrase_words = self.getCurPhrase()   
        if phrase_words[self.word_cnt]  == '.':
            grnd_truth_word = '.'
        else:
            grnd_truth_word = phrase_words[self.word_cnt] + "_"
        click_stats['grnd_truth'] = str(grnd_truth_word) 
        return click_stats 
    
    def getClickStatsSettings(self , i_click_stats):
        click_stats = dict(i_click_stats)
        settings = dict(self.getSettings())
        #Noise settings
        settings['synthetic_noise'] = synthetic_noise
        if synthetic_noise:
            settings['switch_gauss_delay'] = self.noise.gauss_delay
            settings['switch_gauss_std'] = self.noise.gauss_std
            settings['switch_fp_rate'] = self.noise.fp_rate
            settings['switch_fr'] = self.noise.fr
            settings['switch_noise_delay'] = self.noise.delay
        click_stats['settings']  = dict(settings)
        return click_stats
    
    #File to save all results in
    def getSaveFileName(self, i_click_stats):
        if self.action_tutorial.isChecked():
            dir_name = "%s" % (self.tutorial_dir) 
        else:
            dir_name = "%s" % (self.output_dir) 
        if i_click_stats['is_calibrated']: 
            file_name = "%sclick_stats_%.2d_%.2d.cPickle" % (dir_name, self.phrase_cnt, self.word_cnt) 
        else:
            file_name = "%scalibration_stats_%.2d.cPickle" % (dir_name, self.word_cnt)
        return file_name
    
    def getCurPhrase(self):
        if self.isBusyCalibrating(): 
            cur_phrase = self.calibration_word
        else:
            cur_phrase = self.phrases[self.phrase_cnt]
        phrase_words = cur_phrase.split("_")[0:-1]
        if not self.isBusyCalibrating():
            phrase_words.append('.') 
        return phrase_words
    
    def getCurWord(self):
        phrase_words = self.getCurPhrase()
        print " CURRENT PHRASE = ", phrase_words, " IS CALIBRATE = ", self.isBusyCalibrating()
        print " WORD CNT = ", self.word_cnt
        if phrase_words[self.word_cnt]  == '.':
            cur_word = '.' 
        else:
            cur_word = phrase_words[self.word_cnt] + "_"
        return cur_word
    
    ######################################################## Noise
    
    def generateFalsePositives(self):
        if not synthetic_noise:
            return
        print "****************************************************"
        period = self.channel_config.getSoundTimes()[-1,-1] + self.settings_editor.box_end_delay.value()
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
        self.processClick()
      
    def stopTimers(self): 
        TickerGui.stopTimers(self)
        if not self.initialised:
            return
        for n in range(0, len(self.delay_noise_timers)):
            self.delay_noise_timers[n].stop()
         
    def setNoiseParameters(self):
        if not synthetic_noise:
            return
        if not session == 5:
            raise ValueError("Noise should only be tested in session 4") 
        self.settings_editor.box_end_delay.setValue(self.noise.delay)
        self.best_letters_timer.setInterval(1000*self.noise.delay)
        self.reset()
        print "AFTER SET: ", self.best_letters_timer.interval() 
        print "Set noise params: gaus_delay = ", self.noise.gauss_delay, " std = ", self.noise.gauss_std, 
        print " fp rate = ", self.noise.fp_rate, " fr = " , self.noise.fr,
        print " noise delay = ", self.noise.delay
       
if __name__ ==  "__main__":
    app = QtGui.QApplication(sys.argv)
    gui = TickerGuiExperiment()
    gui.show()
    sys.exit( app.exec_())

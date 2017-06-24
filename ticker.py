#! /usr/bin/env python

import sys, os,time, shutil, copy
import numpy as np
from PyQt4 import QtCore, QtGui,QtNetwork 
from ticker_layout import Ui_MainWindow
from ticker_widgets import ChannelVisualisation, AlphabetLabel, ClickGraphScene 
from ticker_widgets import DictionaryDisplay,InstructionsDisplay,SentenceDisplay
from ticker_audio import Audio
from channel_config import  ChannelConfig
from ticker_core import TickerCore
from settings_editor import VolumeEditWidget, SettingsEditWidget
from utils import Utils
from click_distr import ClickDistribution
 
class TickerGui(QtGui.QMainWindow, Ui_MainWindow):
    ##################################### Init
    def __init__(self, i_settings_dir="./settings/", i_settings_file="settings.cPickle"): 
        t=time.time() 
        QtGui.QMainWindow.__init__(self)  
        self.setupUi(self)
        self.utils = Utils()
        #######################################################################
        #Widget  instantiation
        #######################################################################
        self.settings_editor = SettingsEditWidget(self.centralwidget)
        channel_index = self.settings_editor.box_channels.currentIndex()
        overlap = self.getAudioOverlap()
        self.cur_dir = os.path.dirname(os.path.abspath(__file__)) + "/"
        file_length =  self.settings_editor.box_file_length.value()
        self.channel_config  = ChannelConfig(channel_index + 1, overlap, file_length, self.cur_dir)
        self.settings_editor.hide()
        self.alphabet_label_letter_likelihoods = AlphabetLabel(self.label_letter_likelihoods)
        self.letter_likelihood_display = ChannelVisualisation(  self.alphabet_likelihood_view, self.centralwidget, self.alphabet_label_letter_likelihoods )
        self.channel_names_display = ChannelVisualisation(  self.channel_names, self.centralwidget  )
        self.alphabet_display = ChannelVisualisation(self.alphabet_view, self.centralwidget )
        self.click_pdf_display = ClickGraphScene( parent=self.click_distribution_view, 
                                                  i_title=self.label_click_distribution,
                                                  xlabel="click time (seconds)", ylabel="P(letter | click time)")
        self.audio = Audio()
        self.click_pdf = ClickDistribution()
        word_thresh = self.settings_editor.box_word_select.value() 
        self.ticker_core = TickerCore(word_thresh, self.cur_dir + "dictionaries/nomon_dict.txt")       
        self.dictionary_display =  DictionaryDisplay( self.best_words_disp, self.centralwidget,  i_title=None)
        self.instructions_display = InstructionsDisplay(self.label_instructions, self.centralwidget, i_title=None)
        self.sentence_display = SentenceDisplay(self.selected_words_disp, self.centralwidget, i_title=None)
        #Main time calls update audio
        self.main_timer = QtCore.QTimer()
        #Best timer: after the alphabet has been played to the user, this time interval will pass before starting again
        self.best_letters_timer = QtCore.QTimer()
        #Give some time before continuing to the next letter
        self.best_letters_timer.setInterval(1000*self.settings_editor.box_end_delay.value())  
        self.best_letters_timer.setSingleShot(True)
        self.volume_editor = VolumeEditWidget(self.centralwidget)
        self.volume_editor.setChannelConfig(self.channel_config)
        #Hide/show wigets 
        self.initManualHideFlags()  
        self.volume_editor.hide()
        self.hidePhrases()
        self.actionAlphabet(False)
        #######################################################################
        #Complete initialisation of separate components, and connects all signals
        #######################################################################
        self.label_letter_speed.setText(QtCore.QString("Speed:  %0.2f" % overlap))
        #Keybord clicks - if False control happens via something else e.g., software port
        self.enable_clicks = True
        #The number of alphabet repetions
        self.repeat_count = 0
        #Calibrate variables
        self.calibration_word = "yes_"
        self.calibrated = False
        self.manual_calibration = False #Change calibration mode flag manually (not through button press)
        #Pause/play
        self.restart = True
        ##########################################################################
        #The port settings
        ##########################################################################
        self.socket = QtNetwork.QUdpSocket()
        is_socket = self.socket.bind(QtNetwork.QHostAddress(0),20320) 
        if not is_socket:
            print "Binding of socket was unsuccessful"
        else:
            QtCore.QObject.connect(self.socket,  QtCore.SIGNAL("readyRead()"),  self.readPendingDatagrams)
        ##########################################################################
        #Start main 
        ##########################################################################
        self.initSettings(i_settings_dir, i_settings_file)
        self.__connectSignals() 
        #Start the application by pretending to close the settings editor which resets everything 
        self.setFastMode(self.action_fast_mode.isChecked())
        self.setAudioOverlap()
        self.initSize()
        print "Total startup time, ticker = ", time.time() - t, " seconds "     #Connect signal/slots for all the action items
    
    #Reset everything - clickdistr estc
    def reset(self):
        self.audio.stop() 
        self.best_letters_timer.setInterval(1000*self.settings_editor.box_end_delay.value()) 
        file_length =  self.settings_editor.box_file_length.value()
        nchannels = self.settings_editor.getCurrentChannel()
        print "Calling set channel: ", nchannels, " file length = ", file_length, " wait time = ", self.best_letters_timer.interval()
        self.calibrated = False
        self.clearAll()
        self.channel_config.setChannels(nchannels, file_length)
        self.audio.setChannels(nchannels)
        alphabet = self.channel_config.getAlphabetLoader().getUniqueAlphabet( i_with_spaces=True, i_group=True)        
        self.letter_likelihood_display.setChannels(alphabet)
        alphabet =  self.channel_config.getChannelNames()
        self.channel_names_display.setChannels(alphabet)
        alphabet = self.channel_config.getAlphabetLoader().getAlphabet( i_with_spaces=True, i_group=True)        
        self.alphabet_display.setChannels( alphabet)
        self.setClickDistrParams()
        self.ticker_core.setClickDistr(self.click_pdf)
        self.initDisplayForNewWord()
        #The volume
        self.volume_editor.setChannelConfig(self.channel_config)
        #Make sure the volumes are set correctly
        for channel in range(0, nchannels):
            self.setVolume(self.volume_editor.getVolume(channel), channel)
    
    def initDisplayForNewWord(self):
        (words, word_probs) = self.ticker_core.getBestWordProbs(16)
        self.dictionary_display.update(words,word_probs) 
        self.setInstructLetterSelectStr()
        self.letter_likelihood_display.clear()
        self.drawClickDistribution()
            
    def initSize(self):
        width = 1100
        if (not self.alphabet_hidden) and (not self.phrases_hidden):
            self.resize(width, 830)
        elif not self.alphabet_hidden:
            self.resize(width, 730)
        elif not self.phrases_hidden:
            self.resize(width, 590) 
        else:
            self.resize(width, 530) 
            
    def initManualHideFlags(self):
        self.phrases_hidden = True
        self.alphabet_hidden = True
  
    def initSettings(self, i_settings_dir, i_settings_file):
        #Save the settings to default file
        self.settings_dir = i_settings_dir
        self.default_file = self.settings_dir + "default_settings.cPickle"
        self.setClickDistrParams()
        self.ticker_core.setClickDistr(self.click_pdf)
        self.click_pdf.initHistogram()
        self.saveSettings(self.default_file)
        self.settings_file = i_settings_dir + i_settings_file
        if os.path.exists(self.settings_file):
            self.loadSettings(self.settings_file)
            
    def initGaussDistribution(self):
        self.setClickDistrParams()
        self.click_pdf.initHistogram() 
        self.drawClickDistribution()

    def __connectSignals(self):
        #Menubar actions
        QtCore.QObject.connect( self.action_settings, QtCore.SIGNAL("triggered(bool)"), self.actionSettings)
        QtCore.QObject.connect( self.action_dictionary, QtCore.SIGNAL("triggered(bool)"), self.actionEditDictionary)
        QtCore.QObject.connect( self.action_close, QtCore.SIGNAL("triggered(bool)"), self.actionCloseApplication)
        QtCore.QObject.connect( self.action_clear, QtCore.SIGNAL("triggered(bool)"), self.actionClear)
        QtCore.QObject.connect( self.action_alphabet, QtCore.SIGNAL("toggled(bool)"), self.actionAlphabet)
        QtCore.QObject.connect( self.alphabet_widget_display, QtCore.SIGNAL("visibilityChanged(bool)"), self.actionCloseAlphabet)
        QtCore.QObject.connect( self.action_volume, QtCore.SIGNAL("triggered(bool)"), self.actionVolume)
        QtCore.QObject.connect( self.action_calibrate, QtCore.SIGNAL("toggled(bool)"),  self.setCalibration)
        QtCore.QObject.connect( self.action_fast_mode, QtCore.SIGNAL("toggled(bool)"),  self.setFastMode)
        QtCore.QObject.connect( self.action_practise, QtCore.SIGNAL("toggled(bool)"),  self.setPractise)
        QtCore.QObject.connect( self.action_open, QtCore.SIGNAL("triggered(bool)"),  self.loadSettingsDialog)
        QtCore.QObject.connect( self.action_save, QtCore.SIGNAL("triggered(bool)"),  self.saveSettingsDialog)
        #Speed scrollbar
        QtCore.QObject.connect( self.scrollbar_letter_speed, QtCore.SIGNAL("sliderReleased()"),  self.setAudioOverlap )
        QtCore.QObject.connect( self.scrollbar_letter_speed, QtCore.SIGNAL("sliderMoved(int)"),  self.setSliderLabel )
        #Start/stop/pause
        QtCore.QObject.connect( self.clear_button, QtCore.SIGNAL("clicked(bool)"),  self.startSoundFalse )
        #Pause/unpause
        QtCore.QObject.connect( self.button_pause, QtCore.SIGNAL("clicked(bool)"),  self.pauseSlot )
        #Timers
        QtCore.QObject.connect( self.main_timer, QtCore.SIGNAL("timeout()"), self.updateAudio)
        QtCore.QObject.connect( self.best_letters_timer, QtCore.SIGNAL("timeout()"), self.processClicks)
        #Volume editor
        QtCore.QObject.connect( self.volume_editor, QtCore.SIGNAL("volume(float,int)"), self.setVolume)
        #Settings editor - on closing the settings Ticker registers it
        QtCore.QObject.connect( self.settings_editor, QtCore.SIGNAL("close_settings"), self.reset)
        QtCore.QObject.connect( self.settings_editor, QtCore.SIGNAL("edit_click_params"), self.drawClickDistribution)
        QtCore.QObject.connect( self.settings_editor.button_default, QtCore.SIGNAL("released()"), self.loadDefaultSettings)
        QtCore.QObject.connect( self.settings_editor.button_gauss, QtCore.SIGNAL("released()"), self.initGaussDistribution)
       
    ##################################### Main functions
    
    def processClicks(self):
        self.audio.clear()
        #No clicks received
        if not self.ticker_core.clicksReceived():
            self.processAlphabetRepetions()
            return
        #Clicks were received - process it
        self.repeat_count = 0
        selected_word = self.processWordSelections()
        if selected_word is not None:
            #Update values in settings editor
            self.settings_editor.setClickParams(self.settings_editor.clickPdfToSettingsParams(self.click_pdf.getParams()))
            return
        self.updateNextLetter() 
    
    #Call this function if the click has to be processed
    def processClick(self):
        is_ready = not self.audio.isPlayingInstructions()
        if not is_ready:
            is_ready = self.audio.isReady()
        if not is_ready:
            return False
        click_time = self.audio.getTime(self.channel_config)
        print "In Ticker, click received, click_time = ", click_time
        click_log_scores = np.array(self.ticker_core.newClick(np.float64(click_time)))
        n_clicks = self.ticker_core.getNumberClicks() 
        #Undo
        if n_clicks >= self.settings_editor.box_undo.value():
            self.undoLastLetter()
            return True
        #Do not undo process clicks
        click_log_sum = self.utils.expTrick( click_log_scores.reshape([1, len(click_log_scores) ]) )[0]
        alpha_weights = np.exp(click_log_scores - click_log_sum)
        self.letter_likelihood_display.setAlphaWeights(alpha_weights)
        if self.action_practise.isChecked():
            print "************************************************************"
            print "Letter scores"
            print "************************************************************"
            for (n, letter) in enumerate(self.ticker_core.click_distr.alphabet):
                ltimes = self.ticker_core.click_distr.loc[n,:]
                print "%s prob=%1.3f, log_score=%2.3f" % (letter,alpha_weights[n],  click_log_scores[n]), 
                print " click time=%1.3f, letter_time=(%2.3f,%2.3f)" % (click_time,ltimes[0],ltimes[1]), 
                print " delta=(%2.3f,%2.3f)" % (click_time-ltimes[0],click_time-ltimes[1])
        return False
    
    def undoLastLetter(self, i_play_audio=True):
        self.ticker_core.undoLastLetter()
        self.repeat_count = 0
        self.audio.restart()
        self.letter_likelihood_display.clear()
        if i_play_audio:
            cmd_str = ["undo", "repeat"]
            self.playCurrentLetterIndex(cmd_str)
        
    def playCurrentLetterIndex(self, i_cmd_str=[]):
        cmd_str = list(i_cmd_str)
        letter_idx = self.instructions_display.letter_dict[self.ticker_core.getLetterIndex() + 1]
        cmd_str.extend(letter_idx.split(" "))
        cmd_str.append("letter")
        self.audio.playInstructions(cmd_str) 
        
    def updateNextLetter(self):
        """Update if selected word is None, proceeding to the next letter."""
        (words, word_probs) = self.ticker_core.getBestWordProbs(10)
        self.dictionary_display.update(words,word_probs)
        self.setInstructLetterSelectStr()
        self.letter_likelihood_display.clear()
        self.audio.playInstructions(["next"])
    
    def processWordSelections(self):
        #Check if we're busy with the calibration
        is_calibrated = (not self.action_calibrate.isChecked()) or self.calibrated
        is_process = is_calibrated and (not self.action_practise.isChecked())
        selected_word = self.ticker_core.newLetter(i_process_word_selections=is_process) 
        if self.action_practise.isChecked():
            return
        #Calibrating if process_word_selections = False
        if not is_calibrated:
            selected_word = self.processWordSelectCalibrating()
        #No word was selected
        if selected_word is None:
            return
        #A word was selected 
        if selected_word == ".":
            self.sentence_display.update(selected_word)
        else:
            self.sentence_display.update(selected_word[0:-1]) 
        #Don't play "new word" is a word was selected at calibration - this will be done when unchecking the calibration box
        self.newWord(selected_word, i_is_word_selected=True, i_play_new_word=is_calibrated)
        if not is_calibrated:
            self.manual_calibration = True
            self.action_calibrate.setChecked(False)
        return selected_word 
    
    def processWordSelectCalibrating(self):
        letter_idx = self.ticker_core.getLetterIndex()
        #only use the minimum number of alphabet repetions (with clicks) to select 
        #the calibration word and initialise the click distribution with.
        if letter_idx < len(self.calibration_word):
            return
        selected_word = str(self.calibration_word)
        print "In Ticker process word selection trainClickDistrAndInit, selected_word = ", selected_word
        self.ticker_core.trainClickDistrAndInitialise(selected_word)
        self.calibrated = True 
        return selected_word
   
    def processAlphabetRepetions(self):
        if self.action_practise.isChecked():
           return 
        #Process the number of times the alphabet sequence has been repeated when no clicks were received 
        self.repeat_count += 1
        shut_down_repeat = self.settings_editor.box_shut_down.value()
        word_repeat = self.settings_editor.box_restart_word.value()
        prog_repeat = self.settings_editor.box_prog_status.value()
        is_shut_down = self.repeat_count % shut_down_repeat
        is_new_word =  self.repeat_count % word_repeat
        is_prog_status = self.repeat_count % prog_repeat
        if is_shut_down is 0:
            repetion = self.instructions_display.letter_dict[shut_down_repeat]
            cmd_str = (repetion + " repetition reached shutting down").split(" ")  
            self.audio.playInstructions(cmd_str) 
            t = time.time()
            while self.audio.isPlayingInstructions():
                if (time.time() - t) > 5:
                    break
                if (time.time() - t) > 0.05:
                    self.audio.update(self.channel_config)
            self.startSoundFalse()
        elif is_new_word is 0: 
            self.newWord(i_play_new_word=False)
            self.audio.playInstructions(["undo", "restart", "word"])
        elif is_prog_status is 0:
            self.playCurrentLetterIndex()
        #else:
        #    self.audio.playInstructions(["beep"])

    def newWord(self, i_extra_command=None, i_is_word_selected=True, i_play_new_word=True): 
        self.ticker_core.newWord()
        self.initDisplayForNewWord()
        if self.action_practise.isChecked():
           return 
        if i_extra_command is not None:
            if i_is_word_selected:
                self.audio.synthesiseWord(i_extra_command) 
            else:
                self.audio.synthesise(i_extra_command)  
        if i_play_new_word:
            self.audio.playInstructions(self.newWordStr())
          
    #These functions are update functions synchronised with the GUI timer
    def updateAudio(self):
        if self.best_letters_timer.isActive():            
            (is_read_next, is_update_time, is_first_letter) = self.audio.update(self.channel_config,  i_loop=False)
            return (is_read_next, is_update_time, is_first_letter)
        (is_read_next, is_update_time, is_first_letter) = self.audio.update(self.channel_config)
        if is_read_next and (not self.best_letters_timer.isActive()):
            self.audio.readTime(self.channel_config)
            self.best_letters_timer.start()
            return (is_read_next, is_update_time, is_first_letter) 
        self.best_letters_timer.stop()
        if self.alphabet_display.isVisible():
            sound_index = self.audio.getSoundIndex(self.channel_config)
            self.alphabet_display.setColumnFocus(sound_index)
        return (is_read_next, is_update_time, is_first_letter)
                 
    def drawClickDistribution(self, i_gauss_params=None):
        self.click_pdf_display.drawClickDistribution(self.click_pdf.getHistogramRects())
        settings = self.settings_editor.getSettings()
        self.click_pdf_display.setView(settings['delay'],settings['std'])   
    
    def waitAudioReady(self, i_commands=None):
        if i_commands is not None:
            self.audio.playInstructions(i_commands)
        while  self.audio.isPlayingInstructions() or (not self.audio.isReady()):
            self.audio.update(self.channel_config)  
            
    ######################################### Settings
 
        
    def saveSettingsDialog(self):
        self.startSoundFalse()
        (disp_str, dir, files) = self.getSettingsDir()
        filename = QtGui.QFileDialog.getSaveFileName( self, disp_str, dir , files )
        print "SAVE SETTINGS, filename = ", filename
        if len(filename) > 0:
            self.saveSettings(filename)
          
    def saveSettings(self, i_file):
        #i_init_settings_backup=True: Initialising the settings 
        settings = dict(self.getSettings())
        settings['click_pdf'] = copy.deepcopy(self.click_pdf)
        print "SAVING SETTINGS"
        print "d = ", settings['delay'], " std = ", settings['std'], " fr =  ", settings['fr'], " ", settings['fp_rate']
        self.utils.savePickle(settings,i_file)
     
    def loadSettingsDialog(self):
        self.startSoundFalse()   
        (disp_str, dir, files) = self.getSettingsDir()
        filename = QtGui.QFileDialog.getOpenFileName( self,disp_str, dir , files )
        if len(filename) > 0:
            print "LOADNG SETTINGS, filename = ", filename
            self.loadSettings(filename)
        
    def loadDefaultSettings(self):
        print "LOADING DEFAULT SETTTING"
        self.loadSettings(self.default_file)
        self.drawClickDistribution()
        
    def loadSettings(self, i_file):
        print "loading settings from filename = ", i_file
        settings = self.utils.loadPickle(i_file)
        print "GOT SETTINGS:"
        print "delay = ", settings['delay'], " std = ", settings['std']
        #Settings editor
        self.settings_editor.setSettings(settings)
        #The speed
        self.scrollbar_letter_speed.setValue(int(settings['overlap']*100))
        self.__setAudioOverlap()
        self.best_letters_timer.setInterval(1000*self.settings_editor.box_end_delay.value())
        print "SPEED = ", settings['overlap'], " WAIT TIME = ", self.best_letters_timer.interval()
        #Mode
        self.action_fast_mode.setChecked(settings['fast_mode'])
        self.action_tutorial.setChecked(settings['tutorial'])
        self.action_calibrate.setChecked(settings['calibrate'])
        self.action_practise.setChecked(settings['practise'])
        self.action_inc_phrases.setChecked(settings['inc phrases']) 
        #The click pdf
        self.click_pdf = copy.deepcopy(settings['click_pdf'])
        self.ticker_core.setClickDistr(self.click_pdf)
        
        
        print "LOADED SETTINGS:  click_distr params = "
        click_params = self.click_pdf.getParams()
        (delay, std, fr, fp_rate) = click_params
        print "delay = ", delay, " std = ", std, " fr = ", fr, " fp rate = ", fp_rate  
        print "LOADED SETTINGS:  settings  params = "
        s = self.settings_editor.getSettings()
        print "delay = ", s['delay'], " std = ", s['std']
    
    #################################### Start/Stop/Pause/Close
            
    def pauseSlot(self, i_checked):
        if i_checked:
            self.pauseTrue()
            if self.restart:
                self.restart = False
                self.startSoundTrue()
                self.newWord()
            self.main_timer.start(10)
        else:
            self.pauseFalse()

    def pauseTrue(self, i_play_cur_letter_idx=True):
        self.audio.setChannels(self.settings_editor.getCurrentChannel())
        if i_play_cur_letter_idx:
            self.playCurrentLetterIndex()
        self.button_pause.setChecked(True)
        self.button_pause.setText("Pause") 
       
    def pauseFalse(self, i_undo_last=True):
        self.button_pause.setText("Play")
        self.button_pause.setChecked(False)
        self.stopTimers()
        self.audio.stop()
        if i_undo_last:
            self.undoLastLetter(i_play_audio=False)
       
    def startSoundTrue(self):
        self.clearAll()
        self.audio.restart()
        
    def startSoundFalse(self):
        self.clearAll()
        self.audio.stop()
        self.pauseFalse(i_undo_last=False)
        self.restart = True
    
    def closeEvent(self, event):  
        self.startSoundFalse()
        while not self.audio.isReady():
            continue
        QtGui.QMainWindow.close(self)
        
    def clearAll(self):
        self.stopTimers()
        self.repeat_count = 0  
       
    def stopTimers(self):
        self.main_timer.stop()
        self.best_letters_timer.stop()
     
    #################################### Switch Events
    
    def keyPressEvent(self, event):   
        if (event.key() == QtCore.Qt.Key_Space) and self.enable_clicks and (not self.button_pause.isChecked()):
            self.pauseSlot(True)
        if (not self.enable_clicks) or (not self.action_space_bar.isChecked()):
            return  
        if event.key() == QtCore.Qt.Key_Space:
            self.processClick()
            
    #Enable/Disable Keyboard events
    def enableClicks(self):
        self.enable_clicks = True
    
    def disableClicks(self):
        self.enable_clicks = False
    
    #Software port communication
    def readPendingDatagrams(self):
        if not self.action_port.isChecked():
            return
        while (self.socket.hasPendingDatagrams()):
            max_len = self.socket.pendingDatagramSize()
            (data,  host, port)  = self.socket.readDatagram (max_len) 
            for n in range(0, max_len):
                self.processClick()
                self.audio.playClick()
             
    ##################################### Feedback Phrases
    
    def setInstructLetterSelectStr(self):
        if self.action_practise.isChecked():
           return 
        letter_idx = self.ticker_core.getLetterIndex()
        disp_str = str(self.instructions_display.getInstructSentence(letter_idx+1))
        if (not self.calibrated) and self.action_calibrate.isChecked():
            disp_str += (" " + self.calibration_word )
            letter = self.calibration_word[letter_idx]
        self.instructions_display.update(disp_str)
        
    def newWordStr(self):
        if (not self.calibrated) and self.action_calibrate.isChecked():
            instruct_str = ["calibrating"]    
        else:
            instruct_str = ["start"]    
        return instruct_str
    
    ##################################### Set functions

    def setClickDistrParams(self):
        s = self.settings_editor.getSettings() 
        self.click_pdf.setParams(s['is_train'], self.channel_config, s['delay'], s['std'], s['fp_rate'], 
            s['fr'], s['learning_rate'],s['end_delay'])
        self.click_pdf.setFixLearning(s['learn_delay'], s['learn_std'], s['learn_fp'], s['learn_fr'])
        
    def setSliderLabel(self, i_value):
        overlap =  "%.2f" % (i_value/100.0) 
        self.label_letter_speed.setText(QtCore.QString("Speed:  %s" % overlap))
        
    def setAudioOverlap(self):
        self.startSoundFalse()
        overlap = self.__setAudioOverlap()
        file_length =  self.settings_editor.box_file_length.value()
        self.channel_config.setOverlap(overlap, file_length)     
        self.reset()
        
    def __setAudioOverlap(self):
        overlap = self.getAudioOverlap()
        self.setSliderLabel(100.0 * overlap)   
        return overlap
    
    def setVolume(self, i_val, i_channel):
        self.audio.setVolume(i_val, i_channel)
        
    def setCalibration(self, i_checked): 
        self.calibrated = not i_checked 
        if not self.manual_calibration:
            self.startSoundFalse()
        self.manual_calibration = False

    def setPractise(self, i_checked):
        self.startSoundFalse()

    def setFastMode(self, i_checked): 
        self.startSoundFalse()
        if i_checked:
            alphabet_dir = "alphabet_fast/"
            self.settings_editor.box_file_length.setValue(0.21)
        else:
            alphabet_dir = "alphabet_slow/"
            self.settings_editor.box_file_length.setValue(0.4) 
        self.audio.setAlphabetDir(alphabet_dir)
        self.reset()
        
    ##################################### Get functions
    
    def isBusyCalibrating(self):
        return (self.action_calibrate.isChecked()) and (not self.calibrated)
 
    def getAudioOverlap(self):
        val = 0.01*float(self.scrollbar_letter_speed.value())
        val = float(str( "%.2f" % val))
        return val
 
    def getSettings(self):
        settings = self.settings_editor.getSettings()
        #Speed and Channel settings
        settings['overlap'] = self.getAudioOverlap()
        #Mode
        settings['fast_mode'] = self.action_fast_mode.isChecked()
        settings['tutorial'] = self.action_tutorial.isChecked()
        settings['calibrate'] = self.action_calibrate.isChecked()
        settings['practise'] = self.action_practise.isChecked()
        settings['inc phrases'] = self.action_inc_phrases.isChecked()
        #Click pdf
        settings['click_pdf'] = copy.deepcopy(self.click_pdf)
        return settings
    
    def getSettingsDir(self):
        disp_str = "Select output file"
        files = "cPickle Files (*.cPickle)"
        return (disp_str, self.settings_dir, files)

    ##################################### Actions
    
    def actionClear(self):
        self.sentence_display.clear()
        self.startSoundFalse()
 
    def actionSettings(self, i_checked):
        self.startSoundFalse()
        self.settings_editor.show()
        
    def actionVolume(self, i_checked):
        self.volume_editor.show()
    
    def actionAlphabet(self, i_checked): 
        self.action_alphabet.setChecked(i_checked)
        if  i_checked:
            self.showAlphabetSeq()
        else:
            self.hideAlphabetSeq()   
   
    def actionCloseAlphabet(self, i_visible):
        if not i_visible:
            self.action_alphabet.setChecked(False)
        self.adjustSize()
        
    def actionEditDictionary(self, i_trigger):
        self.startSoundFalse()
        dir = self.cur_dir + "dictionaries/"
        filename = QtGui.QFileDialog.getOpenFileName( self, "Select dictionary", dir, "Text Files (*.txt)");
        if len(filename) > 0:
            self.ticker_core.setDict(filename)
            self.initDisplayForNewWord()
        
    def actionCloseApplication(self):
        self.close()
        
    ################################################# Hide show functions
        
    def hidePhrases(self):
        self.phrases_hidden = True
        self.phrase_disp.hide()
        self.label_phrases.hide()
        self.initSize()
        
    def showPhrases(self):
        self.phrases_hidden = False
        self.phrase_disp.show()
        self.label_phrases.show()
        self.initSize()
        
    def hideAlphabetSeq(self):
        self.alphabet_hidden = True
        self.alphabet_widget_display.hide()
        self.initSize()
        
    def showAlphabetSeq(self):
        self.alphabet_hidden = False
        self.alphabet_widget_display.show()
        self.initSize()
  
if __name__ ==  "__main__":
    app = QtGui.QApplication(sys.argv)
    gui = TickerGui()
    gui.show()
    sys.exit( app.exec_())

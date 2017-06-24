#! /usr/bin/env python

import sys, os,time, shutil, copy
sys.path.append("../../")   
import numpy as np
from PyQt4 import QtCore, QtGui,QtNetwork 
from grid_layout import Ui_MainWindow
from ticker_audio import Audio 
from utils import Utils
from ticker_widgets import InstructionsDisplay,SentenceDisplay
from grid_config import ChannelConfig
 
class GridGui(QtGui.QMainWindow, Ui_MainWindow):
    ##################################### Init
    def __init__(self):  
        t=time.time() 
        QtGui.QMainWindow.__init__(self)  
        self.setupUi(self)
        self.utils = Utils()
        #######################################################################
        #The grid settings
        #######################################################################
        self.main_timer_delay = 10             #Timer delay in ms 
        self.n_prog_status = 2                 #Number of iterations before reading program status
        self.n_undo_last = 4                   #Number iterations before undo last action
        self.delete_char = '$'                 #The user writes this character to delete the previous character
        scan_delay = self.getScanDelay() 
        #######################################################################
        #Widget  instantiation
        #######################################################################
        self.cur_dir = os.path.dirname(os.path.abspath(__file__)) + "/"
        self.config_dir =   self.cur_dir + "grid_config/" 
        self.audio = Audio(i_root_dir=self.config_dir)
        self.instructions_display = InstructionsDisplay(self.label_instructions, self.centralwidget, i_title=None)
        self.sentence_display = SentenceDisplay(self.selected_words_disp, self.centralwidget, i_title=None)
        self.channel_config  = ChannelConfig(scan_delay,self.config_dir)
        #Main time calls update audio
        self.main_timer = QtCore.QTimer()
        #Best timer: after the alphabet has been played to the user, this time interval will pass before starting again
        self.best_letters_timer = QtCore.QTimer()
        self.best_letters_timer.setInterval(0)  #Give some time before continuing to the next letter
        self.best_letters_timer.setSingleShot(True)
        #######################################################################
        #Complete initialisation of separate components, and connects all signals
        #######################################################################
        self.setSliderLabel(scan_delay*10.0) 
        self.audio.setTicks(1)   #The number tick sounds before playing the alphabet 
        #Pause/play
        self.restart = True
        self.hideRecordingWidgets()
        #Load initial settings from file
        #self.initSettings(i_settings_dir, i_settings_file)
        self.initDisplayForNewWord()
        self.setInstructLetterSelectStr() 
        self.__connectSignals()
        
    #Reset everything - clickdistr estc
    def reset(self):
        self.audio.setAlphabetDir(self.channel_config.alphabet_dir)
        self.audio.setConfigDir(self.channel_config.config_dir) 
        self.audio.setChannels(1)    
      
    def initDisplayForNewWord(self):
        self.repeat_count = 0  
        self.letter_idx = 0
        self.delete_cnt = 0
        self.nclicks = 0
        self.selected_word = None
        self.click_times = []
        self.scan_delay = self.getScanDelay()
        self.tutorial = self.action_tutorial.isChecked()
        self.undo_last_action_cnt = 0
        self.delete_cnt = 0
        self.nscans = [] 
        self.setInstructLetterSelectStr()
        self.setRowScan()
        
    def __connectSignals(self):
        #Menubar actions
        QtCore.QObject.connect( self.action_close, QtCore.SIGNAL("triggered(bool)"), self.actionCloseApplication)
        QtCore.QObject.connect( self.action_clear_sentence, QtCore.SIGNAL("triggered(bool)"), self.actionClear)
        QtCore.QObject.connect( self.action_tutorial, QtCore.SIGNAL("toggled(bool)"),  self.setTutorial)
        #Speed scrollbar
        QtCore.QObject.connect( self.scrollbar_letter_speed, QtCore.SIGNAL("sliderReleased()"), self.setScanDelay )
        QtCore.QObject.connect( self.scrollbar_letter_speed, QtCore.SIGNAL("sliderMoved(int)"), self.setSliderLabel )
        #Start/stop/pause
        QtCore.QObject.connect( self.clear_button, QtCore.SIGNAL("clicked(bool)"),  self.startSoundFalse )
        #Pause/unpause
        QtCore.QObject.connect( self.button_pause, QtCore.SIGNAL("clicked(bool)"),  self.pauseSlot )
        #Timers
        QtCore.QObject.connect( self.main_timer, QtCore.SIGNAL("timeout()"), self.update)
        QtCore.QObject.connect( self.best_letters_timer, QtCore.SIGNAL("timeout()"), self.processAlphabetRepetions)
          
    ##################################### Main functions
    
    def setRowScan(self):
        self.channel_config.setRowScan()
        self.startNewScan()
        
    def setColScan(self):
        self.stopTimers()
        sound_idx = self.audio.getSoundIndex(self.channel_config)
        id = self.channel_config.alphabet.getAlphabet(i_with_spaces=False, i_group=False)[sound_idx]  
        self.waitAudioReady([id])
        self.channel_config.setColScan(id)
        self.startNewScan() 

    def startNewScan(self):
        self.reset()
        if self.button_pause.isChecked() and (not self.main_timer.isActive()): 
            self.main_timer.start()
            self.audio.clear() 
        self.nscans.append(0)
        print "NSCANS INIT: ", self.nscans
            
    def waitAudioReady(self, i_commands=None):
        if i_commands is not None:
            self.audio.playInstructions(i_commands)
        while  self.audio.isPlayingInstructions() or (not self.audio.isReady()):
            self.audio.update(self.channel_config)  
            
    #Call this function if the click has to be processed
    def processClick(self):
        self.stopTimers()
        click_time = self.audio.getTime(self.channel_config)
        sound_idx = self.audio.getSoundIndex(self.channel_config)
        #self.waitAudioReady(['click'])
        self.nclicks += 1
        self.selected_word = None
        self.repeat_count = 0
        if self.audio.isPlayingInstructions():
            return 
        print "ADDING : TO NSCANS FROM CLICK: ", self.audio.sound_index+1, " NSCANS BEFORE = ", self.nscans
        self.nscans[-1] += (self.audio.sound_index+1) 
        print "NSCANS NOW ", self.nscans
        print " Click time in processClick: ", self.audio.getTime(self.channel_config)
        self.click_times.append(click_time)
        if self.channel_config.row_mode:
            self.setColScan()
            return
        alphabet = self.channel_config.alphabet.getAlphabet(i_with_spaces=False, i_group=False)
        letter = alphabet[sound_idx]
        print "selected letter " , letter
        if letter == self.delete_char: 
            self.deleteLastLetter()
            self.delete_cnt += 1
        elif (letter == "_") or (letter == "."):
            self.selected_word = self.processWord(letter)
        else:
            self.updateNextLetter(letter)
        self.setRowScan()

    def processWord(self, i_letter, i_play_new_word=True):
        selected_word = ""
        self.sentence_display.update(i_letter, i_adjust_stop=False, i_add_space=False)  
        if i_letter == ".":
            selected_word = "."
        else: 
            if self.letter_idx > 0:
                selected_word = self.sentence_display.lastWord()
            self.waitAudioReady(['written', "_"]) 
        print "SELECTED WORD = ", selected_word
        if not selected_word == "": 
            self.newWord(selected_word,  i_is_word_selected=True, i_play_new_word=i_play_new_word)
        else:
            self.newWord(i_is_word_selected=False, i_play_new_word=i_play_new_word)
        self.waitAudioReady()
        return selected_word
 
    def deleteLastLetter(self):
        self.letter_idx -= 1
        self.repeat_count = 0
        is_delete = True
        if self.letter_idx < 0:
            is_delete = False
            self.letter_idx = 0
        else:
            self.sentence_display.deleteLastLetter()
        self.newLetter(['written', self.delete_char])
        return is_delete
        
    def updateNextLetter(self, i_letter):
        self.letter_idx += 1
        self.sentence_display.update(i_letter, i_adjust_stop=False, i_add_space=False)
        self.newLetter(['written', i_letter, "next"])
    
    def newLetter(self, i_cmd_str):
        self.waitAudioReady(i_cmd_str)
        #cmd_str = self.currentLetterIndexStr()
        #self.waitAudioReady(cmd_str)
        self.repeat_count=0
        self.setInstructLetterSelectStr()

    def playCurrentLetterIndex(self, i_cmd_str=[]):
        cmd_str = self.currentLetterIndexStr(i_cmd_str)
        self.audio.playInstructions(cmd_str) 
        
    def currentLetterIndexStr(self, i_cmd_str=[]):
        cmd_str = list(i_cmd_str)
        letter_idx = self.instructions_display.letter_dict[self.letter_idx + 1]
        cmd_str.extend(letter_idx.split(" "))
        cmd_str.append("letter")
        return cmd_str
       
    def processAlphabetRepetions(self):
        #Process the number of times the alphabet sequence has been repeated when no clicks were received 
        self.repeat_count += 1 
        self.nscans[-1] += (self.channel_config.getSoundTimes().shape[0]) 
        self.audio.clear()
        is_undo_last = self.repeat_count % self.n_undo_last
        is_prog_status = self.repeat_count % self.n_prog_status
        if (is_undo_last is 0) and (not self.channel_config.row_mode):
            self.undo_last_action_cnt += 1
            self.newLetter(['undo'])
            self.setRowScan()
        elif is_prog_status is 0:
            self.playCurrentLetterIndex()
    
    def newWord(self, i_extra_command=None, i_is_word_selected=True, i_play_new_word=True):  
        self.initDisplayForNewWord()
        if i_extra_command is not None:
            print "i_extra_command = ", i_extra_command
            if i_is_word_selected:
                self.audio.synthesiseWord(i_extra_command) 
            else:
                self.audio.synthesise(i_extra_command)  
        if i_play_new_word:
            print "PLAY NEW WORD"
            self.audio.playInstructions(self.newWordStr())
          
    #These functions are update functions synchronised with the GUI timer
    def update(self):
        if self.best_letters_timer.isActive():            
            (is_read_next, is_update_time, is_first_letter) = self.audio.update(self.channel_config,  i_loop=False)
            return (is_read_next, is_update_time, is_first_letter)
        (is_read_next, is_update_time, is_first_letter) = self.audio.update(self.channel_config)
        if is_read_next and (not self.best_letters_timer.isActive()): 
            self.audio.readTime(self.channel_config)
            self.best_letters_timer.start()
            return (is_read_next, is_update_time, is_first_letter)
        self.best_letters_timer.stop()
        return (is_read_next, is_update_time, is_first_letter)
        
    #################################### Start/Stop/Pause/Close
            
    def pauseSlot(self, i_checked):
        if i_checked:
            self.pauseTrue()
            if self.restart:
                self.restart = False
                self.startSoundTrue()
                self.newWord()
            self.main_timer.start(self.main_timer_delay)
        else:
            self.pauseFalse()

    def pauseTrue(self, i_play_cur_letter_idx=True):
        self.startSoundTrue()
        self.setRowScan()
        if i_play_cur_letter_idx:
            self.playCurrentLetterIndex()
        self.button_pause.setChecked(True)
        self.button_pause.setText("Pause") 
       
    def pauseFalse(self, i_undo_last=True):
        self.button_pause.setText("Play")
        self.button_pause.setChecked(False)
        self.stopTimers()
        self.audio.stop()
       
    def startSoundTrue(self):
        self.audio.clear()
        self.audio.restart()
        
    def startSoundFalse(self):
        self.audio.clear()
        self.audio.stop()
        self.pauseFalse(i_undo_last=False)
        self.restart = True
         
    def closeEvent(self, event):   
        self.startSoundFalse()
        while not self.audio.isReady():
            continue
        QtGui.QMainWindow.close(self)
     
    def stopTimers(self):
        self.main_timer.stop()
        
    #################################### Switch Events
    
    def keyPressEvent(self, event):   
        if (event.key() == QtCore.Qt.Key_Space) and (not self.button_pause.isChecked()):
            self.pauseSlot(True)
        if event.key() == QtCore.Qt.Key_Space:
            self.processClick()
    
    ##################################### Feedback Phrases
    
    def setInstructLetterSelectStr(self):
        disp_str = str(self.instructions_display.getInstructSentence(self.letter_idx+1))
        self.instructions_display.update(disp_str)
        
    def newWordStr(self):
        instruct_str = ["start"]    
        return instruct_str
    
    ##################################### Set functions
 
    def setSliderLabel(self, i_value):
        scan_delay =  "%.2f" % (i_value/10.0) 
        self.label_letter_speed.setText(QtCore.QString("Scan delay (seconds):  %s" % scan_delay))
        
    def setScanDelay(self):
        self.setNewSetting()
        scan_delay = self.__setScanDelay()
        print "Scan delay  after set scan delay = ", scan_delay
        self.channel_config.setScanDelay(scan_delay)
        
    def __setScanDelay(self):
        scan_delay = self.getScanDelay()
        self.setSliderLabel(10.0 * scan_delay)   
        return scan_delay
    
    def setTutorial(self, i_checked):
        self.setNewSetting()

    def setNewSetting(self):
        self.startSoundFalse()
        self.setRowScan()
        self.initDisplayForNewWord()

    ##################################### Get functions
     
    def getScanDelay(self):
        val = 0.1*float(self.scrollbar_letter_speed.value()) 
        val = float(str( "%.2f" % val))
        return val
 
    ##################################### Actions
    
    def actionClear(self):
        self.sentence_display.clear()
        self.setNewSetting()

    def actionCloseApplication(self):
        self.close()
        
    #################################### Show/Hide recordings widgets
    
    def hideRecordingWidgets(self):
        self.phrase_disp.hide()
        self.label_phrases.hide()
        self.action_inc_phrases.setVisible(False)
    
    def showRecordingWidgets(self):
        self.phrase_disp.show()
        self.label_phrases.show()
        self.action_inc_phrases.setVisible(True)
        
  
if __name__ ==  "__main__":
    app = QtGui.QApplication(sys.argv)
    gui = GridGui()
    gui.show()
    # gui.showRecordingWidgets()
    sys.exit( app.exec_())

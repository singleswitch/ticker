#! /usr/bin/env python

import sys
sys.path.append('../../')

from PyQt4 import QtCore, QtGui
import time
from ticker_audio import Audio
from channel_config import  ChannelConfig

class AudioOutput(QtGui.QTextEdit):
    
    def __init__(self, i_parent=0):
        QtGui.QTextEdit.__init__(self, i_parent)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.sizePolicy().hasHeightForWidth())
        self.setSizePolicy(sizePolicy)
        self.setMinimumSize(QtCore.QSize(500, 300))
        self.setMaximumSize(QtCore.QSize(851, 16777215))
        font = QtGui.QFont()
        font.setFamily("Monospace")
        font.setPointSize(10)
        self.setFont(font)
        self.setFocusPolicy(QtCore.Qt.NoFocus)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.setObjectName("selected_words_disp")
    
    def write(self, i_str):
        self.insertPlainText(QtCore.QString(i_str + " "))
        self.ensureCursorVisible()
        
    
class AudioDisplay(QtGui.QWidget):
    def __init__(self, i_nchannels=5, i_overlap=0.2,  parent=None):
        QtGui.QWidget.__init__(self,  parent)
        self.__config=ChannelConfig( i_nchannels,i_overlap, 0.21, "../../" )
        self.__audio = Audio( i_nchannels, i_root_dir="../../" )
        self.__audio.setAlphabetDir("alphabet_fast/")
        self.__start_button = QtGui.QPushButton(self)
        self.__start_button.setCheckable(True)
        self.__start_button.setObjectName("Alphabet start or stop")
        self.__start_button.setText("Start")
        self.__start_button.setFocusPolicy(QtCore.Qt.NoFocus)
        
        self.__audio_out = AudioOutput(self)
        self.__cur_letter_out = AudioOutput(self)
        v_layout = QtGui.QVBoxLayout()  
        v_layout.addWidget(self.__start_button) 
        v_layout.addWidget(self.__audio_out)    
        v_layout.addWidget( self.__cur_letter_out )    
        self.setLayout(v_layout)
        self.connect( self.__start_button, QtCore.SIGNAL("clicked(bool)"), self.start )
        self.__timer = QtCore.QTimer()
        QtCore.QObject.connect(self.__timer, QtCore.SIGNAL("timeout()"), self.update)
        self.__start_time = 0.0
        
        #Some status variables
        self.__enable_clicks = True #Accept key presses
        self.__display_cur_letters = False#Display the current letter in focus, all the time
        self.__prev_letter = '*'
    
    def closeEvent(self, event): 
        self.__audio.close()
        QtGui.QWidget.close(self)
    
    def keyPressEvent(self, event):        
        if not self.__enable_clicks:
            return 
        if event.key() == QtCore.Qt.Key_Space:
            click_time = self.__audio.getTime()
            alphabet =  self.__config.getAlphabetLoader().getAlphabet( i_with_spaces=False)
            letter_index = self.__audio.getSoundIndex()
            letter = alphabet[letter_index]
            disp_str = " Group: %s, Click time: %0.2f \n" %(letter, click_time)
            disp_str = QtCore.QString(disp_str)
            self.__audio_out.write(disp_str)
 
    def start(self, checked):
        self.__prev_letter = '*'
        if checked:
            self.__audio.restart()
            self.__start_button.setText("Stop")
            self.__timer.start(20)
            self.__start_time = time.time()
            self.__audio.playNext()
        else:
            self.__audio.stop()
            self.__start_button.setText("Start")
            self.__timer.stop()
            
    def update(self):
        update = self.__audio.update(self.__config)
        if update:
            end_time = time.time()
            disp_str = " Audio done: %0.2f seconds\n" % (end_time - self.__start_time )
            self.__start_time = end_time
            self.__audio_out.write(disp_str)
        else:
            if self.__display_cur_letters:
                alphabet = self.__config.getAlphabetLoader().getAlphabet( i_with_spaces=False)
                letter_index = self.__audio.getSoundIndex()
                letter = alphabet[letter_index]
                if not (letter == self.__prev_letter):
                    self.__prev_letter = letter
                    disp_str = letter + "\n"
                    self.__cur_letter_out.write(QtCore.QString(letter + "\n"))
                
 
if __name__ ==  "__main__":
    # FIXME - It would be good to get this test running again.
    #         The issue seems to be related to setting the root_dir
    #         and alphabet dir on the Audio class.
    raise RuntimeError("This test is not currently functional "
                       "- use the C++ one instead.")
    app = QtGui.QApplication(sys.argv)
    disp = AudioDisplay(i_nchannels=5, i_overlap=0.1)
    disp.show()
    retVal = app.exec_()
    sys.exit(retVal)

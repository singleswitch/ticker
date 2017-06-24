 
#FIXME: UNDO, click time at end to undo

from PyQt4 import QtCore, QtGui
import sys, os
import volume_editor_layout, settings_layout, cPickle
import numpy as np
from utils import Utils

class SettingsEditWidget(QtGui.QDialog, settings_layout.Ui_Dialog):
    #################################################### Init
    def __init__(self, i_parent=None):
        QtGui.QDialog.__init__(self, i_parent)  
        self.setupUi(self) 
        QtCore.QObject.connect( self.box_enable_learning, QtCore.SIGNAL("toggled(bool)"), self.setEnableLearning)
        QtCore.QObject.connect( self.box_seconds_delay, QtCore.SIGNAL("valueChanged(double)"),self.editClickParamsEvent)
        QtCore.QObject.connect( self.box_click_dev, QtCore.SIGNAL("valueChanged(double)"),self.editClickParamsEvent)
       
    ############################################### Main
    
    def editClickParamsEvent(self, i_value): 
        self.emit(QtCore.SIGNAL("edit_click_params")) 
        
    def closeEvent(self, event): 
        QtGui.QDialog.close(self)
        self.emit(QtCore.SIGNAL("close_settings"))

    def clickPdfToSettingsParams(self, i_params):
        """Convert click pdf parameters to the ones stored in settings editor."""
        (delay, std, fr, fp_rate) = i_params
        fr *= 100.0
        fp_rate *= 60.0
        return (delay, std, fr, fp_rate)
    
    def settingsToClickPdfParams(self, i_params):
        """Convert settings editorparameters to the ones stored by click pdf."""
        (delay, std, fr, fp_rate) = i_params
        fr /= 100.0
        fp_rate /= 60.0
        return (delay, std, fr, fp_rate)
 
    ################################################ Get
    
    def getSettings(self):
        settings = {}
        #Click-time delay
        delay = self.box_seconds_delay.value()
        std  =  self.box_click_dev.value() 
        settings['is_train'] = self.box_enable_learning.isChecked()
        settings['learning_rate'] = self.box_learning_rate.value()
        settings['learn_delay'] = self.box_learn_delay.isChecked()
        settings['learn_std'] = self.box_learn_std.isChecked()
        #Switch noise
        fp_rate  = self.box_fp_rate.value() 
        fr = self.box_fr.value() 
        settings['learn_fp'] = self.box_learn_fp.isChecked()
        settings['learn_fr'] = self.box_learn_fr.isChecked()
        #Do the conversion
        click_params = (delay, std, fr, fp_rate)
        (settings['delay'], settings['std'], settings['fr'], settings['fp_rate']) = self.settingsToClickPdfParams(click_params)
        #Error correction
        settings['undo']  = self.box_undo.value()
        settings['prog_status'] = self.box_prog_status.value()
        settings['restart_word'] = self.box_restart_word.value()
        settings['shut_down'] = self.box_shut_down.value()
        settings['word_select_thresh'] = self.box_word_select.value()
        #Speed & channels
        settings['file_length'] = self.box_file_length.value()
        settings['channel_index'] = int(self.box_channels.currentIndex()) 
        settings['end_delay'] = self.box_end_delay.value() 
        return settings
    
    def getCurrentChannel(self):
        return self.getChannel(self.box_channels.currentIndex())
    
    def getChannel(self, i_index):
        return int(self.box_channels.itemText(i_index))
    
    #################################################### Set

    def setSettings(self, i_settings):
        #Get the parameters
        click_params = (i_settings['delay'], i_settings['std'], i_settings['fr'], i_settings['fp_rate']) 
        (delay, std, fr, fp_rate) = self.clickPdfToSettingsParams(click_params)
        self.setClickParams((delay, std, fr, fp_rate))
        #More click-time params  
        self.box_enable_learning.setChecked(i_settings['is_train'])
        self.box_learning_rate.setValue(i_settings['learning_rate'])
        self.box_learn_delay.setChecked( i_settings['learn_delay'])
        self.box_learn_std.setChecked(i_settings['learn_std'])
        #More switch noise params
        self.box_learn_fp.setChecked(i_settings['learn_fp'])
        self.box_learn_fr.setChecked(i_settings['learn_fr'])
        #Error correction
        self.box_undo.setValue(i_settings['undo'])
        self.box_prog_status.setValue(i_settings['prog_status'])
        self.box_restart_word.setValue(i_settings['restart_word'])
        self.box_shut_down.setValue(i_settings['shut_down'])
        self.box_word_select.setValue(i_settings['word_select_thresh'])
        #Speed & channels
        self.box_file_length.setValue(i_settings['file_length'])
        self.box_channels.setCurrentIndex(i_settings['channel_index'])
        self.box_end_delay.setValue(i_settings['end_delay'])
   
    def setClickParams(self, i_params):
        (delay, std, fr, fp_rate) = i_params
        self.box_seconds_delay.setValue(delay)
        self.box_click_dev.setValue(std)
        self.box_fp_rate.setValue(fp_rate)
        self.box_fr.setValue(fr)
    
    def setEnableLearning(self, i_checked):
        self.box_learn_delay.setChecked(i_checked)
        self.box_learn_std.setChecked(i_checked)
        self.box_learn_fp.setChecked(i_checked)
        self.box_learn_fr.setChecked(i_checked)
        
class VolumeEditWidget(QtGui.QDialog, volume_editor_layout.Ui_Dialog):
    
    ##################################### Init
    def __init__(self, i_parent=None):
        QtGui.QDialog.__init__(self, i_parent)  
        self.setupUi(self) 
        self.volumes = []
        for n in range(0, 5):
            slider = getattr(self,  "volume_settings_" + str(n))
            self.volumes.append(slider.value())
            func_vol = getattr(self,  "setVolume" + str(n))
            func_mute =  getattr(self,  "mute" + str(n)) 
            box = getattr(self,  "box_mute_" + str(n))
            QtCore.QObject.connect( slider, QtCore.SIGNAL("sliderReleased()"),  func_vol)
            QtCore.QObject.connect( box, QtCore.SIGNAL("toggled(bool)"), func_mute)
        QtCore.QObject.connect( self.box_mute_all, QtCore.SIGNAL("toggled(bool)"), self.muteAll)
        
    ########################################### Signal/slots
 
    def mute0(self, i_checked):
        self.mute(0, i_checked)

    def mute1(self, i_checked):
        self.mute(1, i_checked)
        
    def mute2(self, i_checked):
        self.mute(2, i_checked)
    
    def mute3(self, i_checked):
        self.mute(3, i_checked)
    
    def mute4(self, i_checked):
        self.mute(4, i_checked)
        
    def setVolume0(self):
        self.setVolume(0)
        
    def setVolume1(self):
        self.setVolume(1)
        
    def setVolume2(self):
        self.setVolume(2)
        
    def setVolume3(self):
        self.setVolume(3)
        
    def setVolume4(self):
        self.setVolume(4)
   
        
    ########################################## Get
    
    def getVolume(self, i_channel):
        slider_object  = getattr(self,  "volume_settings_" + str(i_channel))
        val = float(slider_object.value()) / 1000.0
        return val
    
    ########################################## Set
          
    def setVolume(self, i_channel, i_save_volume=True):
        slider_object  = getattr(self,  "volume_settings_" + str(i_channel))
        slider_val = slider_object.value()
        val = float(slider_val) / 1000.0
        if i_save_volume:
            self.volumes[i_channel] = slider_val
        self.emit(QtCore.SIGNAL("volume(float,int)"), float(val), int(i_channel))
        
    def setChannelConfig(self, i_channel_config):
        nchannels = i_channel_config.getChannels()
        channel_names = i_channel_config.getChannelNames()
        for n in range(0, nchannels):
            label_object = getattr(self,  "volume_label_" + str(n))
            label_object.setText(QtCore.QString(channel_names[n][0]))
            label_object.show()
            slider_object  = getattr(self,  "volume_settings_" + str(n))
            slider_object.show()
        for n in range(nchannels, 5):
            object_name = "volume_label_" + str(n)
            label_object = getattr(self, object_name)
            label_object.hide()
            slider_object  = getattr(self,  "volume_settings_" + str(n))
            slider_object.hide()
            
    def mute(self, i_channel, i_checked):
        slider_object  = getattr(self,  "volume_settings_" + str(i_channel))
        if i_checked:
            slider_object.setValue(0) 
        else: 
            slider_object.setValue(self.volumes[i_channel])
        self.setVolume(i_channel, i_save_volume=False)      
   
    def muteAll(self, i_checked):    
        for channel in range(0, len(self.volumes)):
            box_mute =  getattr(self,  "box_mute_" + str(channel)) 
            box_mute.setChecked(i_checked)
   
    ########################################### Signal/slots
 
    def mute0(self, i_checked):
        self.mute(0, i_checked)
 

class VolumeEditGui(QtGui.QMainWindow):
    def __init__(self):
        from channel_config import ChannelConfig
        QtGui.QWidget.__init__(self)
        channel_config = ChannelConfig(i_nchannels=5, i_sound_overlap=0.5 , i_file_length=0.4, i_root_dir="./")
        self.volume_editor = VolumeEditWidget(self)
        self.volume_editor.setChannelConfig(channel_config)
        self.volume_editor.show()
    
if __name__ ==  "__main__":
    app = QtGui.QApplication(sys.argv)
    gui = VolumeEditGui()
    gui.show()
    sys.exit( app.exec_())
# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'grid_layout.ui'
#
# Created: Mon Jun 19 21:36:01 2017
#      by: PyQt4 UI code generator 4.10.4
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName(_fromUtf8("MainWindow"))
        MainWindow.resize(1100, 782)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        MainWindow.setMinimumSize(QtCore.QSize(1100, 300))
        MainWindow.setAutoFillBackground(False)
        self.centralwidget = QtGui.QWidget(MainWindow)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.centralwidget.sizePolicy().hasHeightForWidth())
        self.centralwidget.setSizePolicy(sizePolicy)
        self.centralwidget.setMinimumSize(QtCore.QSize(1013, 0))
        self.centralwidget.setFocusPolicy(QtCore.Qt.NoFocus)
        self.centralwidget.setAutoFillBackground(False)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.gridLayout_6 = QtGui.QGridLayout(self.centralwidget)
        self.gridLayout_6.setObjectName(_fromUtf8("gridLayout_6"))
        self.label_instructions_title = QtGui.QLabel(self.centralwidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(10)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_instructions_title.sizePolicy().hasHeightForWidth())
        self.label_instructions_title.setSizePolicy(sizePolicy)
        self.label_instructions_title.setMinimumSize(QtCore.QSize(0, 20))
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Serif"))
        font.setBold(True)
        font.setWeight(75)
        self.label_instructions_title.setFont(font)
        self.label_instructions_title.setObjectName(_fromUtf8("label_instructions_title"))
        self.gridLayout_6.addWidget(self.label_instructions_title, 0, 0, 1, 1)
        self.label_instructions = QtGui.QTextEdit(self.centralwidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_instructions.sizePolicy().hasHeightForWidth())
        self.label_instructions.setSizePolicy(sizePolicy)
        self.label_instructions.setMinimumSize(QtCore.QSize(500, 70))
        self.label_instructions.setMaximumSize(QtCore.QSize(16777215, 50))
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Monospace"))
        font.setPointSize(18)
        self.label_instructions.setFont(font)
        self.label_instructions.viewport().setProperty("cursor", QtGui.QCursor(QtCore.Qt.IBeamCursor))
        self.label_instructions.setFocusPolicy(QtCore.Qt.NoFocus)
        self.label_instructions.setToolTip(_fromUtf8(""))
        self.label_instructions.setReadOnly(False)
        self.label_instructions.setObjectName(_fromUtf8("label_instructions"))
        self.gridLayout_6.addWidget(self.label_instructions, 4, 0, 1, 1)
        self.label_word_output = QtGui.QLabel(self.centralwidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(10)
        sizePolicy.setHeightForWidth(self.label_word_output.sizePolicy().hasHeightForWidth())
        self.label_word_output.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Serif"))
        font.setBold(True)
        font.setWeight(75)
        self.label_word_output.setFont(font)
        self.label_word_output.setTextFormat(QtCore.Qt.PlainText)
        self.label_word_output.setObjectName(_fromUtf8("label_word_output"))
        self.gridLayout_6.addWidget(self.label_word_output, 7, 0, 1, 1)
        self.selected_words_disp = QtGui.QTextEdit(self.centralwidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(100)
        sizePolicy.setHeightForWidth(self.selected_words_disp.sizePolicy().hasHeightForWidth())
        self.selected_words_disp.setSizePolicy(sizePolicy)
        self.selected_words_disp.setMinimumSize(QtCore.QSize(500, 80))
        self.selected_words_disp.setMaximumSize(QtCore.QSize(16777215, 80))
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Monospace"))
        font.setPointSize(12)
        self.selected_words_disp.setFont(font)
        self.selected_words_disp.setFocusPolicy(QtCore.Qt.NoFocus)
        self.selected_words_disp.setToolTip(_fromUtf8(""))
        self.selected_words_disp.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.selected_words_disp.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.selected_words_disp.setObjectName(_fromUtf8("selected_words_disp"))
        self.gridLayout_6.addWidget(self.selected_words_disp, 8, 0, 1, 1)
        self.gridLayout = QtGui.QGridLayout()
        self.gridLayout.setSizeConstraint(QtGui.QLayout.SetMinimumSize)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.clear_button = QtGui.QPushButton(self.centralwidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.clear_button.sizePolicy().hasHeightForWidth())
        self.clear_button.setSizePolicy(sizePolicy)
        self.clear_button.setMinimumSize(QtCore.QSize(60, 50))
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Serif"))
        font.setBold(True)
        font.setWeight(75)
        self.clear_button.setFont(font)
        self.clear_button.setFocusPolicy(QtCore.Qt.NoFocus)
        self.clear_button.setCheckable(False)
        self.clear_button.setObjectName(_fromUtf8("clear_button"))
        self.gridLayout.addWidget(self.clear_button, 2, 0, 1, 1)
        self.scrollbar_letter_speed = QtGui.QSlider(self.centralwidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.scrollbar_letter_speed.sizePolicy().hasHeightForWidth())
        self.scrollbar_letter_speed.setSizePolicy(sizePolicy)
        self.scrollbar_letter_speed.setMinimumSize(QtCore.QSize(200, 0))
        self.scrollbar_letter_speed.setMinimum(1)
        self.scrollbar_letter_speed.setMaximum(50)
        self.scrollbar_letter_speed.setPageStep(1)
        self.scrollbar_letter_speed.setProperty("value", 8)
        self.scrollbar_letter_speed.setOrientation(QtCore.Qt.Horizontal)
        self.scrollbar_letter_speed.setObjectName(_fromUtf8("scrollbar_letter_speed"))
        self.gridLayout.addWidget(self.scrollbar_letter_speed, 5, 0, 1, 1)
        self.gridLayout_3 = QtGui.QGridLayout()
        self.gridLayout_3.setObjectName(_fromUtf8("gridLayout_3"))
        self.label_letter_speed = QtGui.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Serif"))
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_letter_speed.setFont(font)
        self.label_letter_speed.setObjectName(_fromUtf8("label_letter_speed"))
        self.gridLayout_3.addWidget(self.label_letter_speed, 2, 0, 1, 1)
        spacerItem = QtGui.QSpacerItem(20, 40, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        self.gridLayout_3.addItem(spacerItem, 0, 0, 1, 1)
        self.gridLayout.addLayout(self.gridLayout_3, 3, 0, 1, 1)
        self.button_pause = QtGui.QPushButton(self.centralwidget)
        self.button_pause.setMinimumSize(QtCore.QSize(60, 50))
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Serif"))
        font.setBold(True)
        font.setWeight(75)
        self.button_pause.setFont(font)
        self.button_pause.setFocusPolicy(QtCore.Qt.NoFocus)
        self.button_pause.setCheckable(True)
        self.button_pause.setObjectName(_fromUtf8("button_pause"))
        self.gridLayout.addWidget(self.button_pause, 0, 0, 1, 1)
        self.gridLayout_6.addLayout(self.gridLayout, 4, 1, 5, 1)
        self.phrase_disp = QtGui.QTextEdit(self.centralwidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(50)
        sizePolicy.setHeightForWidth(self.phrase_disp.sizePolicy().hasHeightForWidth())
        self.phrase_disp.setSizePolicy(sizePolicy)
        self.phrase_disp.setMinimumSize(QtCore.QSize(500, 50))
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Monospace"))
        font.setPointSize(18)
        self.phrase_disp.setFont(font)
        self.phrase_disp.setFocusPolicy(QtCore.Qt.NoFocus)
        self.phrase_disp.setStyleSheet(_fromUtf8("border-style:ridge;\n"
"border-color: rgb(255, 92, 144);\n"
"border-width:5px;\n"
""))
        self.phrase_disp.setObjectName(_fromUtf8("phrase_disp"))
        self.gridLayout_6.addWidget(self.phrase_disp, 6, 0, 1, 1)
        self.label_phrases = QtGui.QLabel(self.centralwidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(10)
        sizePolicy.setHeightForWidth(self.label_phrases.sizePolicy().hasHeightForWidth())
        self.label_phrases.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Serif"))
        font.setBold(True)
        font.setWeight(75)
        self.label_phrases.setFont(font)
        self.label_phrases.setObjectName(_fromUtf8("label_phrases"))
        self.gridLayout_6.addWidget(self.label_phrases, 5, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1100, 35))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        self.menuFile = QtGui.QMenu(self.menubar)
        self.menuFile.setObjectName(_fromUtf8("menuFile"))
        self.menuMode = QtGui.QMenu(self.menubar)
        self.menuMode.setObjectName(_fromUtf8("menuMode"))
        self.menuDisplay = QtGui.QMenu(self.menubar)
        self.menuDisplay.setObjectName(_fromUtf8("menuDisplay"))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(MainWindow)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        MainWindow.setStatusBar(self.statusbar)
        self.edit_alphabet = QtGui.QAction(MainWindow)
        self.edit_alphabet.setObjectName(_fromUtf8("edit_alphabet"))
        self.action_click_selections = QtGui.QAction(MainWindow)
        self.action_click_selections.setCheckable(True)
        self.action_click_selections.setChecked(True)
        self.action_click_selections.setObjectName(_fromUtf8("action_click_selections"))
        self.action_click_distribution = QtGui.QAction(MainWindow)
        self.action_click_distribution.setCheckable(True)
        self.action_click_distribution.setChecked(True)
        self.action_click_distribution.setObjectName(_fromUtf8("action_click_distribution"))
        self.action_best_words = QtGui.QAction(MainWindow)
        self.action_best_words.setCheckable(True)
        self.action_best_words.setChecked(True)
        self.action_best_words.setObjectName(_fromUtf8("action_best_words"))
        self.action_open = QtGui.QAction(MainWindow)
        self.action_open.setCheckable(False)
        self.action_open.setEnabled(True)
        self.action_open.setObjectName(_fromUtf8("action_open"))
        self.action_letter_likelihoods = QtGui.QAction(MainWindow)
        self.action_letter_likelihoods.setCheckable(True)
        self.action_letter_likelihoods.setChecked(True)
        self.action_letter_likelihoods.setObjectName(_fromUtf8("action_letter_likelihoods"))
        self.action_minimum_view = QtGui.QAction(MainWindow)
        self.action_minimum_view.setCheckable(True)
        self.action_minimum_view.setChecked(False)
        self.action_minimum_view.setObjectName(_fromUtf8("action_minimum_view"))
        self.action_dictionary = QtGui.QAction(MainWindow)
        self.action_dictionary.setCheckable(False)
        self.action_dictionary.setObjectName(_fromUtf8("action_dictionary"))
        self.action_save = QtGui.QAction(MainWindow)
        self.action_save.setCheckable(False)
        self.action_save.setEnabled(True)
        self.action_save.setObjectName(_fromUtf8("action_save"))
        self.action_close = QtGui.QAction(MainWindow)
        self.action_close.setObjectName(_fromUtf8("action_close"))
        self.action_alphabet = QtGui.QAction(MainWindow)
        self.action_alphabet.setCheckable(True)
        self.action_alphabet.setChecked(True)
        self.action_alphabet.setObjectName(_fromUtf8("action_alphabet"))
        self.actionB = QtGui.QAction(MainWindow)
        self.actionB.setObjectName(_fromUtf8("actionB"))
        self.action_volume = QtGui.QAction(MainWindow)
        self.action_volume.setObjectName(_fromUtf8("action_volume"))
        self.action_settings = QtGui.QAction(MainWindow)
        self.action_settings.setCheckable(False)
        self.action_settings.setObjectName(_fromUtf8("action_settings"))
        self.action_space_bar = QtGui.QAction(MainWindow)
        self.action_space_bar.setCheckable(True)
        self.action_space_bar.setChecked(True)
        self.action_space_bar.setObjectName(_fromUtf8("action_space_bar"))
        self.action_port = QtGui.QAction(MainWindow)
        self.action_port.setCheckable(True)
        self.action_port.setChecked(True)
        self.action_port.setObjectName(_fromUtf8("action_port"))
        self.action_about_ticker = QtGui.QAction(MainWindow)
        self.action_about_ticker.setObjectName(_fromUtf8("action_about_ticker"))
        self.action_clear = QtGui.QAction(MainWindow)
        self.action_clear.setObjectName(_fromUtf8("action_clear"))
        self.action_calibrate = QtGui.QAction(MainWindow)
        self.action_calibrate.setCheckable(True)
        self.action_calibrate.setChecked(True)
        self.action_calibrate.setObjectName(_fromUtf8("action_calibrate"))
        self.action = QtGui.QAction(MainWindow)
        self.action.setCheckable(True)
        self.action.setObjectName(_fromUtf8("action"))
        self.action_tutorial = QtGui.QAction(MainWindow)
        self.action_tutorial.setCheckable(True)
        self.action_tutorial.setVisible(True)
        self.action_tutorial.setObjectName(_fromUtf8("action_tutorial"))
        self.action_inc_phrases = QtGui.QAction(MainWindow)
        self.action_inc_phrases.setCheckable(True)
        self.action_inc_phrases.setChecked(False)
        self.action_inc_phrases.setVisible(True)
        self.action_inc_phrases.setObjectName(_fromUtf8("action_inc_phrases"))
        self.action_fast_mode = QtGui.QAction(MainWindow)
        self.action_fast_mode.setCheckable(True)
        self.action_fast_mode.setChecked(True)
        self.action_fast_mode.setObjectName(_fromUtf8("action_fast_mode"))
        self.action_clear_2 = QtGui.QAction(MainWindow)
        self.action_clear_2.setObjectName(_fromUtf8("action_clear_2"))
        self.action_clear_sentence = QtGui.QAction(MainWindow)
        self.action_clear_sentence.setObjectName(_fromUtf8("action_clear_sentence"))
        self.menuFile.addAction(self.action_open)
        self.menuFile.addAction(self.action_save)
        self.menuFile.addAction(self.action_close)
        self.menuMode.addAction(self.action_tutorial)
        self.menuMode.addAction(self.action_inc_phrases)
        self.menuDisplay.addAction(self.action_clear_sentence)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuMode.menuAction())
        self.menubar.addAction(self.menuDisplay.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "Grid", None))
        self.label_instructions_title.setToolTip(_translate("MainWindow", "Visual instructions", None))
        self.label_instructions_title.setText(_translate("MainWindow", "Instructions:", None))
        self.label_instructions.setStyleSheet(_translate("MainWindow", "border-style:ridge;\n"
"border-color: rgb(92, 114, 255);\n"
"border-width:5px;\n"
"", None))
        self.label_word_output.setToolTip(_translate("MainWindow", "Word selections are displayed here", None))
        self.label_word_output.setText(_translate("MainWindow", "Output sentences:", None))
        self.selected_words_disp.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Monospace\'; font-size:12pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"> </p></body></html>", None))
        self.clear_button.setText(_translate("MainWindow", "Restart", None))
        self.label_letter_speed.setToolTip(_translate("MainWindow", "Sound overlap, if=0.0 all voice will speak simultaneously.", None))
        self.label_letter_speed.setText(_translate("MainWindow", "Speed: Scan delay (seconds))", None))
        self.button_pause.setText(_translate("MainWindow", "Play", None))
        self.label_phrases.setText(_translate("MainWindow", "Phrases:", None))
        self.menuFile.setTitle(_translate("MainWindow", "File", None))
        self.menuMode.setTitle(_translate("MainWindow", "Mode", None))
        self.menuDisplay.setTitle(_translate("MainWindow", "Display", None))
        self.edit_alphabet.setText(_translate("MainWindow", "Alphabet", None))
        self.action_click_selections.setText(_translate("MainWindow", "Selected letters/words", None))
        self.action_click_distribution.setText(_translate("MainWindow", "Click distribution", None))
        self.action_best_words.setText(_translate("MainWindow", "Most probable words", None))
        self.action_open.setText(_translate("MainWindow", "Open", None))
        self.action_letter_likelihoods.setText(_translate("MainWindow", "Letter likelhoods", None))
        self.action_minimum_view.setText(_translate("MainWindow", "Show minimum", None))
        self.action_dictionary.setText(_translate("MainWindow", "Dictionary", None))
        self.action_save.setText(_translate("MainWindow", "Save", None))
        self.action_close.setText(_translate("MainWindow", "Close", None))
        self.action_close.setShortcut(_translate("MainWindow", "Ctrl+W", None))
        self.action_alphabet.setText(_translate("MainWindow", "Alphabet", None))
        self.actionB.setText(_translate("MainWindow", "b", None))
        self.action_volume.setText(_translate("MainWindow", "Volume", None))
        self.action_settings.setText(_translate("MainWindow", "Settings", None))
        self.action_space_bar.setText(_translate("MainWindow", "Space bar", None))
        self.action_port.setText(_translate("MainWindow", "Port 20320", None))
        self.action_about_ticker.setText(_translate("MainWindow", "About Ticker", None))
        self.action_clear.setText(_translate("MainWindow", "Clear", None))
        self.action_calibrate.setText(_translate("MainWindow", "Calibrate (\"yes_\")", None))
        self.action.setText(_translate("MainWindow", "Tutorial", None))
        self.action_tutorial.setText(_translate("MainWindow", "Tutorial", None))
        self.action_inc_phrases.setText(_translate("MainWindow", "Incremet Phrases", None))
        self.action_fast_mode.setText(_translate("MainWindow", "Fast Mode", None))
        self.action_clear_2.setText(_translate("MainWindow", "Clear", None))
        self.action_clear_sentence.setText(_translate("MainWindow", "Clear", None))


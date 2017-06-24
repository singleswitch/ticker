# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'sound_gui.ui'
#
# Created: Tue Oct 26 13:38:27 2010
#      by: PyQt4 UI code generator 4.7.2
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(228, 335)
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.circle_box = QtGui.QDoubleSpinBox(self.centralwidget)
        self.circle_box.setGeometry(QtCore.QRect(150, 40, 62, 28))
        self.circle_box.setMaximum(1000.0)
        self.circle_box.setSingleStep(50.0)
        self.circle_box.setProperty("value", 400.0)
        self.circle_box.setObjectName("circle_box")
        self.period_label = QtGui.QLabel(self.centralwidget)
        self.period_label.setGeometry(QtCore.QRect(0, 40, 181, 51))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setWeight(75)
        font.setBold(True)
        self.period_label.setFont(font)
        self.period_label.setObjectName("period_label")
        self.circle_view = QtGui.QGraphicsView(self.centralwidget)
        self.circle_view.setGeometry(QtCore.QRect(10, 80, 192, 192))
        self.circle_view.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.circle_view.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.circle_view.setObjectName("circle_view")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 228, 25))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QtGui.QApplication.translate("MainWindow", "MainWindow", None, QtGui.QApplication.UnicodeUTF8))
        self.period_label.setText(QtGui.QApplication.translate("MainWindow", "Clock period (ms):", None, QtGui.QApplication.UnicodeUTF8))


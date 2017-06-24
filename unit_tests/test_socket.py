
from sys import stdin, exit, argv
import os
from PyQt4 import QtCore, QtGui, QtNetwork 

 
global port
port = 20320

class Receiver():
    def __init__(self):
        self.socket = QtNetwork.QUdpSocket()
        is_socket = self.socket.bind(QtNetwork.QHostAddress(0), port) 
        print "is_socket = ", is_socket
        if not is_socket:
            print "Binding of socket was unsuccessful"
        else:
            QtCore.QObject.connect(self.socket,  QtCore.SIGNAL("readyRead()"),  self.readPendingDatagrams)
    
    def readPendingDatagrams(self):
        print "Reading from port"
        while (self.socket.hasPendingDatagrams()):
            max_len = self.socket.pendingDatagramSize()
            (data,  host, port)  = self.socket.readDatagram (max_len) 
            print "data = ", data, " max_len = ", max_len
        #self.getClick()
        
class Sender(QtGui.QWidget):
    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self, parent)
        self.send_button = QtGui.QPushButton(self)
        self.send_button.setCheckable(True)
        self.send_button.setObjectName("sendButton")
        self.send_button.setText("Send data")
        self.socket = QtNetwork.QUdpSocket()
  
        v_layout = QtGui.QVBoxLayout() #All the buttons go in this layout
        v_layout.addWidget(self.send_button)
        self.setLayout(v_layout)
        self.connect( self.send_button, QtCore.SIGNAL("clicked(bool)"),  self.sendSlot ) 
        
    def sendSlot(self):
        self.socket.writeDatagram(QtCore.QString("y"),QtNetwork.QHostAddress(0), port) 
        
if __name__ ==  "__main__":
    app = QtGui.QApplication(argv)
    receiver = Receiver()
    sender = Sender()
    sender.show()
    exit( app.exec_())
    

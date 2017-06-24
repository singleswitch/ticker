 
from PyQt4 import QtCore, QtGui
import sys, os
from sound_gui import Ui_MainWindow
import numpy as np
 
class SoundGui(QtGui.QMainWindow, Ui_MainWindow):
    def __init__(self):
        QtGui.QMainWindow.__init__(self)  
        self.setupUi(self)
        QtCore.QObject.connect(self.circle_box, QtCore.SIGNAL("valueChanged(double)"), self.updateTimer)
        self.__graphicScene =  QtGui.QGraphicsScene(self.centralwidget)
        self.circle_view.setScene(self.__graphicScene)
        view_rect = self.circle_view.mapToScene( self.circle_view.rect() ).boundingRect()
        pen = QtGui.QPen(QtGui.QColor("black"))
        pen.setWidth(3.0)
        self.__circle = QtGui.QGraphicsEllipseItem(view_rect.x(), view_rect.y(), view_rect.width()/2.0, view_rect.height()/2.0)
        self.__circle.setPen(pen)
        
        x1  = view_rect.x() + 0.25*view_rect.width()
        y1  = view_rect.y() + 0.25*view_rect.height()
        x2  = x1 
        y2  = y1 - 0.25*view_rect.height()  
        self.__line_item = QtGui.QGraphicsLineItem(x1,y1,x2,y2)
        self.__graphicScene.addItem(self.__circle)
        self.__line_item.setPen(pen)
        
        self.__graphicScene.addItem(self.__line_item)
        
        pen = QtGui.QPen(QtGui.QColor("red"))
        pen.setWidth(10.0)
        self.__circ2 =  QtGui.QGraphicsEllipseItem(x2, y2, 10.0,10.0)
        self.__circ2.setPen(pen)
        #self.__graphicScene.addItem(self.__circ2)
        
        self.__rot_inc = 180.0
        self.__radius =  0.25*view_rect.height()  
        self.__rot_angle = 0.0
            
        
        self.__colors = [QtGui.QColor("red"), QtGui.QColor("white")]
        self.__timer = QtCore.QTimer()
        QtCore.QObject.connect(self.__timer, QtCore.SIGNAL("timeout()"), self.updateDrawing)
        self.updateTimer(self.circle_box.value())

       
    def updateTimer(self, i_value):
        self.__timer.stop()
        self.__timer.start(i_value)
    
    def updateDrawing(self):
        line = self.__line_item.line()
        x2 = line.x2()
        y2 = line.y2()
        x1 = line.x1()
        y1 = line.y1()
        
        x2 =  x1 + self.__radius*np.cos( self.__rot_angle * np.pi / 180.0)
        y2 =  y1 + self.__radius*np.sin( self.__rot_angle * np.pi / 180.0)
        #self.__circ2.setRect(x2,y2,10., 10. )
    
        self.__line_item.setLine(x1,y1,x2,y2)
        
        self.__rot_angle += self.__rot_inc
        if self.__rot_angle >= 360.0:
            self.__rot_angle = 0.0
        
  
if __name__ ==  "__main__":
    app = QtGui.QApplication(sys.argv)
    gui = SoundGui()
    gui.show()
    sys.exit( app.exec_())
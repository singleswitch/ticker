 
import numpy as np
import scipy.stats.distributions as sd
from PyQt4 import QtCore, QtGui
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.pyplot import bar
from matplotlib import rcParams

""" * This file contains all the widgets of the Ticker Gui
    *  A layout (i.e., parent) has to be provided to all widgets that specifies the font,sizes etc. """
 
class TickerWidget(QtGui.QWidget):
    def __init__(self, i_title=None, i_parent=0):
        """ * Each ticker widget consists of a widget and a title associated with it - this title is also a widget, typically a QLabel
           * The title can be none """
        QtGui.QWidget.__init__(self, i_parent)
        self.__title = i_title
        
    def show(self):
        QtGui.QWidget.show(self)
        if self.__title is not None:
            self.__title.show()
        
    def hide(self):
        QtGui.QWidget.hide(self)
        if self.__title is not None:
            self.__title.hide()
         
    def toggleDisplay(self, i_checked):
        if  i_checked:
            self.show()
        else:
            self.hide()
                
    def setTitle(self, i_str):
        self.__title.clear()
        self.__title.setText( QtCore.QString(i_str) )
    
    def getTitleText(self):
        """Return only the text associated with the title"""
        return str(self.__title.text())

    def getTitle(self):
        """Return the whole Qlable/QTextEdit containing the title text"""
        return self.__title
    
class AlphabetLabel( QtGui.QLabel):
    """This is Qlabel that can emit a signal after its repaint function was called"""
    def __init__(self, i_parent):
        QtGui.QLabel.__init__(self, i_parent)
        self.__emit = False
        
    def setEmit(self, i_is_true):
        self.__emit = i_is_true
        
    def setText(self, i_str):
        QtGui.QLabel.clear(self)
        QtGui.QLabel.setText( self, QtCore.QString(i_str) )
        
    def paintEvent(self,i_event):
        """Emit a signal when repainted"""
        self.setUpdatesEnabled(True)
        QtGui.QLabel.paintEvent(self, i_event)
        if self.__emit:
            self.emit(QtCore.SIGNAL("repainted"))
            self.__emit = False
            


class ChannelVisualisation( TickerWidget ):
    """ - This class is useful to display the parts sets of strings (e.g., the letters of alphabet) associated with different audio channels.
        - Each string is displayed as a QGraphicsTextItem within a QGraphicsRectangle.
        - Strings in the same channel are displayed next to each other.
        - Rectangles associated with strings in the same channel have the same colour.
        - The strings are displayed in a grid, with rows corresponding to strings associated with the same channel.  
        - List must have the same dimensions for display in an N x M grid,
           where N = number of channels  and M = number of strings associated with each channel. 
        - Spacers can be added with a '*" character, so that M is equal for all channels.
        Input:
            *  i_graphics_view: The graphics view to contain all the strings. The graphics view specifies the size of the grid, font, etc.
            *  i_title: An optional Qlabel/QTextEdit for the channel_strings display graphics view.
        - After the class has been instantiated, call setChannels with channel_strings to display.
        - Certain strings can be highlighted, i.e., making them appear as if they have focus.
            * Highlighting is done by setting the transparancy/alpha of the rectangle associated with a string. 
            *  A whole column can be highlighted.
            * The amount of higlighting of each item can be set with a weight between 0.0 and 1.0 """
    ###################################### Init and clear functions: 
    
    def __init__(self, i_graphics_view, i_parent_widget,  i_title=None, i_channel_strings=None ):
        TickerWidget.__init__( self, i_title, i_parent_widget)
        self.channel_colours = [ QtGui.QColor("red"), 
                                                    QtGui.QColor("green"),
                                                    QtGui.QColor("blue"), 
                                                    QtGui.QColor("Brown"),
                                                    QtGui.QColor("Purple")]
        self.space_colour = QtGui.QColor("white")
        self.__alpha_no_focus = 25
        self.__alpha_focus = 80
        self.__scene = QtGui.QGraphicsScene(i_parent_widget)
        self.__graphics_view = i_graphics_view
        self.__graphics_view.setScene(self.__scene)
        if i_channel_strings is not None:
            self.setChannels(i_channel_strings)
            
    ###################################### Main functions: 
 
    def setAlphaWeights(self,  i_weights):
        """ i_weights:
             * A list of values between 0 and 1 with the same length as the original input channel  string without spaces.
             * The weights therefore correspond to the original channel string entries excluding spaces, i.e., '*'.  
             * The new alpha value will be 200*weight, but no less than alpha_no_focus. """
        for n  in range(0, len(self.__sound_index_mapping)):
            rect_index = self.__sound_index_mapping[n]
            alpha = 220.*i_weights[n]  
            if alpha < self.__alpha_no_focus:
                alpha = self.__alpha_no_focus
            brush = self.__rects[rect_index].brush()
            color = brush.color()
            color.setAlpha(alpha)
            brush.setColor(color)
            self.__rects[rect_index].setBrush(brush)

    def clear(self):
        self.__setFocus( i_focus=False )
        self.__column_focus = None

    def setColumnFocus(self, i_sound_index):
        """ -Set the whole column corresponding to the sound index in focus 
           -Focus transparancy = alphaba_focus.
           -The input index does not include spacers (no sound), and internal counter therefore has to do the book keeping """
        if self.__column_focus is None:
            self.__setFocus( True, 0 , self.__nchannels)
            self.__column_focus =  0
        else:
            rect_index =  self.__sound_index_mapping[i_sound_index]
            col = np.int( np.floor( rect_index / self.__nchannels) )
            if not (col == self.__column_focus):
                start_index = self.__column_focus * self.__nchannels
                self.__setFocus(  False, start_index,  start_index + self.__nchannels)
                start_index = col * self.__nchannels
                self.__setFocus(  True, start_index,  start_index + self.__nchannels)
                self.__column_focus = col
        
    def setTitle(self, i_str, i_emit=False):
        TickerWidget.getTitle(self).setEmit(i_emit)
        TickerWidget.setTitle(self, i_str)

    def setChannels(self, i_channel_strings ):
        """Input:
               * A list of arrays/lists. 
               * The outer list contains a list/array of strings associated with that channel.  
               *  For example, if i_channel[0] = ['a','b',c'], 'a','b','c' will all be associated with the first channel, and will all have the same colours."""
        #Remove all existing items
        scene_items = self.__scene.items()
        for item in scene_items:
            self.__scene.removeItem(item)
        #Extract all the properties from the graphics view, to setup the letter display
        view_rect = self.__graphics_view.mapToScene( self.__graphics_view.rect() ).boundingRect()
        font = self.__graphics_view.font()
        self.__nchannels =  len( i_channel_strings)
        rect_height = float(view_rect.height())/(float(self.__nchannels))
        self.__nstrings = len(i_channel_strings[0])
        rect_width =  float(view_rect.width())/(float(self.__nstrings))
        self.__rects = []
        x_offset = 0.0
        #Also store which rectangles correspond to which sound indices (as spacers do not have sounds associated with them)
        self.__sound_index_mapping = []
        for  s in range(0,  self.__nstrings):
            y_offset = 0.0
            for n in range(0, self.__nchannels):
                channel_string =  i_channel_strings[n][s]
                #Add the text items - '*' is seen as a spacer, i.e., text =" "  
                if  channel_string == '*':
                    text_item = QtGui.QGraphicsTextItem(" ")
                    rect_colour =   self.space_colour
                else:
                    text_item = QtGui.QGraphicsTextItem(channel_string)
                    rect_colour = self.channel_colours[n]
                    self.__sound_index_mapping.append(len(self.__rects))
                text_item.setFont(font)
                self.__scene.addItem(text_item)
                #Adjust the position of the text items, so that letters in the same channels are next to each other.
                text_rect = text_item.sceneBoundingRect()
                new_x =  text_rect.x() + x_offset  
                new_y =  text_rect.y() + y_offset
                text_item.setPos(new_x, new_y)
                #Now add the rectangles according to the text item sizes and positions
                self.__rects.append( QtGui.QGraphicsRectItem(new_x, new_y, rect_width, rect_height) ) 
                self.__rects[-1].setBrush( rect_colour )
                pen = self.__rects[-1].pen()
                pen.setStyle( QtCore.Qt.PenStyle(QtCore.Qt.NoPen))
                self.__rects[-1].setPen(pen)
                self.__scene.addItem(self.__rects[-1])
                #Now readjust the text position so that that each letter is in the middle of its rectangle
                new_x +=  0.5*( rect_width - text_rect.width())
                new_y +=  0.5*( rect_height- text_rect.height())
                text_item.setPos(new_x, new_y)
                y_offset += rect_height
            x_offset +=rect_width
        self.clear( )
        
    ####################################### Private
            
    def __setFocus(self, i_focus=True, i_start_index=0,  i_end_index=-1):
        """Set items from start index to end index to have focus or no focus by changing the alpha value of the corresponding rectangles.
          If end_index = -1, end index will be set to the length of all the rectangles stored.
          The default therefore sets all items to have focus."""
        if i_end_index < 0:
            end_index = len(self.__rects)
        else:
            end_index  = i_end_index
        for n  in range( i_start_index, end_index ) :
            brush =  self.__rects[n].brush()
            colour = brush.color()
            if i_focus:
                colour.setAlpha(self.__alpha_focus)
            else:
                colour.setAlpha(self.__alpha_no_focus)
            brush.setColor(colour)
            self.__rects[n].setBrush(brush)
            
class ClickGraphScene(TickerWidget,FigureCanvas):
    """A matplotlib widget is used to display click pdf histograms"""
    def __init__(self, parent=None, i_title=None, width=7.5, height=4.5, dpi=50, xlabel="", ylabel=""):
        TickerWidget.__init__(self, i_title, parent)
        params = {'axes.labelsize': 16, 
                 'xtick.labelsize': 16,
                 'ytick.labelsize': 16 }
        rcParams.update(params)
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor='w', edgecolor=None, frameon=True)
        self.axes = self.fig.add_axes([0.095, 0.2, 0.9, 0.75], axisbg='w')  
        self.axes.set_ylabel( ylabel )
        self.axes.set_xlabel( xlabel )
        self.axes.hold(True)
        self.axes.grid('on')
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
        FigureCanvas.updateGeometry(self)
        self.figure.canvas.draw()
        self.xlabel = xlabel
        self.ylabel = ylabel
        
    def setView(self, i_mean, i_std):
        n_std = 4.0 
        (min_x, max_x) = (i_mean - n_std*i_std, i_mean + n_std*i_std)
        self.axes.set_xlim([min_x, max_x])
        n_samples = 200
        x = np.linspace( -n_std*i_std, n_std*i_std,  n_samples) + i_mean 
        y = sd.norm.pdf(x, loc=i_mean, scale=i_std)
        self.axes.plot(x, y, 'r', linewidth=2 )
        self.draw()
    
    def drawClickDistribution(self,  i_histograms ):
        self.axes.clear()
        (x, w, h) = i_histograms
        for n in range(0, len(x)):
            self.axes.bar(x[n], h[n], w[n], alpha=0.5)
        self.axes.autoscale_view(tight=False)
        self.axes.set_ylabel( self.ylabel )
        self.axes.set_xlabel( self.xlabel )
        self.draw()
  
class DictionaryDisplay( TickerWidget):
    def __init__(self, i_dict_text_edit,  i_parent_widget=0,  i_title=None ):
        TickerWidget.__init__( self, i_title, i_parent_widget)
        self.dict_display =  i_dict_text_edit
                
    def update(self, i_words, i_probs):
        """Update the text edit box specifically dedicated to display dictionary words and their probabilities"""
        self.dict_display.clear()
        max_length = 0
        for w in i_words:
            max_length = max(len(w), max_length)
        for n in range(0,len(i_probs)):
            #w = "{0:{1}}".format( words[n][0:-1], max_length ) + " :"
            w = i_words[n]
            if w == '.':
                w = '. :'
            else:
                w = i_words[n][0:-1] + " :"
            disp_str = "%s%0.3f \n" %(w, i_probs[n]) 
            self.dict_display .insertPlainText(disp_str)
        self.dict_display.moveCursor(QtGui.QTextCursor.Start)
        self.dict_display.ensureCursorVisible()
        
class  InstructionsDisplay(TickerWidget):
    """This class contains a label with a text command, telling the user what to do next"""
    def __init__(self, i_instructions_text_edit,  i_parent_widget=0,  i_title=None ):
        TickerWidget.__init__( self, i_title, i_parent_widget)
        self.instructions =  i_instructions_text_edit
        self.letter_dict = {1:"first",2:"second",3:"third",4:"fourth",5:"fifth",6:"sixth",7:"seventh",8:"eighth",9:"ninth",10:"tenth", 
                            11:"eleventh",12:"twelfth",13:"thirteenth",14:"fourteenth",15:"fifteenth", 16:"sixteenth",17:"seventeenth",18:"eighteenth",19:"nineteenth",20:"twentieth",
                            21:"twentyfirst",22:"twentysecond",23:"twentythird",24:"twentyfourth",25:"twentyfifth",26:"twentysixth",27:"twentyseventh",28:"twentyeighth",29:"twentyninth",30:"thirtieth", 
                            31:"thirtyfirst",32: "thirtysecond",33:"thirtythird",34:"thirtyfourth",35:"thirtyfifth",36:"thirtysixth",37:"thirtyseventh",38:"thirtyeighth",39:"thirtyninth",40:"fourtieth", 
                            41:"fourtyfirst",42: "fourtysecond",43:"fourtythird",44:"fourtyfourth",45:"fourtyfifth",46:"fourtysixth",47:"fourtyseventh",48:"fourtyeighth",49:"fourtyninth",50:"fiftieth"} 
    
    def clear(self):
        self.instructions.clear()
        
    def getInstructSentence(self, i_nth_letter):
        return "Select the " + self.letter_dict[ i_nth_letter] + " letter:"
    
    def instructLetterSelect(self, i_nth_letter):
        disp_str = self.getInstructSentence(i_nth_letter)
        self.update( disp_str )
        
    def update(self, i_str):
        self.instructions.setText(QtCore.QString( i_str))
        
    def highLight(self, i_offset):
        #FIXME: check this function again when refactoring demo code.
        """ Highlight the character with offset i_offset from the current character.
        i_offset: +1 means right with two character, -2 means left with 2 characters etc"""
        if i_offset == 0:
            self.instructions.moveCursor(QtGui.QTextCursor.Start, QtGui.QTextCursor.MoveAnchor)
        else:
            delta = i_offset / np.abs(i_offset)
            #At this point the cursor is at previous position
            if delta > 0:
                end_point = i_offset - 1
            else:
                end_point = np.abs(i_offset) + 1
            for n in range(0, end_point):
                if delta < 0:
                    self.instructions.moveCursor(QtGui.QTextCursor.Left, QtGui.QTextCursor.MoveAnchor)
                else:
                    self.instructions.moveCursor(QtGui.QTextCursor.Right,QtGui.QTextCursor.MoveAnchor )
        cursor = self.instructions.textCursor()
        pos = cursor.position()
        cursor.setPosition(pos, QtGui.QTextCursor.MoveAnchor ) 
        self.instructions.setTextCursor(cursor)
        self.instructions.moveCursor( QtGui.QTextCursor.Right,QtGui.QTextCursor.KeepAnchor )
        
        
class SentenceDisplay(TickerWidget):
    """Display for selected words"""
    def __init__(self, i_sentence_text_edit,  i_parent_widget=0,  i_title=None ):
        TickerWidget.__init__( self, i_title, i_parent_widget)
        self.sentences = i_sentence_text_edit
        self.sentences.setOverwriteMode(False)
    
    def clear(self):
        self.sentences.clear()

    def update(self, i_str, i_adjust_stop=True, i_add_space=True):
        if i_adjust_stop and (i_str == "."):
            self.sentences.moveCursor(QtGui.QTextCursor.Left, QtGui.QTextCursor.MoveAnchor)
        disp_str = str(i_str)
        if i_add_space:
            disp_str += " "
        self.sentences.insertPlainText(QtCore.QString(disp_str))
        self.sentences.ensureCursorVisible()
        
    def deleteLastLetter(self):
        self.sentences.textCursor().deletePreviousChar()
        
    def lastWord(self):
        last_word = str(self.sentences.toPlainText()) 
        last_word = last_word.split("_")[0:-1][-1].split(".")[-1]
        return last_word
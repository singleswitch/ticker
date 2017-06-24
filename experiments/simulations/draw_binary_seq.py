
import sys
import pylab as p
import numpy as np
sys.path.append("../../")
sys.path.append("../")
from utils import Utils, WordDict, DispSettings
from channel_config import AlphabetLoader

class DrawBinarySequences():
    def __init__(self, i_save_figures=False):
        self.disp = DispSettings()
        self.disp.setDrawBinarySequences()
        self.utils = Utils()
        self.results_dir = "./results/"
        self.save_figures = i_save_figures
    
    def compute(self):
        n_channels = 5
        root_dir = "../../config/channels" + str(n_channels) + "/"
        alphabet_loader = AlphabetLoader(root_dir)
        alphabet_loader.load(n_channels)
        alphabet = alphabet_loader.getAlphabet(i_with_spaces=False, i_group=False)  
        #Draw the binary codes for the letters in the alphabet
        self.drawLetters(alphabet)
        #Draw the binary codes for word examples
        self.drawWords(alphabet)
        p.show()
        
    def drawWords(self, i_alphabet):
        input_words = ["ace_", "act_", "and_", "awe_", "bag_", "bar_"]
        line_length=0.75
        aspect_ratio = 4.0
        x_delta = 1.5
        y_delta = 1.5
        fontsize = 8
        col_text=False
        row_text=True
        params = (line_length, aspect_ratio, x_delta, y_delta, col_text, row_text, fontsize) 
        self.drawBinaryCodes(i_alphabet,input_words, params)
        self.saveFig(self.results_dir + "word_pict_small")          
      
    def drawLetters(self, i_alphabet):
        line_length = 1.0
        aspect_ratio = 1.2
        x_delta = 1.5
        y_delta = 1.5
        fontsize = 6
        col_text=True
        row_text=True
        params = (line_length, aspect_ratio, x_delta, y_delta, col_text, row_text, fontsize) 
        self.drawBinaryCodes(i_alphabet, i_alphabet, params)
        self.saveFig(self.results_dir + "letter_pict")          
            
    def saveFig(self, i_file_name):
        if not self.save_figures:
            return
        print "***************************************************"
        print "Saving file_name = ", i_file_name
        print "***************************************************"
        self.disp.saveFig(i_file_name, i_save_eps=True)   
        
    def drawBinaryCodes(self, i_alphabet, i_input_words, i_params):
        self.disp.newFigure()
        (line_length, aspect_ratio, x_delta, y_delta, col_text, row_text, fontsize)  = i_params
        y_top = line_length
        radius = 0.5*y_top
        tol = 0.25
        x_left_letter = -2.0*x_delta -0.5
        y_top_letter = 2.0*x_delta-0.75 
        
        for (n, word) in enumerate(i_input_words):
            y_offset = -y_delta*n - y_top
            yc = y_offset+radius
            if row_text:
                if len(word) > 1:
                    p.text(x_left_letter, yc, word, fontsize=fontsize, verticalalignment='baseline', horizontalalignment='right')
                else:
                    p.text(x_left_letter, yc, word, fontsize=fontsize, verticalalignment='center',  horizontalalignment='left')
            xc = 0
            for row_letter in word:
                for (m, col_letter) in enumerate(i_alphabet):
                    if n == 0:
                        if col_text:
                            p.text(xc,  y_top_letter, col_letter, fontsize=fontsize, verticalalignment='baseline', horizontalalignment='center')
                        #p.plot([xc, xc],  [0.1, y_top_letter-0.25], 'k')
                    if row_letter == col_letter:
                        #Make a 1 (draw aline)
                        p.plot([xc,xc],[y_offset , y_offset+y_top],'k',linewidth=1.0) 
                    else:
                        #Make 0 (draw an ellipse) 
                        #radii = np.array([[radius, 0.0],[0.0, radius]])
                        #(x,y) = self.ellipseFromRadii(xc,yc,radii, nsamples=50)
                        #p.plot(x, y,'k',linewidth=1.0)
                        p.plot(xc, yc,'ko',linewidth=1.0, markersize=0.5) 
                    xc += x_delta
        p.axis('off'); p.grid('off')
        if aspect_ratio is not None:
            ax = p.gca()
            ax.set_aspect(aspect_ratio)
    
    def ellipseFromRadii(self, i_x, i_y, radii, nsamples=100):
        theta = np.array( np.linspace(0.,2.*np.pi, nsamples))
        circle = np.vstack([np.cos(theta), np.sin(theta)]).transpose()
        x = i_x + circle[:,0]*radii[0,0] + circle[:,1]*radii[0,1]
        y = i_y + circle[:,1]*radii[1,1] + circle[:,0]*radii[0,1]
        return (x,y)    
    

if __name__=="__main__":   
    app = DrawBinarySequences(i_save_figures=True)
    app.compute()

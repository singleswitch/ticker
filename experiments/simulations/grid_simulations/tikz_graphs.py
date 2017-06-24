
import numpy as np

class TikzGraph():
    def __init__(self, i_font_scaler, i_global_scale, i_filename):
        self.font_scaler = i_font_scaler
        self.global_scale = i_global_scale
        self.filename = i_filename
        
    ###################################### File/Picture operations 
    def initPicture(self):
        self.fout.write("\\begin{tabular}{c} \n")
        self.addPicture();

    def addPicture(self):
        if self.global_scale > 0.75:
            thickness = "thin"
        else:
            thickness = "very thin"
        arrow_style= ">=latex , " + thickness
        self.fout.write("\t\\begin{tikzpicture}[scale=%.2f,%s] \n" % (self.global_scale, arrow_style))      

    def endPicture(self):
        self.fout.write("\t\end{tikzpicture} \n")
        self.fout.write("\t\\\ \n")

    def initGraphFile(self):
        self.fout = open(self.filename, "w")
        self.initPicture();

    def endGraphFile(self):
        self.endPicture();
        self.fout.write("\end{tabular} \n")
        self.fout.close()
    
    def writeToFile(self, i_str):
        self.fout.write( i_str + "\n")
        
    ##################################### Include 
    
    def includePictureAsNodeText(self, picture_file,  x,  y, i_width):
        """Include a picture as the text inside a node at the given x,y coordinates """
        include_str = "{\includegraphics[width=" + str(i_width) + "cm]{" +   picture_file + "}};"
        o_str =  self.getNode( x,  y, node_description="above")  + include_str
        self.writeToFile(o_str)
        
    ##################################### Get
    
    def getNodeLabel(self, i_text, x, y, node_description=None, label=None):
        """Return text as node at x,y"""
        node_str = self.getNode( x, y,label,  node_description) +  "{" +  i_text + "};"
        self.writeToFile(node_str) 
 
    
    def getFloatLabel(self, i_display_text, val, node_description=None,x=None,y=None):
        """   *Return the label associated with a transition link for the given probability (float. 
             *This will be a node inserted into a specific path
             *node_description: square brackets [] to follow node"""
        o_str = "node"        
        if node_description is not None:
            o_str += ( "["+node_description+"]")
        if (x is not None) and (y is not None):
            o_str += (" at " + self.getXY(x,y)) 
        if not i_display_text:
            return o_str + "{}" 
        return o_str + ("{%.1f}" % val )
    
    def getLabel(self, label):
        """The label of coordinate/node to connect later on"""
        return  "(%s)" % label
    
    def getRadiusAngle(self, angle,radius):
        """Get coordinate in polar coordinates"""
        return  "(%.9f: %.9fcm)" % (angle,radius)
   
    def getXY(self, x, y):
        """Return the xy coordinate in the correct string format"""
        return  "(%.9fcm, %.9fcm)" % (x,y)

    def getArc(self, angle1, angle2, radius):
        """Return the arc parameterisation in the correct string format"""
        return  "(%.9f:%.9f:%.9fcm)" % (angle1, angle2, radius)
     
    def getNode(self, x, y, label=None,  node_description=None):
        """Return a node at x,y with/without a label in the correct format"""
        x_str = str(x) + "cm"
        y_str = str(y) + "cm"
        o_str = "\t\t\\node" 
        if node_description is not None:
            o_str += ( "["+node_description+"]")
        if label is not None:
            o_str += self.getLabel(label)
        o_str += (  " at" +  self.getXY(x,y) )
        return o_str
    
class StateGraphParams( ):
    def __init__(self, i_filename, i_scale=1.0 ):
        self.filename = i_filename
        #Typical variables to adjust
        self.global_scale =i_scale                        #Scaling factor of the whole picture
        #State variables    
        self.ns = 2.0                                     # The distance between nodes in cm
        self.rect_width = 0.4                             # The rectangle width of a state
        self.rect_height = 2.0
        #Transition variables
        self.height_offset=1.0                            # The height of the transition link top above top of state
        self.show_probs = True                            # Draw transition probs
        #Text properties
        self.font_scaler = 1.0                            # Scale all the font with this number 
        #Picture variables
        self.pict_width=1.0                               # Picture width associated with the state at the top
        self.pdf_files = None                             #List of pictures associated with the pdf of each state
        #x offset of origin in cm
        self.x_offset = 0 
 
class StateGraph(TikzGraph):
    ##################################### Init 
    """ Generate a tikz graph for a generic first-order state, where the origin is at (0,0)""" 
    def __init__(self, i_graph_params):
        self.p = i_graph_params
        TikzGraph.__init__(self, self.p.font_scaler, self.p.global_scale, self.p.filename)
        
    ##################################### Main
     
    def compute(self, i_input_string, i_states, i_transitions, i_transition_probs):
        colour_table = {'Click': 'red', 'Miss':'black','Err': 'magenta', 'Failure':'blue','Correct':'green'}    
        self.initGraphFile()
        #Do the bounding box
        o_str = "\t\t\draw[-] (-1.7cm,4.6cm) rectangle ( 18.8cm, -4.8cm);"
        self.writeToFile(o_str)
        #Show how the states are labelled in the bottom right corner
        self.getStateLabelText() 
        state_idx = {}
        for i in range(0,len(i_states)):
            state_idx[i_states[i]] = i+1
        #Iterate through the states and generate the nodes
        for i in range(0, len(i_states)):
            state_id = i_states[i]
            state_num = state_idx[state_id]
            state_text = list(state_id)
            
            if i < (len(i_states)-3):
                input_idx = int(state_text[2])+1
                state_text[2] = i_input_string[int(state_text[2])]
                if state_text[2] == "_":
                    state_text[2] = "\_"
                if state_text[0] == "_":
                    state_text[0] = "\_"
            else: 
                input_idx = "-"
            label="s"+str(state_num)
            print "i = ", i, " state_text = " , state_text
            self.drawState(state_num, state_text, label, input_idx)
            all_dest = np.array([state_idx[t] for t in i_transitions[state_id]]) 
            transition_probs = np.array([t for t in i_transition_probs[i]])
            #Draw the transtions
            for j in range(0, len(all_dest)):
                dest = all_dest[j]
                dest_id = i_transitions[i_states[i]][j]
                dest_dest = np.array([state_idx[t] for t in i_transitions[dest_id]])
                forwards = True
                line_thickness = "thin"
                if (state_id[0] == "D") and (j < 1):
                    """Links far apart associated with Delete key"""
                    forwards = False
                if (dest_id == "Correct") or (dest_id == "Failure") or (dest_id == "Err"):
                    colour = colour_table[dest_id]
                    if (dest_id == "Failure") or (dest_id == "Err"):
                        forwards = False
                    else:
                        line_thickness = "thick"
                elif j==0:
                    colour = colour_table["Click"]
                elif j==1:
                    colour = colour_table["Miss"]
                    if (not state_id[0] == "D") and (state_num > dest):
                        forwards = False
                else:
                    raise ValueError("No colour found for j=" + str(j))
                
                if state_num > dest:
                    forwards = False
                else:
                    forwards = True
                self.drawTransition( state_num, dest, dest_dest, transition_probs[j], colour, forwards, line_thickness)
        self.endGraphFile()
        
    ##################################### Get
    def getStateCenterCoordinate(self, state ):
        """Get xy coordate of a the centroid associated with a specific state - state in [1,...,N], 
           where N is the number of states.
           if i_row_state is True the counter will increment in y direction downwards otherwise 
           in vertical direction."""
        return ((state-1)*self.p.ns + self.p.x_offset, 0)
    
    def getStateRectCorners(self, state):
        (x, y) = self.getStateCenterCoordinate(state)
        top_x = x - 0.5*self.p.rect_width  
        top_y = y + 0.5*self.p.rect_height
        bottom_x = x + 0.5*self.p.rect_width  
        bottom_y = y - 0.5*self.p.rect_height
        return (top_x, top_y, bottom_x, bottom_y)
    
    def getStateText(self, i_state_num,  i_state_label, input_idx ):
        """The text inside the state circle"""
        str_label = ''.join(i_state_label)
        if (str_label == "Correct") or (str_label == "Failure") or (str_label == "Err"):
            if str_label == "Failure":
                state_text = "Failure (" + str(i_state_num) + ")" 
            elif str_label == "Correct":
                state_text = "Correct (" + str(i_state_num) + ")" 
            else:
                state_text = "Error (" + str(i_state_num) + ")" 
            return "{\\begin{sideways}" + state_text + " \\end{sideways}};"
             
        i_letter = "%s" % i_state_label[2]
        i_click  = "$%s$" % i_state_label[4]    
        if i_state_label[0]  == "D":
            o_letter = "$\\leftarrow$"
        else:
            o_letter = i_state_label[0]
        
        o_letter = "%s" % o_letter 
        if i_state_label[1] == "*":
            i_undo = ""
            scan = "R"
        else:
            i_undo = "$%s$" % i_state_label[6]    
            scan = "C"  
        state_text = "{\\begin{tabular}{c}"
        state_text += ( "{\\scriptsize  "+ str(i_state_num) + " }" + "\\\\")
        state_text += (scan + "\\\\")
        state_text += (str(input_idx) + "\\\\")
        state_text += (o_letter + "\\\\")
        state_text += (i_letter + "\\\\")
        state_text += (i_click + "\\\\")
        state_text += (i_undo + "\\\\")
        state_text += "\end{tabular}};" 
        return  state_text
    
    def getStateLabelText(self):
        """Return the text describing how each cell is labelled"""
        state_text = "{\\begin{tabular}{c}"
        state_text += ("$n$ \\\\")
        state_text += ("R/C \\\\")
        state_text += ("$m$ \\\\")
        state_text += ("$\ell_{v'}$ \\\\")
        state_text += ("$w_{\mathrm{x}}^{m}$ \\\\")
        state_text += ("$e$ \\\\")
        state_text += ("$u$ \\\\")
        state_text += "\end{tabular}};" 
        (x,y) = self.getStateCenterCoordinate(1)    
        o_str =  self.getNode(x-self.p.x_offset, y, "s100" ) +  state_text
        self.writeToFile(o_str)
         
    ##################################### Main draw

    def drawState(self, state_num, state_id, label, input_idx):
        """ * Draw an State state - it is a node with a circle around it. The text inside the 
           * circle indicates the state number (top) and pdf number (bottom), both from 1..N (states) 
               and 1...K (pdfs)"""
        (x, y) = self.getStateCenterCoordinate(state_num)    
        o_str =  self.getNode(x, y, label ) +  self.getStateText(  state_num, state_id, input_idx )
        self.writeToFile(o_str) 
        (top_x, top_y, bottom_x, bottom_y) = self.getStateRectCorners(state_num) 
        rect_label = "rect" + list(label)[-1]
        o_str = "\t\t\draw (%.4fcm,%.4fcm) rectangle(%.4fcm,%.4fcm);"    % (top_x,top_y,bottom_x,bottom_y)
        self.writeToFile(o_str) 
        
    def drawTransition(self, state, dest, dest_dest, prob, i_colour="black", forwards=True, i_line_thickness="thin" ):
        """ * Neighbouring transitions are drawn treated a bit special because they occur so frequently 
             and have to be drawn neatly. 
          * If the links between two neighbouring states are bidirectional, i.e., state can go to dest 
            and dest can go to 
             state with nonzero probability, the links are drawn the same as other transition links, but the lines will be straight (not bended). 
          * If the links between two neighbouring states are unidirectional, the transition link will be a straight line originating at zero angle.""" 
        properties =  i_line_thickness + " , " + i_colour
        if (dest == state):
            (x_src, y_src) = self.getStateCenterCoordinate(state)
            (top_x, top_y, bottom_x, bottom_y) = self.getStateRectCorners(state) 
            xy_src = self.getXY(x_src,  bottom_y)
            above = True 
            o_str = "\t\t\path[-]" +  xy_src + " edge[" 
            in_angle = 270 + 30.0
            out_angle = 270 - 30.0
            dist =  self.p.height_offset + np.cos(np.pi*in_angle / (180.0))
            loop = "out=%.4f, in=%.4f, distance=%.4fcm" % (in_angle, out_angle, dist) 
           
            properties =  i_line_thickness + "," + i_colour + "," + loop + ", ->]"
            if above:
                node_desciption = "above "
            else: 
                node_description = "below "
            node =  self.getFloatLabel( self.p.show_probs, prob, node_desciption)
            o_str += ( properties  + node + xy_src + ";")
            self.writeToFile(o_str)
        else:
            self.__drawNonSelfloopsTransitions( state, dest, prob, i_colour, forwards,  i_line_thickness )
       
        ##Bidirectional links
        #self.__drawNonSelfloopsTransitions( state, dest, transition_probs[state-1, dest-1], self.p.dangle_skiplinks )
   
    #################################### Private draw
    
    def __drawNonSelfloopsTransitions(self,  state,dest, prob, i_colour ,i_forwards , i_line_thickness ):
        """ Draw a transition from the node associated with "state" to the node associate with "dest. """
        #Compute arrow starting and end points (angle from x axis in state)
        #All skiplinks start from the same point and all neighbouring links start from the same point.
        (x_src, y_src) = self.getStateCenterCoordinate(state)
        (x_dest, y_dest) = self.getStateCenterCoordinate(dest)
        (top_x, top_y, bottom_x, bottom_y) = self.getStateRectCorners(state) 
        x_middle = 0.5*(x_dest - x_src) + x_src 
        properties =  i_line_thickness + " , " + i_colour + ", overlay"
        if not i_forwards:
            xy_middle = self.getXY(x_middle, bottom_y -  self.p.height_offset)
            xy_src = self.getXY(x_src, bottom_y)
            xy_dest = self.getXY(x_dest, bottom_y)
            above = False
            angle_str = "to[out=200,in=270]" 
        else:
            xy_middle = self.getXY(x_middle, top_y + self.p.height_offset)
            xy_src = self.getXY(x_src, top_y)
            xy_dest = self.getXY(x_dest, top_y)
            above = True
            angle_str = "to[out=30,in=60]"
        #o_str = "\t\t\path[-]" +  xy_src + " edge[" 
        #if above:
        #    o_str += "above, "
        #else: 
        #    o_str += "below, "
        #o_str += ( properties + "] "  + self.getFloatLabel( self.p.show_probs, prob) + xy_middle  + ";")
        #self.writeToFile(o_str)
        #o_str =  "\t\t\draw[edge, ->, " + properties + "] " + xy_middle + "--" + xy_dest + ";"
        o_str =  "\t\t\draw[edge, ->, " + properties + "] " + xy_src + angle_str + xy_dest  + ";"
        self.writeToFile(o_str)
       
        
 
    def __drawSelfloop(self, xc, yc, prob):
        (selfloop_x, selfloop_y) = self.p.getSelfLoopXY()
        (x,y) = (selfloop_x+xc, selfloop_y+yc)
        angle1 = -45.0
        angle2  = -45.0 + 270.0
        o_str = "\t\t\draw[edge,<-]" + self.getXY(x[0],y[0]) + " arc " +  self.getArc(angle1,angle2, self.p.selfloop_rad) + ";"
        self.writeToFile(o_str) 
        label_x = xc
        label_y = y[0] + np.sqrt(self.p.selfloop_rad**2 - (x[0] - label_x)**2) + self.p.selfloop_rad
        if self.p.show_probs:
            o_str = "\t\t\\" + self.getFloatLabel(  self.p.show_probs, prob, node_description="above",x=label_x , y=label_y ) + ";"
            self.writeToFile(o_str) 
 
  
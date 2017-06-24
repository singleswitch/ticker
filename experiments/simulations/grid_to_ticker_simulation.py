
import sys, time
sys.path.append("../../")
sys.path.append("../")
import numpy as np
import pylab as p
import scipy.stats.distributions as s 
#Display
from tikz_graphs import StateGraphParams, StateGraph
from utils import Utils, PhraseUtils
from grid_simulation import GridSimulation
"""
* Nog uitstaande vir Grid2:
* Make seker Gaussian offset is reg en alle moontlike opsies som na een. (begin en einde delay)
* Stel plot op i.t.v. duration i.pv om tyd te fix
* Verstel Gaussian noise en maak plots (mean en std)
* Wys ook geen uittree simbool

* Doen dieselfde vir Ticker: Run dp randomsier vir 1 and 2 channels
* Onthou die enigste uittree codes is eintlik 1000001000 ens. nie alle permutasies van 1'e nie
  (so eintlik makliker as die grid)
* Ook p(x) is uniform wat die information rate drasties makliker maak
* Doen wpm eksperiment: vir 'n gegewe stel letter probs om bitrate te bepaal
"""

class GridToTickerSimulation(GridSimulation):
  
   ########################## Probability computations
    #Function that computate the probability distributions needed to copmute expectations
    def stateProbs(self, i_states, i_transitions, i_transition_probs,  i_scan_delays, i_T, i_grnd_truth_word, i_max_spurious, i_display):
        """
        *o_probs in format:
            * rows: times steps
            * cols: 
                0 - word error state 
                1 - system failure state
                2 - correct state
        * o_err_probs: Probs of states associated with column and "_", but not correct, i.e., an erroneous word selection
        """ 
        #The number of correct letters in a word and total number of states
        (M, N) = (len(i_grnd_truth_word),len(i_states))
        #Sate ids
        (state_nums, dest)  = self.convertDest( i_states, i_transitions )   
        #Forwards probs, the prob to be in any state at any given time
        state_probs = -np.inf*np.ones(N)
        state_probs[0] = 0.0
        #Probs for with the number of scans to write any word 
        max_scans = np.max(i_scan_delays)
        scan_probs = -np.inf*np.ones( [N, max_scans*i_T+1])
        scan_probs[0,0] = 0.0
      
        #Probs for number of clicks to write any word
        click_probs = None
        
        #Probs for number of erroneous characters
        err_probs = -np.inf*np.ones(M+i_max_spurious+1)
        
        for cur_time in range(0, i_T):
            state_probs_prev = np.array(state_probs)
            state_probs[0:N-3] = -np.inf*np.ones(N-3)
            scan_probs_prev = np.array(scan_probs)
            scan_probs[0:N-3,:] = -np.inf
  
            #Do the non-terminating states first
            for i in range(0, N-3):
                #The state id
                (letter_t, col, n_error, m, n_undo) = self.itemizeStateId(i_states[i])
             
                #Only click destinations, the first entry is always associated with a miss
                for j in range(0, len(i_transition_probs[i])):
                    p_click = np.log(i_transition_probs[i][j])
                    psum_click = self.utils.elnprod(state_probs_prev[i], p_click)
                    dest_click = dest[i][j]
                    #State probs
                    state_probs[dest_click] = self.utils.elnsum(state_probs[dest_click], psum_click)
                    #Scan probabilities
                    prob_click = scan_probs_prev[i, 0:-i_scan_delays[i]] + p_click
                    tmp_probs = np.vstack( (scan_probs[dest_click,i_scan_delays[i]:], prob_click)).transpose()
                    scan_probs[dest_click,i_scan_delays[i]:] = self.utils.expTrick(tmp_probs).flatten()
 
                    #At all time steps, except for T, a terminating state can only be reached by clicking 
                    #and being busy with a column scan
                    if not(dest_click >= N-3) :
                        continue 
    
                    #scan_probs2[cur_time+1] = self.utils.elnsum(scan_probs2[cur_time+1], psum_click) 
                    """Char error probs: See if the current letter occurs in remaining part of word
                    If the system failed the whole word is taken to be wrong
                    if i_grnd_truth_word.find(letter_t, m, M) >= 0:
                    """
                    #In case of a click and the destination is not the correct state , the 
                    #Remaining word from the spurious event is taken to be wrong.
                    if not (dest[i][j] == N-1):
                        m += 1
                    else:
                        n_error += 1  
                    err_probs[M-m+n_error] = self.utils.elnsum(err_probs[M-m+n_error], psum_click)
            
            sum_probs = np.sum( np.exp( scan_probs ), axis= 0) 
            idx_max = np.argmax(sum_probs  ) 
            
            if self.isEarlyBailTestStateProbs(state_probs, cur_time+1, i_T, i_display):
                break
     
        (state_probs, scan_probs, click_probs, err_probs) = self.updateMaxTimeFailureProbs(state_probs, scan_probs, 
            click_probs, err_probs, M, N, i_T, cur_time)
        scan_probs = self.utils.expTrick(scan_probs[-3:,:].transpose()) 
        (scan_probs, err_probs) = ( np.exp(scan_probs),  np.exp(err_probs)) 
        wpm_probs = np.array(scan_probs) 
        print "state_probs = ", state_probs[-3:]
        return (scan_probs, click_probs, err_probs, wpm_probs)
   

    ##################################### State Topology

    def getStates(self,i_word, i_params ):
        """States in format: 
        [output letter, row (*) / col (_), input letter number, #spurious click, #undo]"""
        (n_errors, n_undo, self.fr, self.fp_rate, scan_delay, click_time_delay, self.std, n_max, draw_tikz, display) = i_params
        self.displayParams(i_word, i_params)
        n_input_letters = len(i_word)  
        (row_states, col_states, row_transitions, col_transitions, row_probs, col_probs) = ([],[],{},{},[],[])
        (states, transitions,transition_probs, state_ids) = ([], {},[], [])
        scan_delays = [] #Add the number of scan delays associated with each state
        for letter_idx in range(0, n_input_letters):
            letter = i_word[letter_idx]
            if letter_idx >=  (n_input_letters-1):
                correct_id = "Correct"
            else:
                correct_id = self.getStateId(self.config[0][0], letter_idx+1, i_click=0, i_undo=0, i_row=True)
            for click in range(0, n_errors):
                if display :
                    print "******************************"
                    print "Letter = ", letter, " click ", click   
                    print "---------------------------------------"
                (tmp_row_states, tmp_row_transitions, tmp_row_probs, tmp_scan_delays) = self.getRowStates( letter, letter_idx, click, self.letter_info['0'] , display )
                row_states.extend(tmp_row_states)
                scan_delays.extend(tmp_scan_delays)
                states.extend(tmp_row_states)
                row_probs.extend(tmp_row_probs)
                transition_probs.extend(tmp_row_probs)
                for key in tmp_row_transitions.keys():
                    row_transitions[key] = tmp_row_transitions[key]
                    transitions[key] = row_transitions[key]
                
                for undo in range(0, n_undo):
                    (tmp_col_states, tmp_col_transitions, tmp_col_probs, tmp_scan_delays) = self.getColStates( letter, letter_idx, click, 
                        undo,  n_undo,  n_errors, correct_id, self.letter_info['1'], display ) 
                    col_states.extend(tmp_col_states)
                    scan_delays.extend(tmp_scan_delays)
                    states.extend(tmp_col_states)
                    state_ids.append([str(letter_idx),click,undo]) 
                    col_probs.extend(tmp_col_probs)
                    transition_probs.extend(tmp_col_probs)
                    for key in tmp_col_transitions.keys():
                        col_transitions[key] = tmp_col_transitions[key] 
                        transitions[key] = col_transitions[key]
        if display:
           self.displayStates( row_states, col_states, row_transitions, col_transitions, n_undo, n_errors, n_input_letters, scan_delays)
        #Update word selection (error state), the system failure state and the correct state
        states.extend(['Err', 'Failure', 'Correct'])
        scan_delays.extend([0,0,0])
        transition_probs.extend([np.array([1.0]),np.array([1.0]), np.array([1.0])]) 
        transitions['Correct'] = ['Correct']
        transitions['Failure'] = ['Failure']
        transitions['Err'] = ['Err']
         
        if draw_tikz:
            self.tikzStateDiagram( i_word, states, transitions, transition_probs)
        return (states, transitions, transition_probs, scan_delays)
    
    def getRowStates(self, i_input_letter, i_input_letter_num, i_click, i_letter_info, i_display):
        states = []
        transition_probs = []
        scan_delays = []
        transitions = {}
        row_ids = self.getGridRow()
        row_ids.append(row_ids[0])
        if i_click > 0:
            """The user will try to undo an erroneous click"""
            gauss_mean = self.getInputClickInfoRow('D',   i_letter_info)
        else:
            gauss_mean = self.getInputClickInfoRow( i_input_letter,   i_letter_info)
        #Compute the row-scan destinations and state ids
        for r in range(0 , len(row_ids)-1):
            states.append(self.getStateId(row_ids[r], i_input_letter_num, i_click, None, i_row=True) )
            #The scan delays - count only group scans
            if r == (len(row_ids)-2):
                scan_delays.append( self.last_scan_time )
            elif r == 0:
                scan_delays.append( 1.0 + 1.0*float(self.add_tick) )
            else:
                scan_delays.append( 1.0 )
            #The transition probs
            transition_probs.append([])
            if r == (len(row_ids)-2):
                dest_list = list(self.config[r])
                """Proceed to the next letter if this one is missed, repeat scan at last letter"""
                dest_miss = self.getStateId(row_ids[r+1], i_input_letter_num, i_click, None,i_row=True) 
                transitions[states[-1]] = [dest_miss]
                transition_probs[-1] =  np.zeros( len(row_ids) )
                
                for rr in range(0 , len(row_ids)-1):
                    dest_click = self.getStateId(self.config[rr][0], i_input_letter_num, i_click, i_undo=0, i_row=False) 
                    transitions[states[-1]].append(dest_click) 
                    (prob_click, tmp_prob_miss ) = self.clickProb( row_ids[0:-1],  i_letter_info,  gauss_mean, rr ,  i_display )
                    transition_probs[-1][rr+1] = prob_click
                transition_probs[-1][0] = 1.0 - np.sum(transition_probs[-1][1:])
            else:
                dest_miss = self.getStateId(row_ids[r+1], i_input_letter_num, i_click, None,i_row=True) 
                transitions[states[-1]]= [ dest_miss ] 
                transition_probs[-1] = np.array([ 1.0 ] )
        return (states, transitions, transition_probs, np.array(scan_delays))
    
    def getColStates(self, i_input_letter, i_input_letter_num, i_click, i_undo, i_n_undo, i_n_errors, i_correct_id, i_letter_info,  i_display ):
        #Compute the current row ids
        states = []
        scan_delays = []
        transition_probs = []
        transitions = {}
        row_ids = self.getGridRow()
        for r in range(0 , len(row_ids)):
            gauss_mean= self.getInputClickInfoCol(i_input_letter, i_letter_info,  i_click,  r )
            dest_list = list(self.config[r])
         
            for c in range(0, len(dest_list)):
                
                states.append(self.getStateId( dest_list[c], i_input_letter_num, i_click, i_undo, i_row=False))
                
                #The scan delays
                if c == (len(dest_list)-1):
                    scan_delays.append( self.last_scan_time )
                elif c == 0:
                    scan_delays.append( 1.0 + 1.0*float(self.add_tick) )
                else:
                    scan_delays.append( 1.0 )
               
                transition_probs.append([])                    
                if c < ( len(dest_list) -1 ):
                    dest_miss = self.getStateId(dest_list[c+1], i_input_letter_num, i_click, i_undo, i_row=False)
                    transitions[states[-1]] = [dest_miss] 
                    transition_probs[-1] = np.array([ 1.0 ] )
                    continue
                
                if ( (i_undo + 1)  >= i_n_undo) and (i_n_undo > 0) :
                    """We've reached an undo after false positive click, go back to row scan"""
                    dest_miss = self.getStateId( row_ids[0], i_input_letter_num, i_click , i_undo=0, i_row=True)
                else:
                    """Increment the undo counter"""
                    dest_miss = self.getStateId( dest_list[0], i_input_letter_num, i_click , i_undo + 1, i_row=False)
           
                transitions[states[-1]] = [dest_miss]
                transition_probs[-1] = np.zeros( len(dest_list) + 1 )
           
                for cc in range(0, len(dest_list)):
                    #TODO fix this in other version
                    #The destinations if a click happens
                    if (dest_list[cc] == i_input_letter) and (i_click == 0):
                        """If the user clicks on the desired letter and nothing else has been written,
                           i.e., all errors undone, the correct state is reached""" 
                        dest_click = i_correct_id
                    elif dest_list[cc] == "D":
                        """Click on backspace/delete"""
                        dest_click = self.getStateId(  row_ids[0], i_input_letter_num-1, i_click-1, i_undo=0, i_row=True)
                    elif (i_click+1) >= i_n_errors:
                        """If too many letters have been selected the system fails"""
                        dest_click = 'Failure'
                    elif (dest_list[cc] == "_") or (dest_list[cc] == "."):
                        dest_click = 'Err'
                    else:
                        """If a false positive click happens the undo counter is set to zero and the current 
                           click is incremented"""
                        dest_click = self.getStateId( row_ids[0], i_input_letter_num, i_click+1, i_undo=0, i_row=True)
                        
                    transitions[states[-1]].append(dest_click) 
                    (prob_click, tmp_prob_miss ) = self.clickProb( dest_list ,  i_letter_info[r],  gauss_mean, cc ,  i_display )
                    transition_probs[-1][cc+1] = prob_click
                
                transition_probs[-1][0] = 1.0 - np.sum(transition_probs[-1][1:])
        return (states, transitions, transition_probs, np.array(scan_delays))
 
    
    ##################################### Code Probs
    
    def clickProb(self, seq, time_info, gauss_mean, cnt, i_display ):
        """
        * gauss_mean: The mean of the Gaussian the user is supposed to click on
        * seq: All the cells that have to be scanned, currently we're at cell cnt."""
        #The cell's might have different lengths
        delta_time =  np.float32( time_info['t_end'][-1] )
        fp = self.getFalsePositiveProb(self.fp_rate, delta_time)
        
        if gauss_mean is None:
            #The user is has clicked on the wrong row and now waits to cancel it
            click_output_1 = 1.0 - fp
            click_output_0 = fp 
            if i_display:
                print "%s obs_1 =%.4f, obs_0=%.4f  " % (seq[cnt], click_output_1, click_output_0) 
        else:
            """The user wants to click on either undo or the target letter"""
            (letter, t_start, t_end, current_mean) = (seq[cnt], time_info['t_start'][cnt], time_info['t_end'][cnt], time_info['gauss_mean'][cnt])
            click_pdf_end = s.norm.cdf(x=(t_end + self.click_time_delay), loc=gauss_mean, scale=self.std)  
            click_pdf_start = s.norm.cdf(x=(t_start + self.click_time_delay), loc=gauss_mean, scale=self.std)
            q = click_pdf_end - click_pdf_start
            click_output_0 = fp * ( 1.0 -  ((1.0-self.fr)*q) )
            click_output_1 = 1.0 - click_output_0
            if i_display:
                print "%s g_mean=%.4f,  t_mean=%.4f, q=%.6f  " % (letter,  gauss_mean, current_mean, q),
                print " obs_1 =%.4f, obs_0=%.4f " % ( click_output_1, click_output_0 ),
                print " t_start = %.4f, t_end =  %.4f " % ( t_start + self.click_time_delay, t_end+ self.click_time_delay )
        return (click_output_1,  click_output_0 )
 
 
    def getSeqDuration(self, i_seq, i_display):
        durations = {}
        durations['t_start'] = np.cumsum( np.array(self.scan_delay*(np.ones(len(i_seq)))) ) 
        if not self.add_tick:
           durations['t_start'] -= self.scan_delay
        durations['t_end'] = durations['t_start'] + self.scan_delay
        durations['t_end'][-1] += self.group_delta
        durations['gauss_mean'] =  durations['t_start'] + self.click_time_delay + 0.5*self.scan_delay
        durations['t_start'][0] = 0.0 
        total_time = durations['t_end'][-1]
       
        if i_display:
      
            print "Input seq=%s " % (''.join(i_seq)),
            print "t_start=" , self.utils.stringVector( durations['t_start'] + self.click_time_delay, i_type="%.1f" ),
            print "t_end=" , self.utils.stringVector( durations['t_end'] + self.click_time_delay, i_type="%.1f" ),
            print "g_mean=", self.utils.stringVector( durations['gauss_mean'], i_type="%.1f" ),
            print "total_time = %.3f " % (total_time),
            print " add_tick = " , self.add_tick, " scan delay = " ,self.scan_delay, " click_time delay = ", self.click_time_delay
         
        return durations    
    
    
    ############################################# Display
    
 
    def tikzStateDiagram(self, i_input_letters, i_states, i_transitions, i_transition_probs):
        params = StateGraphParams(  i_filename=(self.output_dir + "grid2.tikz"), i_scale=0.8)
        params.ns = 0.4
        params.rect_height = 3.0
        params.show_probs = False 
        params.x_offset = 0.8                                  
        graph = StateGraph(params)
        graph.compute( i_input_letters, i_states,  i_transitions,  i_transition_probs, flip_click_labels=True)
 
 
    ####################################### Get
 
    def getMinScans(self, i_word, i_last_scan_time=None):
        """Get the minimum time (measured as #scans) it should take to write a sentence using Grid 2"""
        if i_last_scan_time is None:
            last_scan_time = 1
        else:
            last_scan_time = self.last_scan_time
        t = 0
        for letter in i_word:
            (row, col) =  self.letter_pos[letter]
            
            self.letter_pos[self.config[row][col]] = (row,col)
            total = len(self.config[row])-1 + len(self.config) - 1  + 2*last_scan_time 
            if self.add_tick: 
                total += 2
            t += total 
            #print "sentence = ", i_sentence, " letter = ", letter, " row = ", row, " col = ", col, " total = ", total, " t = " , t
        return t
    
    #Diagnositc
    def displaySpeed(self, i_scan_probs, i_input_word):
        scans = np.arange(0, len(i_scan_probs))
        avg_scans = np.sum(scans*i_scan_probs)
        std_scans = np.sqrt(np.sum(((scans-avg_scans)**2)*i_scan_probs)) 
        idx_max = np.argmax(i_scan_probs)
      
        print "***************************************************************"
        print "Speed stats"
        print "***************************************************************"
        print "scan_probs sum = ", np.sum( i_scan_probs ) , " avg_scans = ", avg_scans, " std_scans = ", std_scans             
        print "Min scans = ", self.getMinScans(i_input_word, self.last_scan_time) 
        print "Best probs = ", idx_max, " prob = ", i_scan_probs[idx_max]
        
        
        
if __name__ ==  "__main__":
    app = GridToTickerSimulation()
    w = "abcdefghijklmnopqrstuvwxyz_."
    m = app.getMinScans(w)
    print "minimum scans = " , m, " avg scans per letter = ", m / float(len(w))
    

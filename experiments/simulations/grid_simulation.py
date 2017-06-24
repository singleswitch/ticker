
import sys, time
sys.path.append("../../")
sys.path.append("../")
import numpy as np
import pylab as p
import scipy.stats.distributions as s 
#Display
from tikz_graphs import StateGraphParams, StateGraph
from utils import Utils, PhraseUtils
 
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

class GridSimulation():
    ########################################### Init
    
    def __init__(self, i_display=False):
        #Assume Gaussians in middle of file
        self.display = i_display 
        self.output_dir = "/home/emlimari/ticker_dev/pami_2017_scanning/figures/" #./results/simulations/"  "./results/simulations/"
        self.word_length = 5.0
        self.debug = False
        self.utils = Utils()
        self.phrase_utils = PhraseUtils()
        self.initDefaultConfig()
        self.getLetterPos(self.display)
        self.early_bail_prob = 0.01
        #Add an extra sound at the beginning to help pre-empting of the first character
        #The first sound will be double the length 
        self.add_tick = True
 
    def setGroupDelta(self, i_group_delta):
        self.group_delta = i_group_delta
        self.last_scan_time = int(np.ceil( self.scan_delay + self.group_delta / self.scan_delay))
        
    def init(self, i_scan_delay, i_click_time_delay, i_std, i_config ):
        self.scan_delay= i_scan_delay
        self.click_time_delay = i_click_time_delay
        self.std = i_std
        self.setGroupDelta(self.click_time_delay + 2.0* self.std)
     
        if i_config is not None:
            self.config = list(i_config)
        else:
            self.initDefaultConfig()
        self.getLetterPos(self.display)
        self.letter_info = self.getLetterDurations(self.display)
        self.n_rows = len(self.config)
        self.n_cols = 0
        for cols in self.config:
            self.n_cols = max(self.n_cols, len(cols))
            
    def initDefaultConfig(self):
        self.config = [['a','b','c','d','_','D'],
                           ['e','f','g','h','.'],
                           ['i','j','k','l','m','n'],
                           ['o','p','q','r','s','t'],
                           ['u','v','w','x','y','z']]
    
    ##################################### Main
   
    def compute(self, i_sentence, i_params):
        words = self.phrase_utils.wordsFromSentece(i_sentence) 
        (n_errors, n_undo, fr, fp_rate, scan_delay, click_time_delay, std, n_max, draw_tikz, display) = i_params
        self.init(scan_delay, click_time_delay, i_std=std, i_config=None)
        print "****************************************************************************************************************************"
        print "Words to test = ", words
        M = len(words)
        total = np.zeros(10)
        for m in range(0, M):
            print "****************************************************************************************************************************"
            grnd_truth_word = self.phrase_utils.getWord(words[m]) 
            (results, scan_probs, click_probs, err_probs, wpm_probs)  = self.wordResults(grnd_truth_word, i_params)   
            self.displayResults(results, True)
            total += np.array(list(results))
        total /= M
        print "****************************************************************************************************************************"
        print "Final results Grid"
        print "****************************************************************************************************************************"
        self.displayResults(tuple(total), True, True)
        print "****************************************************************************************************************************"
        return total
    
    def probResults(self, i_grnd_truth_word, i_params):
        #results: (min_scans, avg_scans, std_scans, min_wpm avg_wpm, std_wpm, avg_chr_err, std_chr_err)
        (n_errors, n_undo, fr, fp_rate, scan_delay, click_time_delay, std, n_max, draw_tikz, display) = i_params
        max_scans = self.getMaxScans(i_grnd_truth_word, n_max)
        #Get the Markov chain structure
        (states, transitions, transition_probs, scan_delays ) = self.getStates(i_grnd_truth_word, i_params )
        #Compute the pdfs for the stats
        (scan_probs, click_probs, err_probs, wpm_probs) = self.stateProbs(states, transitions, transition_probs, scan_delays, 
            max_scans,  i_grnd_truth_word, n_errors, display)
        return (scan_probs, click_probs, err_probs, wpm_probs)
    
    def wordResults(self, i_grnd_truth_word, i_params):
        (scan_probs, click_probs, err_probs, wpm_probs) = self.probResults(i_grnd_truth_word, i_params)
        #Compute the stats using the probs and possible outputs that correspond to the probs 
        results = self.probsToResults( i_grnd_truth_word, scan_probs, click_probs, err_probs, wpm_probs)
        (n_errors, n_undo, fr, fp_rate, scan_delay, click_time_delay, std, n_max, draw_tikz, display) = i_params
        self.displayResults( results, display)
        if click_probs is None:
            click_probs = 0.0 
        return (results, scan_probs, click_probs, err_probs, wpm_probs)
 
    def probsToResults(self, i_grnd_truth_word, i_scan_probs, i_click_probs, i_err_probs, i_wpm_probs):
        if i_click_probs is None:
            n_clicks = None
        else:
            n_clicks = len(i_click_probs)
        (scans,scans_wpm, clicks, errors) = self.getPossibleResultValues(i_grnd_truth_word, len(i_scan_probs), n_clicks, len(i_err_probs)) 
            
        #Text-entry speed: number of scans and wpm
        avg_scans = np.sum(scans*i_scan_probs)
        std_scans = np.sqrt(np.sum(((scans-avg_scans)**2)*i_scan_probs)) 
        min_scans = self.getMinScans(i_grnd_truth_word)
        min_wpm = self.scansToWpm(min_scans, i_grnd_truth_word) 
        #Do a variable transformastion to find the pdf for wpm
        avg_wpm =  self.scansToWpm(avg_scans, i_grnd_truth_word)  #np.sum(scans_wpm*i_wpm_probs) 
        std_wpm  = avg_wpm - self.scansToWpm(avg_scans+std_scans, i_grnd_truth_word)  # np.sqrt(np.sum(((scans_wpm-avg_wpm)**2)*i_wpm_probs))
        #Number of clicks
        if clicks is not None:
            avg_clicks =  np.sum(clicks*i_click_probs)
            std_clicks = np.sqrt(np.sum(((clicks-avg_clicks)**2)*i_click_probs)) 
        else:
            avg_clicks = 0.0 
            std_clicks = 0.0
        #Number of character errors
        avg_chr_err =  np.sum(errors*i_err_probs)
        std_chr_err = np.sqrt(np.sum(((errors-avg_chr_err)**2)*i_err_probs)) 
        #Final results 
        results = (min_scans, avg_scans, std_scans, min_wpm, avg_wpm, std_wpm, avg_chr_err, std_chr_err, avg_clicks, std_clicks)
        return results

    def getPossibleResultValues(self, i_grnd_truth_word, i_n_scans, i_n_clicks, i_n_errors):
        scans = np.array(range(0,i_n_scans))
        scans_wpm = self.scansToWpm(scans,  i_grnd_truth_word) 
        if i_n_clicks is not None:
            clicks = np.array(range(0, i_n_clicks))
            clicks /= len(i_grnd_truth_word)
        else: clicks = None
        errors = 100.0*np.array(range(0, i_n_errors)) / len(i_grnd_truth_word)
        return (scans, scans_wpm, clicks, errors)

    def getProbStateSeq(self, i_states, i_transitions, i_transition_probs, i_seq, i_display ):
        if i_display:
            print "************************************************************"
            print "i_seq = ", i_seq
            print "************************************************************"
        (states, dest) = self.convertDest(i_states, i_transitions )
        states -= 1 
        seq = np.array(i_seq) - 1
        o_seq_prob = []
        for n in range(0, len(seq)-1):
            (src, dst) = (seq[n], seq[n+1])
            idx = np.nonzero( dest[src] == dst )[0]
            if len( idx ) < 1:
                print "ERR: idx = ", idx, " src = ", src, " dst = ", dst
                raise ValueError("no state has empty transitions!")
            o_seq_prob.extend( i_transition_probs[src][idx] )
        print o_seq_prob
        o_seq_prob = np.array(o_seq_prob)
        return (np.cumprod(o_seq_prob)[-1], o_seq_prob.flatten())
  
    ##################################### Probability computations
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
        t = time.time()
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
  
        #scan_probs2 = -np.inf*np.ones( i_T +1 )
        #Probs for number of clicks to write any word
        click_probs = -np.inf*np.ones( [N, i_T+2]) 
        click_probs[0,0] = 0.0
        #Probs for number of erroneous characters
        err_probs = -np.inf*np.ones(M+i_max_spurious+1)
        for cur_time in range(0, i_T):
            state_probs_prev = np.array(state_probs)
            state_probs[0:N-3] = -np.inf*np.ones(N-3)
            click_probs_prev = np.array(click_probs)
            click_probs[0:N-3,:] = -np.inf
            scan_probs_prev = np.array(scan_probs)
            scan_probs[0:N-3,:] = -np.inf
          
            #Do the non-terminating states first
            for i in range(0, N-3): 
                (p_click, p_miss) = (np.log(i_transition_probs[i][0]), np.log(i_transition_probs[i][1]))
                (dest_click, dest_miss) = (dest[i][0] , dest[i][1])
                psum_click = self.utils.elnprod(state_probs_prev[i], p_click)
                psum_miss  = self.utils.elnprod(state_probs_prev[i], p_miss)
                #State probs
                state_probs[dest_click] = self.utils.elnsum(state_probs[dest_click], psum_click)
                state_probs[dest_miss]  = self.utils.elnsum(state_probs[dest_miss], psum_miss)
                #The scan probs (how long it takes)
                prob_click = scan_probs_prev[i, 0:-i_scan_delays[i]] + p_click
                prob_miss  = scan_probs_prev[i, 0:-i_scan_delays[i]] + p_miss
                tmp_probs = np.vstack( (scan_probs[dest_click,i_scan_delays[i]:], prob_click)).transpose()
                scan_probs[dest_click,i_scan_delays[i]:] = self.utils.expTrick(tmp_probs).flatten()
                tmp_probs = np.vstack( (scan_probs[dest_miss,i_scan_delays[i]:], prob_miss)).transpose()
                scan_probs[dest_miss,i_scan_delays[i]:] = self.utils.expTrick(tmp_probs).flatten()
                #The click probabilities
                prob_click = click_probs_prev[i,0:cur_time+2] + p_click  
                prob_miss =  click_probs_prev[i,0:cur_time+2] + p_miss
                tmp_probs = np.vstack( (click_probs[dest_click,1:cur_time+2], prob_click[0:cur_time+1])).transpose()
                click_probs[dest_click,1:cur_time+2] = self.utils.expTrick(tmp_probs).flatten()
                tmp_probs = np.vstack( (click_probs[dest_miss,0:cur_time+2], prob_miss)).transpose()
                click_probs[dest_miss,0:cur_time+2] = self.utils.expTrick(tmp_probs).flatten()
                #The state id
                (letter_t, col, n_error, m, n_undo) = self.itemizeStateId(i_states[i])
                #At all time steps, except for T, a terminating state can only be reached by clicking 
                #and being busy with a column scans
                if not(self.isTerminatingDest(dest,i,0,N) and col):
                    continue 
                #scan_probs2[cur_time+1] = self.utils.elnsum(scan_probs2[cur_time+1], psum_click) 
                """Char error probs: See if the current letter occurs in remaining part of word
                If the system failed the whole word is taken to be wrong
                if i_grnd_truth_word.find(letter_t, m, M) >= 0:
                """
                #In case of a click an the destination is not the correct state , the 
                #Remaining word from the spurious event is taken to be wrong.
                if dest[i][0] == N-1:
                    m += 1
                else:
                    n_error += 1  
                err_probs[M-m+n_error] = self.utils.elnsum(err_probs[M-m+n_error], psum_click)
            if self.isEarlyBailTestStateProbs(state_probs, cur_time+1, i_T, i_display):
                break
    
        (state_probs, scan_probs, click_probs, err_probs) = self.updateMaxTimeFailureProbs(state_probs, scan_probs, 
            click_probs, err_probs, M, N, i_T, cur_time)
        scan_probs = self.utils.expTrick(scan_probs[-3:,:].transpose()) 
        click_probs = self.utils.expTrick(click_probs[-3:,:].transpose())
        (scan_probs, click_probs, err_probs) = ( np.exp(scan_probs), np.exp(click_probs), np.exp(err_probs)) 
        wpm_probs = np.array(scan_probs)
        print "Total time = ", time.time() - t, " seconds"    
        print "state_probs = ", state_probs[-3:]
        return (scan_probs, click_probs, err_probs, wpm_probs )
 
    def updateMaxTimeFailureProbs(self, state_probs, scan_probs, click_probs, err_probs, M, N, i_T, i_bail_out_time ):
        #A system failure has occured, because the maximum time was reached, include these probabilities
        #These prob are included if the time of early bail out is close to total time allowed
        eps_time = 5
        if (i_bail_out_time + eps_time) < i_T:
            return (state_probs, scan_probs, click_probs, err_probs) 
         
        sum_fail = self.utils.expTrick(np.atleast_2d(state_probs[0:N-3]))[0]
        state_probs[N-3] = self.utils.elnsum(state_probs[N-3], sum_fail)
        #Text-entry rate
        fail_scan_probs = self.utils.expTrick( scan_probs[0:N-3,:].transpose())
        fail_scan_probs = np.vstack( (scan_probs[N-3,:], fail_scan_probs)).transpose()
        scan_probs[N-3,:] = np.array(self.utils.expTrick(fail_scan_probs))
        #scan_probs[-1] = self.utils.elnsum(scan_probs[-1],  sum_fail)
        #Click probs
        if click_probs is not None:
            fail_click_probs = self.utils.expTrick( click_probs[0:N-3,:].transpose())
            fail_click_probs = np.vstack( (click_probs[N-3,:], fail_click_probs)).transpose()
            click_probs[N-3,:] = np.array(self.utils.expTrick(fail_click_probs))
        #Number of character errors
        err_probs[M] = self.utils.elnsum(err_probs[M], sum_fail)
        return  (state_probs, scan_probs, click_probs, err_probs) 

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
                (tmp_row_states, tmp_row_transitions, tmp_row_probs) = self.getRowStates( letter, letter_idx, click, self.letter_info['0'] , display )
                row_states.extend(tmp_row_states)
                #Add the number of scan_delays associated with all states (typically more if a tick sound is added at the beginning
                tmp_scan_delays =  np.ones(len(tmp_row_states), dtype=np.int)
                if self.add_tick:
                    tmp_scan_delays[0] += 1
                scan_delays.extend(tmp_scan_delays)
                states.extend(tmp_row_states)
                row_probs.extend(tmp_row_probs)
                transition_probs.extend(tmp_row_probs)
                for key in tmp_row_transitions.keys():
                        row_transitions[key] = tmp_row_transitions[key]
                        transitions[key] = row_transitions[key]
                for undo in range(0, n_undo):
                    (tmp_col_states, tmp_col_transitions, tmp_col_probs) = self.getColStates( letter, letter_idx, click, 
                        undo,  n_undo,  n_errors, correct_id, self.letter_info['1'], display ) 
                    col_states.extend(tmp_col_states)
                    tmp_scan_delays =  np.ones(len(tmp_col_states), dtype=np.int)
                    if self.add_tick:
                        tmp_scan_delays[0] += 1
                    scan_delays.extend(tmp_scan_delays)
                    states.extend(tmp_col_states)
                    state_ids.append([str(letter_idx),click,undo]) 
                    col_probs.extend(tmp_col_probs)
                    transition_probs.extend(tmp_col_probs)
                    for key in tmp_col_transitions.keys():
                        col_transitions[key] = tmp_col_transitions[key] 
                        transitions[key] = col_transitions[key]
        if display:
           self.displayStates( row_states, col_states, row_transitions, col_transitions, n_undo, n_errors, n_input_letters,  scan_delays)
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
            #The colum-scan id if a click happens - reset undo to zero
            dest_click = self.getStateId(self.config[r][0], i_input_letter_num, i_click, i_undo=0,i_row=False) 
            """Proceed to the next letter if this one is missed, repeat scan at last letter"""
            dest_miss = self.getStateId(row_ids[r+1], i_input_letter_num, i_click, None,i_row=True) 
            states.append(self.getStateId(row_ids[r], i_input_letter_num, i_click, None, i_row=True) )
            transitions[states[-1]]=[dest_click,  dest_miss ] 
            (prob_click,  prob_miss )  = self.clickProb( row_ids[0:-1],  i_letter_info,  gauss_mean, r ,  i_display )
            transition_probs.append(np.array([prob_click, prob_miss]))
        return (states, transitions, transition_probs)
    
    def getColStates(self, i_input_letter, i_input_letter_num, i_click, i_undo, i_n_undo, i_n_errors, i_correct_id, i_letter_info,  i_display ):
        #Compute the current row ids
        states = []
        transition_probs = []
        transitions = {}
        row_ids = self.getGridRow()
        for r in range(0 , len(row_ids)):
            gauss_mean= self.getInputClickInfoCol(i_input_letter, i_letter_info,  i_click,  r )
            dest_list = list(self.config[r])
            for c in range(0, len(dest_list)):
                #The destinations if a click happens
                if (dest_list[c] == i_input_letter) and (i_click == 0):
                    """If the user clicks on the desired letter and nothing else has been written,
                       i.e., all errors undone, and its the end of a word the correct state is reached
                    """
                    dest_click = i_correct_id
                elif dest_list[c] == "D":
                    """Click on backspace/delete"""
                    dest_click = self.getStateId(  row_ids[0], i_input_letter_num-1, i_click-1, i_undo=0, i_row=True)
                elif (i_click+1) >= i_n_errors:
                    """If too many letters have been selected the system fails"""
                    dest_click = 'Failure'
                elif (dest_list[c] == "_") or (dest_list[c] == "."):
                    dest_click = 'Err'
                else:
                    """If a false positive click happens the undo counter is set to zero and the current 
                       click is incremented"""
                    dest_click = self.getStateId( row_ids[0], i_input_letter_num, i_click+1, i_undo=0, i_row=True)
                #The destinations if a click is missed
                if c == ( len(dest_list) - 1): 
                    if ( (i_undo + 1)  >= i_n_undo) and (i_n_undo > 0) :
                        """We've reached an undo after false positive click, go back to row scan"""
                        dest_miss = self.getStateId( row_ids[0], i_input_letter_num, i_click , i_undo=0, i_row=True)
                    else:
                        """Increment the undo counter"""
                        dest_miss = self.getStateId( dest_list[0], i_input_letter_num, i_click , i_undo + 1, i_row=False)
                else:
                    dest_miss = self.getStateId(dest_list[c+1], i_input_letter_num, i_click, i_undo, i_row=False)
                #Get the current state id
                states.append(self.getStateId( dest_list[c], i_input_letter_num, i_click, i_undo, i_row=False))
                #Compute the transition for this state
                transitions[states[-1]]= [dest_click, dest_miss]  
                #The transition probabilities    
                (prob_click,  prob_miss )  = self.clickProb( dest_list ,  i_letter_info[r],  gauss_mean, c,  i_display  )
                transition_probs.append(np.array([prob_click, prob_miss]))
        return (states, transitions, transition_probs)

    def convertDest(self, i_states, i_transitions ):
        state_idx = {}
        o_dest = []
        #The states
        for i in range(0, len(i_states)):
            state_idx[i_states[i]] = i
        #The Destinations  
        for i in range(0,len(i_states)):
            o_dest.append( np.array([state_idx[t] for t in i_transitions[i_states[i]]]))
        o_states = np.array( [i+1 for i in range(0, len(i_states))])
        return (o_states, o_dest) 
    
    def getStateId(self, i_output_letter, i_input_letter_num, i_click, i_undo, i_row):
        """Each state has an id depending on the letter we're trying to write given the input letter, 
        whether we're busy with a row scan (i_row=True, id will have *) or column scan (i_row=False, id will have _), the number of spurious clicks 
        that has happend, the current undo iteration (only applicable to column scans)."""    
        id = str(i_output_letter)
        if i_row:
            id += '*'
        else:
            id += '_'
        if i_click < 0:
            click = 0
        else:
            click = i_click
        if not isinstance(i_input_letter_num, int):
            raise ValueError("Input str should be an int")
        if i_input_letter_num < 0:
            input_letter_num = 0
        else:
            input_letter_num  = i_input_letter_num 
        id += (str(input_letter_num) + "," + str(click))
        if not i_row: 
            id +=  ("," + str(i_undo))
        else: 
            id += "  "
        return id
    
    def itemizeStateId(self, i_state_id):
        (letter_t, col, m, n_error) = (i_state_id[0], bool(i_state_id[1] == '_'), int(i_state_id[2]), int(i_state_id[4]))               
        if not col:
            n_undo = 0
        else:
            n_undo = int(i_state_id[6])
        return (letter_t, col, n_error, m, n_undo)
    
    ##################################### Code Probs
    
    def clickProb(self, seq, time_info, gauss_mean, cnt, i_display ):
        """
        * gauss_mean: The mean of the Gaussian the user is supposed to click on
        * current_mean: The mean of the Gaussian associated with the current scan (row/col cnt)
        * current scan info extracted from time_info
        * seq: All the cells that have to be scanned, currently we're at cell cnt."""
        #The cell's might have different lengths
        delta_time =  np.float32( time_info['t_end'][cnt] - time_info['t_start'][cnt])
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
            click_pdf_end = s.norm.cdf(x=t_end, loc=gauss_mean, scale=self.std)  
            click_pdf_start = s.norm.cdf(x=t_start, loc=gauss_mean, scale=self.std)
            q = click_pdf_end - click_pdf_start
            click_output_0 = fp * ( 1.0 -  ((1.0-self.fr)*q) )
            click_output_1 = 1.0 - click_output_0
            if i_display:
                print "%s g_mean=%.4f,  t_mean=%.4f, q=%.6f  " % (letter,  gauss_mean, current_mean, q),
                print " obs_1 =%.4f, obs_0=%.4f " % ( click_output_1, click_output_0 ),
                print " t_start = %.4f, t_end =  %.4f " % ( t_start + self.click_time_delay, t_end+ self.click_time_delay )
        return (click_output_1,  click_output_0 )
    
    def getFalsePositiveProb(self, i_fp_rate, i_delta_time):
        fp = np.exp(-self.fp_rate * i_delta_time) 
        return fp
        
    ############################################# Display
    
    def displayParams(self, i_grnd_truth_word, i_params):
        (n_errors, n_undo, fr, fp_rate, scan_delay, click_time_delay, std, n_max, draw_tikz, display) = i_params
        max_scans = self.getMaxScans(i_grnd_truth_word, n_max)
        disp_str = "Params for %s:" %  i_grnd_truth_word
        disp_str += " #errors=%d, #undo=%d, draw_tikz=%d, disp=%d,"  % (n_errors, n_undo, draw_tikz, display)
        disp_str +=" fr=%.2f, fp_rate=%.5f, scan_delay=%1.3s, " % (fr, fp_rate, scan_delay)
        disp_str +=" click_delay=%1.3s, click_std=%1.3s," % (click_time_delay, std) 
        disp_str +=" max scans=%d" % (max_scans)
        print disp_str
 
    def tikzStateDiagram(self, i_input_letters, i_states, i_transitions, i_transition_probs):
        params = StateGraphParams(  i_filename=(self.output_dir + "grid1.tikz"), i_scale=0.8)
        params.ns = 0.4
        params.rect_height = 3.0
        params.show_probs = False 
        params.x_offset = 0.8                                  
        graph = StateGraph(params)
        graph.compute( i_input_letters, i_states,  i_transitions,  i_transition_probs)
    
    def displayStates(self, i_row_states, i_col_states, i_row_transitions, i_col_transitions, i_n_undo, i_n_clicks, i_n_input_letters, scan_delays):
        n_letters_rows = len(self.getGridRow())
        n_letters_cols = len(self.letter_pos.keys())
        for click in range(0, i_n_clicks):
            disp_str="======================================================================================="
            disp_str+= "===================================================================="
            print disp_str
            self.__displayStates(i_row_states, i_row_transitions, click, 1, i_n_clicks, n_letters_rows, i_n_input_letters, scan_delays)
            disp_str="---------------------------------------------------------------------------------------"
            disp_str+= "--------------------------------------------------------------------"
            print disp_str
            self.__displayStates(i_col_states, i_col_transitions, click, i_n_undo, i_n_clicks, n_letters_cols, i_n_input_letters, scan_delays)
        print disp_str
            
    def __displayStates(self, i_states, i_transitions, i_click, i_n_undo, i_n_clicks, i_n_output_letters, i_n_input_letters, i_scan_delays):
        for n in range(0, i_n_output_letters):
            for m in range(0, i_n_input_letters):
                for undo in range(0, i_n_undo ):
                    disp_str = ""
                    letter_offset = m*i_n_output_letters * i_n_clicks * i_n_undo
                    click_offset = i_click * i_n_undo * i_n_output_letters
                    undo_offset = undo*i_n_output_letters
                    state_idx = letter_offset + click_offset + undo_offset  + n
                    state_id = i_states[state_idx]
                    disp_str += "%s: [ " % state_id
                    for k in range(0, len(  i_transitions[state_id] ) ):
                        disp_str += "%s" % i_transitions[state_id][k] 
                        if k < (len(  i_transitions[state_id] ) - 1):
                            disp_str += " , "
                    disp_str += (" ], scan_delay = %1.1f " % i_scan_delays[state_idx] )
                 
                    print disp_str

    def displayResults(self, i_results, i_display, i_display_all=False):
        if not i_display:
            return
        (min_scans, avg_scans, std_scans, min_wpm, avg_wpm, std_wpm, avg_err_rate, std_err_rate, avg_cpc, std_cpc) = i_results
        if i_display_all:
            print "min scans=%d, avg scans=%.2f, std scans=%.2f, min_wpm=%.2f" % (min_scans,avg_scans,std_scans,min_wpm)
        r= (avg_cpc, std_cpc, avg_wpm, std_wpm, avg_err_rate, std_err_rate) 
        self.utils.dispResults(r)
       
    def displayProbs(self, i_scan_probs, i_click_probs, i_err_probs):
        print "********************************************************"
        print "Scan probs (for text-entry rate, total number of scans (0 ... T))"
        print "********************************************************"
        self.utils.printMatrix(np.atleast_2d(i_scan_probs))
        print "********************************************************"
        print "Click probs (for number total number of clicks (0 ... T))"
        print "********************************************************"
        if i_click_probs is None:
            print "No click probabilities computed"
        else:
            self.utils.printMatrix(np.atleast_2d(i_click_probs))
        print "********************************************************"
        print "Error probs (for number total number of erroneous characters, 0 ... M+#spurious)"
        print "********************************************************"
        self.utils.printMatrix(np.atleast_2d(i_err_probs))
    
        
    def displaySpeed(self, i_scan_probs, i_input_word):
        
        scans = np.arange(0, len(i_scan_probs))
        avg_scans = np.sum(scans*i_scan_probs)
        std_scans = np.sqrt(np.sum(((scans-avg_scans)**2)*i_scan_probs)) 
        idx_max = np.argmax(i_scan_probs)
      
        print "***************************************************************"
        print "Speed stats"
        print "***************************************************************"
        print "scan_probs sum = ", np.sum( i_scan_probs ) , " avg_scans = ", avg_scans, " std_scans = ", std_scans             
        print "Min scans = ", self.getMinScans(i_input_word) 
        print "Best probs = ", idx_max, " prob = ", i_scan_probs[idx_max]
        
    ####################################### Get
    
    def getMaxScans(self, i_grnd_truth_word, i_n_max):
        max_scans = i_n_max*len(i_grnd_truth_word)*self.n_rows*self.n_cols
        return max_scans
    
    def getLetterPos(self,i_display=False):
        if i_display:
            print "===================================="
            print "Letter Pos"
            print "====================================" 
        self.letter_pos = {}
        for row in range(0, len(self.config)):
            for col in range(0, len(self.config[row])):
                self.letter_pos[self.config[row][col]] = (row,col)
                if i_display:
                    print self.config[row][col], " ", (row,col)
 
    def getLetterDurations(self , i_display=False):
        letter_info = []
        if i_display:
            print "===================================="
            print "Letter Duration Info"
            print "====================================" 
        for row_cnt in range(0, len(self.config)):
            row = self.config[row_cnt]
            durations = self.getSeqDuration(row, i_display)
            letter_info.append(dict(durations))
        durations = self.getSeqDuration(self.getGridRow(), i_display)
        o_info = {'1' : list(letter_info), '0': dict(durations) }
        return o_info
      
    def getSeqDuration(self, i_seq, i_display):
        durations = {}
        durations['t_start'] = np.cumsum( np.array(self.scan_delay*(np.ones(len(i_seq)))) ) 
        if not self.add_tick:
           durations['t_start'] -= self.scan_delay
        durations['t_end'] = durations['t_start']  + self.scan_delay
        durations['gauss_mean'] =  durations['t_start'] + self.click_time_delay  - 0.5*self.scan_delay
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
    
    def getInputClickInfoRow(self, x_id, click_info): 
        letter_pos = self.letter_pos[x_id][0] 
        gauss_mean = click_info['gauss_mean'][letter_pos]
        return gauss_mean
    
    def getInputClickInfoCol(self, x_id, click_info, i_click, i_cur_row): 
        (letter_pos_row,  letter_pos_col)  = ( self.letter_pos[x_id][0] , self.letter_pos[x_id][1] )
        gauss_mean_grnd_truth  = click_info[letter_pos_row]['gauss_mean'][letter_pos_col]
        (letter_pos_row,  letter_pos_col)  = ( self.letter_pos["D"][0] , self.letter_pos["D"][1] )
        gauss_mean_del  = click_info[letter_pos_row]['gauss_mean'][letter_pos_col]
        if i_click == 0:
            """No spurious clicks received"""
            if i_cur_row  == self.letter_pos[x_id][0]: 
                """The intentional click is still on track: That is, no spurious clicks received and the current row iteration is equal to the ground truth row."""
                gauss_mean =  gauss_mean_grnd_truth
            else:
                """It is assumed that the user would like to undo row scan by wait if the wrong row was selected"""
                gauss_mean =  None
        else:
            if i_cur_row   == self.letter_pos["D"][0]: 
                """Spurious clicks, the user is in the correct row to be able to UNDO"""
                gauss_mean = gauss_mean_del 
            else:
                """It is assumed that the user would like to undo row scan by wait if the wrong row was selected"""
                gauss_mean =  None   
        return gauss_mean

    def getGridRow(self):
        """Return the letters associated with selecting a row, i.e., the first click"""
        return [letters[0] for letters in self.config]
    
    def getMinScans(self, i_word):
        """Get the minimum time (measured as #scans) it should take to write a sentence using Grid 2"""
        t = 0
        for letter in i_word:
            (row, col) =  self.letter_pos[letter]
            total = row+1 + col + 1
            if self.add_tick: 
                total += 2
            t += total
            #print "sentence = ", i_sentence, " letter = ", letter, " row = ", row, " col = ", col, " total = ", total, " t = " , t
        return t
 
    def scansToWpm(self, i_scans, i_grnd_truth_word):
        wpm = self.scansToWpmConst(i_grnd_truth_word)/ i_scans
        return wpm
    
    def scansToWpmConst(self, i_grnd_truth_word):
        c = 60.0*len(i_grnd_truth_word) / (self.word_length*self.scan_delay)
        return c

    def isTerminatingDest(self, i_dest, i_state_num, i_dest_num, i_total_states):
        if i_dest[i_state_num][i_dest_num] > (i_total_states-4):
            return True
        return False
    
    def isTerminatingState(self, i_state_num, i_total_states ):
        if i_state_num < (i_total_states-3):
            return False
        return True
    
    def isEarlyBailTestStateProbs(self, i_state_probs, i_time, i_T, i_display):
        #Look at the total prob in cases where the simulation should stop (e.g., if the correct state is reached)
        #Bail out if this total prob does not change much
        is_bail = False
        p_tmp = np.exp(i_state_probs[-3:])
        p_sum = np.sum(p_tmp) 
        n_states = len(i_state_probs)
        str_probs = self.utils.stringVector(p_tmp) 
        if i_display:
            print "Current time %d of %d, probs=%s, prob sum=%.4f, total states=%d" % (i_time, i_T, str_probs, p_sum, n_states)
        if np.abs(p_sum - 1.0) < self.early_bail_prob:
            print "Ending at time %d of %d, probs=%s, prob sum=%.4f, total states=%d" % (i_time, i_T, str_probs, p_sum, n_states) 
            return True
        return False
    
    def itemizeResults(self, i_results):
        (min_scans, avg_scans, std_scans, min_wpm, avg_wpm, std_wpm, avg_chr_err, std_chr_err, avg_clicks, std_clicks) = i_result 
        return (min_scans, avg_scans, std_scans, min_wpm, avg_wpm, std_wpm, avg_chr_err, std_chr_err, avg_clicks, std_clicks)
    
if __name__ ==  "__main__":
    app = GridSimulation(i_display=False)
    w = "abcdefghijklmnopqrstuvwxyz_."
    m = app.getMinScans(w)
    print "minimum scans = " , m, " avg scans per letter = ", m / float(len(w))
    

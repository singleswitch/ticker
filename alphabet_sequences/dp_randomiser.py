#!/usr/bin/env python
import sys
sys.path.append("../")
import numpy as np
from utils import  WordDict
import cPickle, os
from utils import Utils 

class AlphabetOptimiseSpacingRandomiser(object):
    def __init__(self):
        self.utils=  Utils()
        self.__randomiser = AlphabetRandomiser()
        self.__loadDictionary(i_load_from_file=True)
       
    def setParams(self, i_nchannels,  i_min_dist=3, i_max_repeat=2):
        self.__randomiser.setParams(i_nchannels, i_min_dist, i_max_repeat)
    
    def channel3Hack(self, i_possible_sequences):
        print "================================================"
        print "Using channel 3 hack"
        o_sequences = []
        for s in i_possible_sequences:
            group_space = np.array(self.__randomiser.getGrouping( s,i_nchannels=3, i_erase_spacers=False, i_display=False))
            group_no_space = self.__randomiser.getGrouping( s,i_nchannels=3, i_erase_spacers=True, i_display=False)
            for n in range(0, len(group_no_space)):
                g = np.array(group_no_space[n])
                for letter in g:
                    idx = np.nonzero(g == letter)[0]
                    if len(idx) < 2:
                        group_space[n][-1] = letter
            new_seq = group_space.transpose().flatten()
            o_sequences.append(list(new_seq))
        return (o_sequences, True) 
    
    def compute(self, i_alphabet):
        possible_alphabets = self.__getPossibleAlphabets(i_alphabet)
        print "POSSIBLE"
        print possible_alphabets
        
        final_sequences = []
        nchannels = self.__randomiser.getChannels()
        for alphabet in possible_alphabets:
            print "*************************************"
            print "".join(alphabet)
            group = self.__randomiser.getGrouping( alphabet, i_nchannels=nchannels, i_erase_spacers=False, i_display=True )
            print "group = ", group
            (possible_sequences, niterations, max_iterations, is_valid) = self.__randomiser.compute(alphabet, i_display=False)
            #FIXME:
            if nchannels == 3:
                (possible_sequences, is_valid) = self.channel3Hack( possible_sequences)
            if is_valid:
                for s in possible_sequences:
                    final_sequences.append(s)
        print "NUMBER OF FINAL SEQ SIZE= ", len(final_sequences)
        if len(final_sequences) < 1:
            return final_sequences
        self.displayDistances(i_alphabet,final_sequences)
        
    def displayDistances(self, i_alphabet, final_sequences):
        nchannels = self.__randomiser.getChannels()
        final_scores  = []
        for seq  in final_sequences:
            alphabet_length = len(self.__randomiser.getSequenceAlphabet(i_alphabet))
            seq_alphabet = self.__randomiser.getSequenceAlphabet(seq )
            start_index = len(seq_alphabet) - alphabet_length 
            #score  = self.getSeqMutualInformation( start_index, list(seq_alphabet) )
            score  = self.getSeqDistances( start_index, list(seq_alphabet) )
            final_scores.append(score)
        final_scores = np.array(final_scores)
        sort_arg = np.argsort(final_scores)
        for idx in sort_arg:
            print "============================================================================"
            print   " [ score = ", final_scores[idx] , "] sequence = ", ''.join(final_sequences[idx]) 
            for letter in final_sequences[idx]:
                print letter,
            print " "
            print "-------------------------------------------------------------------------------------------------------------------"
            groups = np.array(self.__randomiser.getGrouping(final_sequences[idx] , nchannels, i_erase_spacers=False, i_display=False))
            for group in groups:
                for letter in group:
                    print letter,
                print " "
 
    
    ##################################  Optimisation based on integer letter distances
    def getSeqDistances(self, i_start_index, i_letter_seq):
        distance = 0.0
        window_length = self.__randomiser.getMinDist()
        search_seq = np.array(i_letter_seq)[0:i_start_index]
        
        for n in range(i_start_index, len(i_letter_seq)-window_length):
            cur_letter= i_letter_seq[n]
            end_index = n+window_length+1
            if end_index > i_letter_seq:
                end_index = len(i_letter_seq)
            if end_index <= (n+1):
                continue
            next_letters = list(i_letter_seq[n-window_length:n])
            next_letters.extend(i_letter_seq[n+1:end_index])
            print "Cur letter = ",  cur_letter, " neighbours = ", ''.join(next_letters), " letter seq = ", "".join(i_letter_seq)
            
        return  distance
    
    ################################## Validation tests
    def __validateGroupsForSimilarSounds(self, i_alphabet):
        #validate the sequence to see if any successive sounds are similar
        for n in range(1, len(i_alphabet)):
            letter = i_alphabet[n]
            neighbour = i_alphabet[n-1]
            is_valid = self.__randomiser.validateSimilarSoundContraints(letter, neighbour)
            if not is_valid:
                return False
        return True
    ################## Determining the possible alphabets  
    
    def __getPossibleAlphabets(self, i_alphabet):
        #Get all possible combinations of the alphabet, based on the constraint
        #That no group can be preceded by more than 1 spacer. 
        nchannels = self.__randomiser.getChannels()
        modulus = len(i_alphabet) % nchannels
        if modulus == 0:
            return [self.__seqAlphabetToChannels(i_alphabet)]
        possible_alphabets = []
        nspacers =  nchannels - modulus
        spacer_combinations = self.__getSpaceCombinations(nchannels, nspacers)
        #Insert a spacer when there is a 1 in the spacer combination
        group_length = self.__getGroupLength(  i_alphabet, i_nspacers= nspacers) 
        for combination in spacer_combinations:
            alphabet = list(i_alphabet)
            for n in range(0, len(combination)):
                if combination[n] < 1:
                    alphabet.insert(n*group_length, '*')
            list_alphabet = self.__seqAlphabetToChannels( alphabet, nspacers)
            if self.__validateGroupsForSimilarSounds( list_alphabet ):
                possible_alphabets.append(list_alphabet)
                #print grouped_alphabet.transpose()
                #print ''.join(list_alphabet)
        return possible_alphabets
    
    def __seqAlphabetToChannels(self, i_alphabet, i_nspacers=0):
        """Map the alphabet sequence abc to channel display so that abc is in sequence in a specific channel"""
        group_length = self.__getGroupLength(  i_alphabet, i_nspacers= i_nspacers) 
        alphabet = np.array(i_alphabet)
        grouped_alphabet = []
        for n in range(0, group_length):
            grouped_alphabet.append(alphabet[range(n,len(alphabet),group_length)])
        grouped_alphabet = np.array(grouped_alphabet)
        list_alphabet = list(grouped_alphabet.flatten())
        return list_alphabet
            
    def __getGroupLength(self, i_alphabet, i_nspacers=0):
        nchannels = self.__randomiser.getChannels()
        group_length = (len(i_alphabet) + i_nspacers) / nchannels
        return group_length
            
    def __getSpaceCombinations(self, i_nchannels, i_nspacers):
        """Get the possible ways the spacer could be placed - only one spacer per group
           is allowed. First determine the number of spacers, e.g., if there are 
           2 spacers for five channels a vector of the form [0,0,1,1,1] will be computed,
           and all its permutations will be returned. A zero is returned when the group
           should start with a spacer"""
        #Only allow a channel to start with one spacer
        combinations = np.int32(np.ones(i_nchannels))
        for n in range(0, i_nspacers):
            combinations[n] = 0
        combinations = list(combinations) 
        #Find all permutations of the values in combinations
        permutations = np.array([ p  for p in self.utils.xpermutations(combinations) ])
        #Only use the unique ones
        o_permutations = [permutations[0,:]]
        for n in range(1, permutations.shape[0]):
            combination = permutations[n,:]
            stored = np.array(o_permutations)
            unique_thresh = permutations.shape[1]
            idx = np.nonzero( np.sum( np.int32( stored == combination ), 1) == unique_thresh )[0]
            if len(idx) == 0:
                o_permutations.append(combination)
        return np.array(o_permutations)
    
    ################################# Optimisation based on entropy
    
    def getSeqMutualInformation(self, i_start_index, i_letter_seq):
        mutual_info = 0.0
        window_length = 1 #self.__randomiser.getMinDist()
        for n in range(i_start_index, len(i_letter_seq)-window_length):
            cur_letter= i_letter_seq[n]
            end_index = n+window_length+1
            if end_index > i_letter_seq:
                end_index = len(i_letter_seq)
            if end_index <= (n+1):
                continue
            #next_letters = list(i_letter_seq[n-window_length:n])
            next_letters = i_letter_seq[n+1:end_index]
            letter_mutual_info = [self.getLetterMutualInformation(cur_letter,letter)for letter in next_letters]
            mutual_info += np.sum(np.array(letter_mutual_info))
        return mutual_info
            
    def getLetterMutualInformation(self, i_cur_letter, i_next_letter):
        log_px = np.log(self.__letter_probs[i_cur_letter]) - np.log(2)
        log_py = np.log(self.__letter_probs[i_next_letter]) - np.log(2)
        key = ''.join([i_cur_letter, i_next_letter])
        if not self.__letter_pair_probs.has_key(key):
            key = ''.join([i_next_letter, i_cur_letter])
        pxy =  self.__letter_pair_probs[key]
        log_pxy = np.log(pxy)
        if np.abs(pxy) < 1E-6:
            mutual_information = 0.0
        else:
            log_pxy -= np.log(2)
            mutual_information = np.sum(pxy*(log_pxy - log_px - log_py)  )
        return mutual_information
  
    def __marginalLetterLogProbs(self):
        alphabet = np.unique(self.__word_list)
        max_word_length = self.__word_list.shape[1]
        #Compute the probabilities that ln=to a specific letter for all letters in the alphabet
        o_probs = {}
        for letter in alphabet:
            o_probs[letter] = 0.0
        for letter in alphabet:
            for n in range(0 , max_word_length):
                #Compute p(l1,l2,... | ln=letter)
                idx = np.nonzero( self.__word_list[:,n] == letter )[0]
                if len(idx) < 1: 
                    continue
                log_probs = np.array(self.__word_log_probs[idx].flatten())
                log_prob_sum = self.utils.expTrick( log_probs.reshape([1,len(log_probs)]))
                o_probs[letter] += np.exp(log_prob_sum)
        for letter in alphabet:
            o_probs[letter] /= 26.0
        return o_probs
    
    def __marginalJointLogProbs(self):
        alphabet = np.unique(self.__word_list)
        max_word_length = self.__word_list.shape[1]
        o_prob_pairs = {}
        for j in range(0, len(alphabet)):
            for k in range(j+1, len(alphabet)):
                letter1 = alphabet[j]
                letter2  = alphabet[k]
                print "Computing pair ", letter1, letter2
                for n in range(0 , max_word_length):
                    idx = np.nonzero( np.logical_or( self.__word_list[:,n] == letter2, self.__word_list[:,n] == letter1) )[0]
                    if len(idx) < 1: 
                        continue
                    key = ''.join([letter1, letter2])
                    if not o_prob_pairs.has_key(key):
                        o_prob_pairs[key] = 0.0
                    else:
                        log_probs = np.array(self.__word_log_probs[idx].flatten())
                        log_prob_sum = self.utils.expTrick( log_probs.reshape([1,len(log_probs)]))
                        o_prob_pairs[key] += np.exp(log_prob_sum)
        prob_sum = np.sum(o_prob_pairs.values())
        for key in o_prob_pairs.keys():
            o_prob_pairs[key] /= prob_sum
        return o_prob_pairs
    
    def __wordListsToDict(self, i_list_words, i_vals):
        wi = [ [i_list_words[n], i_vals[n]] for n in range(0, len(i_vals))]
        return dict(wi) 
        
    def __loadDictionary(self, i_load_from_file=True):
        filename = "entropy_dict.cPickle"
        if i_load_from_file and os.path.exists(filename):
            d = self.utils.loadPickle(filename)
            self.__word_list = np.array(d['word_list'])
            self.__word_log_probs = np.array(d['word_log_probs'])
            self.__letter_probs = d['letter_probs']
            self.__letter_pair_probs = d['letter_pair_probs']
            prob_sum = np.sum(self.__letter_pair_probs.values())
            for key in self.__letter_pair_probs.keys():
                self.__letter_pair_probs[key] /= prob_sum
            prob_sum = np.sum(self.__letter_probs.values())
            for key in self.__letter_probs:
                self.__letter_probs[key] /= prob_sum
            print "Loaded word list from file: ", self.__word_list.shape
            print "Loaded word_log_probs from file", self.__word_log_probs.shape
            print "letter_pair_probs len = ", len(self.__letter_pair_probs.values()), " sum = ", np.sum(self.__letter_pair_probs.values())
            print "letter probs len = ", len(self.__letter_probs.values()), " sum = ", np.sum(self.__letter_probs.values())
        else:
            d = WordDict("../dictionaries/nomon_dict.txt")
            self.__max_word_length = np.max(d.word_lengths)
            self.__word_log_probs = d.log_probs
            print "MAX WORD LENGTH = ", self.__max_word_length
            self.__word_list = []
            cnt_word = 1
            for word in d.words:
                if (cnt_word % 1000) == 0:
                    print " Word = ", cnt_word, " out of ", len(d.words),
                    print " Word list shape = ", np.array(self.__word_list).shape
                if word == '._':
                    cur_word = []
                    for n in range(0, self.__max_word_length):
                        cur_word.append('.')
                else:
                    max_repeat = np.int(np.ceil( float(self.__max_word_length) / float(len(word))))
                    cur_word = []
                    for n in range(0, max_repeat):
                        cur_word.extend(word)
                    cur_word = np.array(cur_word)[0:self.__max_word_length]
                if not(len(cur_word) == self.__max_word_length):
                    print "Word = ", cur_word, " len = ", len(cur_word), " should be ", self.__max_word_length
                    raise ValueError("Word length ERROR")
                self.__word_list.append(cur_word)
                cnt_word += 1
            self.__word_list = np.array(self.__word_list)
            self.__word_log_probs = np.array(self.__word_log_probs)
            self.__letter_probs = self.__marginalLetterLogProbs()
            self.__letter_pair_probs = self.__marginalJointLogProbs()
            
            print "SAVING TO FILE", filename, "Word list shape = ", self.__word_list.shape
            d = {'word_list': self.__word_list, 'word_log_probs': self.__word_log_probs,
                 'letter_probs': self.__letter_probs, 'letter_pair_probs':self.__letter_pair_probs}
            self.utils.savePickle(d, filename)
            
        #Make sure the word probs sum to one
        sum_probs = np.sum(np.exp(self.__word_log_probs))
        diff =  sum_probs - 1.0
        if diff > 1E-3:
            self.__word_log_probs -= np.log(sum_probs)
class AlphabetRandomiser(object):
    ########################################### Init functions
    def __init__(self):  
        self.loadSimilarSoundConstraints()
        self.setParams( i_nchannels = 0)
        
        
    def loadSimilarSoundConstraints(self):
        constraints = [['a','h'],['q','k'],['m','n'],['b','d'],['a','i']]
        new_constraints = [[constraints[n][1], constraints[n][0]] for n in range(0,len(constraints))]
        constraints.extend(new_constraints) 
        self.__constraints = np.array(constraints)

    def setParams(self, i_nchannels,  i_min_dist=3, i_max_repeat=2):
        self.__nchannels = i_nchannels
        self.__min_dist = i_min_dist
        self.__max_letter_repeat = i_max_repeat
        
    ########################################### Main functions
            
    def compute(self, i_alphabet, i_display=False):
        possible_sequences = [list(i_alphabet)]
        cur_group_idx = 0
        n_iterations = len(i_alphabet) * (self.__max_letter_repeat - 1)
        for n in range(0,  n_iterations):
            #With each iteration we choose a set of letters
            if i_display:
                print "****************************************************"
                print "Iteration", n, " out of ", n_iterations
            if np.array(possible_sequences).shape[1] >= (self.__max_letter_repeat*n_iterations):
                return (possible_sequences, n, n_iterations, True)
            new_sequences = []
            for cur_alphabet in possible_sequences:
                isGroupDone = self.__isGroupDone(i_alphabet, cur_alphabet, cur_group_idx)
                if isGroupDone:
                    new_sequences = self.__addLetter(cur_alphabet,'*', new_sequences, i_display=i_display)
                    continue
                #Get the alphabet without any spaces, as neighbours no not include spacings
                cur_alphabet_seq = self.getSequenceAlphabet(cur_alphabet)
                next_possible_letters = cur_alphabet_seq[0:(-self.__min_dist)]
                cur_group = self.getGrouping( cur_alphabet, self.__nchannels, i_display=False)[cur_group_idx]
                for letter in next_possible_letters:
                    if not self.validateLetter( letter, cur_group, cur_alphabet_seq ):
                        continue
                    new_sequences = self.__addLetter(cur_alphabet, letter, new_sequences, i_display=i_display)
            if len(new_sequences) < 1:
               return (self.__appendSpaces(possible_sequences, len(i_alphabet)), n, n_iterations, False)
            possible_sequences = list(new_sequences)
            cur_group_idx += 1
            if cur_group_idx >= self.__nchannels:
                cur_group_idx = 0
        return (self.__appendSpaces(possible_sequences, len(i_alphabet)), n, n_iterations, True)
        
    def getChannels(self):
        return self.__nchannels
    
    def getMinDist(self):
        return self.__min_dist
    
    ############################################# Display the alphabet in different ways
    def getGrouping( self, i_alphabet, i_nchannels, i_erase_spacers=True, i_display=False):        
        alphabet = list(i_alphabet)
        modulus =  len(i_alphabet) % i_nchannels
        if modulus  > 0:
            n_spaces =  i_nchannels - modulus 
            for n in range(0, n_spaces):
                alphabet.append('*')
        alphabet = np.array(alphabet)    
        alphabet_groups = []
        for n in range(0, i_nchannels):
            group = alphabet[range(n, len(alphabet), i_nchannels)]
            if i_erase_spacers:
                group = [letter for letter in group if not letter=='*']
            alphabet_groups.append(group)
            if i_display:
                print group
        return alphabet_groups 
    
    def getSequenceAlphabet(self, i_alphabet):
        #Return the alphabet in sequence without the spaces
        return np.array([letter for letter in i_alphabet if  not letter == '*'])

    def getAlphabetLength(self, i_alphabet):
        seq_alphabet = self.getSequenceAlphabet(i_alphabet)
        return len(seq_alphabet)
    
 ################################## Validation tests

    def validateLetter(self, i_letter, i_cur_group, i_sequence):
        #This is the main validation test to determine if a letter should be selected as part 
        #of the random sequence
        if not self.validateLetterInCurrentGroup(i_cur_group, i_letter):
            return False
        if not self.validateSuccssiveGroupNeigbours(i_cur_group, i_letter):
            return False
        if not self.validateMaxLetterRepeat( i_sequence, i_letter ):
            return False
        if not self.validateSimilarSoundContraints( i_letter, i_sequence[-1]):
            return False
        if not self.validateNeigbourDistances(i_letter, i_sequence):
            return False
        return True
    
    def validateLetterInCurrentGroup(self, i_current_group, i_letter):
        #check if the letter appears in the right group - we always choose 
        #letters from group 1 then group 2 etc
        #We therefore have to select a letter from the current group
        idx = np.nonzero( np.array(i_current_group) == i_letter )[0]
        if len(idx) == 0:
            return False
        return True
        
    def validateSimilarSoundContraints(self, i_letter, i_immediate_neigbour):
        #Make sure the other neighbour constraints based on how similar the sounds are, are valid
        #That is, sounds that are similar are not allowed to be neighbours
        #i_immediate_neigbour is a letter that has already been chosen.
        #i_letter can possibly be chosen - it can't be chosen if it sounds too similar to i_immediate_neigbour
        idx = np.nonzero(self.__constraints[:,0] == i_letter)[0]
        if len(idx) > 0:
            for neighbour in self.__constraints[idx,1]:
                valid_similar_sound = ( not(neighbour == i_immediate_neigbour ))
                if not valid_similar_sound:
                    return False
        return True
            

    def validateNeigbourDistances(self, i_letter, i_seq_alphabet): 
        #Check if none of the current neighbours within max_dist have been
        #a neighbour within max_dist before
        idx = np.nonzero( i_seq_alphabet == i_letter)[0]
        start_index = idx - self.__min_dist
        end_index = idx + self.__min_dist + 1
        if start_index < 0:
            start_index = 0
        if end_index > len(i_seq_alphabet):
            end_index = len(i_seq_alphabet)
        prev_neighbours = i_seq_alphabet[range(start_index, end_index)]
        idx = np.nonzero( np.logical_not( prev_neighbours == i_letter) )[0]
        prev_neighbours = prev_neighbours[idx]
        cur_neighbours = i_seq_alphabet[-self.__min_dist:]
        for neighbour in cur_neighbours:
            valid_neighbours = len(np.nonzero(  prev_neighbours == neighbour )[0]) < 1
            if not valid_neighbours:
                return False
        return True

    def validateMaxLetterRepeat(self, i_sequence, i_letter):
        #Check how many times this letter already occurs, shoud be less than max num
        if len(np.nonzero( np.array(i_sequence) == i_letter)[0]) >= self.__max_letter_repeat:
            return False
        return True

    def validateSuccssiveGroupNeigbours(self, i_group, i_letter):
        #In any one group we do not allow the same letter to occur successively
        if i_letter == i_group[-1]:
            return False
        return True
    
    def validateFinalSequence(self, i_alphabet, i_display=True):
        print "========================================================"
        print "Validating the neighbours of group (rest can be verified manually)"
        grouping =  self.getGrouping(i_alphabet, self.__nchannels, i_erase_spacers=False,i_display=True)
        seq_alphabet = np.array(self.getSequenceAlphabet(i_alphabet))
        for letter in seq_alphabet:
            idx = np.nonzero(seq_alphabet == letter)[0]
            neighbours = []
            for n in idx:
                start_index = n - self.__min_dist
                if start_index < 0:
                    start_index = 0
                end_index = n + self.__min_dist + 1
                if end_index > len(seq_alphabet):
                    end_index = len(seq_alphabet)
                neighbours.extend(seq_alphabet[start_index:n])
                neighbours.extend(seq_alphabet[n+1:end_index])
            unique_neighbours = np.unique(neighbours)
            if i_display:
                print "Neigbours of all letters in the current alphabet:"
                print "Letter = ", letter, " Neigbours = ", neigbours
            if len(  unique_neighbours ) < len(   unique_neighbours ):
                print "INVALID SEQUENCE, BASED ON NEIGHBOURS!"
                return False
        print "VALID SEQUENCE"
        return True
    ###################################### Pivate Functions
    def __appendSpaces(self, i_sequences, i_original_alphabet_length):
        """Append spaces to all possible sequences until the correct length is reached""" 
        o_sequences = list(i_sequences)
        for sequence in o_sequences:
            for n in range(len(sequence), self.__max_letter_repeat * i_original_alphabet_length):
                sequence.append('*')
        return o_sequences
    
    def __addLetter(self, i_sequence, i_letter, i_new_sequences, i_display=False):
        #All the validation tests were successful - add a possible sequence
        o_sequences = list(i_new_sequences)
        sequence = list(i_sequence)
        sequence.append(i_letter)
        o_sequences.append(sequence)
        if i_display:
            print "Adding new sequence ", ''.join(sequence)
        return o_sequences

    def __isGroupDone(self, i_original_alphabet, i_cur_alphabet, i_cur_group_idx):
        #A group can be done before the end of the sequence is reached (if there are spacers)
        original_group = self.getGrouping( i_original_alphabet, self.__nchannels, i_display=False)
        cur_group = self.getGrouping( i_cur_alphabet, self.__nchannels, i_display=False)[i_cur_group_idx]
        max_group_len = self.__max_letter_repeat*len(original_group[i_cur_group_idx])
        if len(cur_group) >= max_group_len: 
            return True
        return False
    ####################################### Examples 
    
    def fiveChannelExample(self, i_min_dist):
        self.setParams(i_nchannels=5, i_min_dist=i_min_dist)
        test_grouping = np.array([['a' , 'b' , 'c' ,'d', 'e','f'],
                                  ['*' , 'g' , 'h' , 'i' ,'j', 'k'],
                                  ['*' , 'l' , 'm' , 'n' ,'o', 'p'],
                                  ['q' , 'r' , 's' , 't' ,'u' ,'v'],
                                  ['w' , 'x' , 'y' , 'z' ,'_' ,'.']])
        cur_alphabet = np.array(test_grouping.transpose().flatten())
        print "ORIGINAL ALPHABET = ",  ''.join(cur_alphabet)
        print "GROUPING = "
        print test_grouping
        (o_sequences,  n, n_iterations, is_valid) = self.compute( cur_alphabet, i_display = True)
        print "OUTPUT:"
        print "------------"
        if is_valid:
            print "Valid sequences:"
            for s in o_sequences:
                self.validateFinalSequence(s, i_display=False)
        else:
            print "NO VALID SEQ"
            
class AlphabetSequenceComputer():
    def compute(self):
        disp_single_config =  False
        min_letter_separation = 4
        channels = 5
        if disp_single_config:
            randomiser = AlphabetRandomiser() 
            randomiser.fiveChannelExample( min_letter_separation )
        else:
            test_alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','_', '.']
            randomiser = AlphabetOptimiseSpacingRandomiser()
            randomiser.setParams(i_nchannels=channels, i_min_dist=min_letter_separation)
            randomiser.compute(test_alphabet)
 
if __name__=="__main__":
    r = AlphabetSequenceComputer()
    r.compute()
        

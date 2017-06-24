
import numpy as np
import cPickle

"""* This file contains everything that has to be loaded from lookuptables e.g., the sound file lengths, the alphabet etc
    * Lookuptables stored in files, all depend on a root directory"""

class FileLoader():
    def __init__(self , i_root_dir):
        self.setRootDir(i_root_dir)
         
    def setRootDir( self,  i_root_dir):
        self.__root_dir =  i_root_dir
        
    def getRootDir(self):
        return self.__root_dir
    
class LookupTables(FileLoader):
    def __init__(self, i_dir="./"):
        FileLoader.__init__(self, i_dir)
        self.__file_lengths =  SoundFileLengthLoader(self.getRootDir() + "config/channels");
        self.__alphabet = AlphabetLoader(self.getRootDir() + "config/channels");
        self.__letter_utterances =  LetterUtteranceLookupTables()
    
    def setChannels(self, i_nchannels):
        self.__alphabet.load(i_nchannels)
        self.__nchannels = i_nchannels
        
    def getSoundFileLengths(self):
        return self.__file_lengths.load(self.__nchannels)
    
    def getAlphabetLoader(self):
        return self.__alphabet
    
    def getChannels(self):
        return self.__nchannels
    
    def getLetterUtteranceFromIndex(self, i_index):
        return  self.__letter_utterances.getLetterStringFromIndex(self, i_index)

class SoundFileLengthLoader(FileLoader):
    def __init__(self, i_dir):
        FileLoader.__init__(self, i_dir)
   
    def load(self, i_nchannels):
        file_name =  self.getRootDir() + str(i_nchannels) + "/sound_lengths.cPickle"
        f = open( file_name, 'r')
        file_lengths  = cPickle.load(f)
        f.close()
        return file_lengths

class AlphabetLoader(FileLoader):
    """This class contains all the loading functions associated with loading the alphabet, and configuring it for multiple channels usage
       Input: 
            * The setChannels functions is expected to be called to change the configuration
            * Otherwise the get functions should be called for different representations of the same alphabet."""
    
    ###################################### Init functions
    def __init__(self, i_dir ):
        FileLoader.__init__(self, i_dir)

    ##################################### Load the alphabet
            
    def load(self, i_nchannels):
        file_name = self.getRootDir() + str(i_nchannels) + "/alphabet.txt"
        file_name = file(file_name)
        alphabet = file_name.read()
        file_name.close()
        alphabet = alphabet.split('\n')[0]
        alphabet = alphabet.split(" ")[0]
        alphabet = [letter for letter in alphabet if not (letter == '') ]
        array_alphabet = np.array(alphabet)
        repeat = np.array([len(np.nonzero(array_alphabet == letter)[0]) for letter in alphabet if not( letter == '*') ])
        idx = np.nonzero(repeat == repeat[0])[0]
        if not ( len(idx) == len(repeat) ):
            print "Repeat = ", repeat
            raise ValueError("Error in alphabet, all letters should repeat the same number of times")
        repeat = repeat[0]
        self.__alphabet = list(alphabet)
        alphabet_len = len(self.__alphabet) / repeat
        self.__unique_alphabet = list( self.__alphabet[0:alphabet_len])
        self.__alphabet_len = self.__getAlphabetLength(self.__alphabet)
        self.__unique_alphabet_len = self.__getAlphabetLength(self.__unique_alphabet)
        
    ##################################### Get functions 
    def getAlphabet(self, i_with_spaces=True):
        if i_with_spaces:
            return self.__alphabet
        return self.__getSequenceAlphabet(self.__alphabet)
    
    def getAlphabetLen(self,  i_with_spaces=True):
        if i_with_spaces:
            return len(self.__alphabet)
        return  self.__alphabet_len
    
    def getUniqueAlphabet(self, i_with_spaces=True):
        if i_with_spaces:
            return self.__unique_alphabet
        return self.__getSequenceAlphabet(self.__unique_alphabet)
    
    def getUniqueAlphabetLen(self,  i_with_spaces=True):
        if i_with_spaces:
            return len(self.__unique_alphabet)
        return self.__unique_alphabet_len
 
    ##################################### Private functions
    
    def __getSequenceAlphabet(self, i_alphabet):
        #Return the alphabet in sequence without the spaces
        return [letter for letter in i_alphabet if  not letter == '*']

    def __getAlphabetLength(self, i_alphabet):
        seq_alphabet = self.__getSequenceAlphabet(i_alphabet)
        return len(seq_alphabet)
    
    ##################################### Display functions
    def plotIntegerDistances(self):
        alphabet = np.array(i_alphabet)
        sequence = self.getSequenceAlphabet(self.__alphabet)
        for letter in alphabet:
            idx = np.nonzero(sequence == letter)[0]
            if not (len(idx) == 2):
                disp_str = "Letter " + letter + " occurances= "  +str(len(idx))
                raise ValueError(disp_str)
            pylab.plot( dx[0], idx[1], '+' )
            pylab.text(idx[0]+0.3, idx[1], letter)
        
class  LetterUtteranceLookupTables():
    def __init__(self):
        self.__letter_dict = {1:"first",2:"second",3:"third",4:"fourth",5:"fifth",6:"sixth",7:"seventh",8:"eighth",9:"ninth",10:"tenth", 
                             11:"elenvth",12:"twelfth",13:"thirteenth",14:"fourteenth",15:"fifteenth", 16:"sixteenth",17:"seventeenth",18:"eighteenth",19:"nineteenth",20:"twentieth",
                             21:"twentyfirst",22:"twentysecond",23:"twentythird",24:"twentyfourth",25:"twentyfifth",26:"twentysixth",27:"twentyseventh",28:"twentyeighth",29:"twentyninth",30:"thirtieth", 
                             31:"thirtyfirst",32: "thirtysecond",33:"thirtythird",34:"thirtyfourth",35:"thirtyfifth",36:"thirtysixth",37:"thirtyseventh",38:"thirtyeighth",39:"thirtyninth",40:"fourtieth", 
                             41:"fourtyfirst",42: "fourtysecond",43:"fourtythird",44:"fourtyfourth",45:"fourtyfifth",46:"fourtysixth",47:"fourtyseventh",48:"fourtyeighth",49:"fourtyninth",50:"fiftieth"} 
        
    def   getLetterStringFromIndex(self, i_index):
        return self.__letter_dict[i_index]
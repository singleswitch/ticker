
 

import sys, os, copy
sys.path.append("../../")
from utils import Utils, WordDict 
import numpy as np

u = Utils()
n_random = 30
in_file = "phrases_00.txt"
out_file = "phrases.txt"
phrases = u.loadText(in_file).replace("\r", "")
phrases = phrases.split('\n')[0:-1]

indices = np.random.randint(low=0, high=(len(phrases)-1), size=n_random)
new_phrases = [phrases[idx] for idx in indices] 
new_phrases = ("\n").join(new_phrases)
u.saveText(new_phrases, out_file)

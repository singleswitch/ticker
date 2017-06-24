
import sys, os, copy
sys.path.append("../../")
from utils import Utils, WordDict 

u = Utils()
my_dict = WordDict("../../dictionaries/nomon_dict.txt")
my_dict = my_dict.listsToDict(my_dict.words, my_dict.log_probs)
in_file = "phrases_00.txt"
out_file = "tmp.txt"
punctuations = ['?', '!']
phrases = u.loadText(in_file).replace("\r", "")
phrases = phrases.split('\n')[0:-1]
new_phrases = []
for n in range(0, len(phrases)):
    if phrases[n] == "":
        continue
    phrase = str(phrases[n])
    phrases[n] = phrase.lower()
    for m in range(0, len(punctuations)):
        phrases[n] = phrases[n].replace(punctuations[m], ".")
    if not (phrases[n][-1] == "."):
        phrases[n] = phrases[n] +  "."
    index = phrases[n].find(",")
    if index >= 0:
        continue
    index = phrases[n].find("'")
    if index  >= 0:
        continue
    phrase = str(phrases[n])
    words = phrase.split("_")
    is_valid = True
    for word in words:
        if word == ".":
            cur_word = "."
        else:
            cur_word = word + "_"
        if not (my_dict.has_key(cur_word)):
            print "Can not find ", cur_word
            is_valid=False
            break
    if not is_valid:
        continue
    new_phrases.append(str(phrases[n]))
    #print "n  = ", n, " old = ", phrases[n], " index = ", index, " new_phrases = ", new_phrases[-1]

new_phrases = ("\n").join(new_phrases)
u.saveText(new_phrases, out_file)
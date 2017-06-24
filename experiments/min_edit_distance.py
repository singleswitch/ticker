

import numpy as np
import cPickle
import pylab as p

class MinEditDistance(object):
    
    def __init__(self):
        self.subst_penalty = 1
        
    def compute(self, i_grnd_truth, i_test_word):
        self.cost  = np.zeros([len(i_test_word), len(i_grnd_truth)])
        self.cost[0,:] = range(0, len(i_grnd_truth))
        self.cost[:,0] = range(0,len(i_test_word))
        self.back_track_ids = np.atleast_2d(np.int32(np.zeros(self.cost.shape)))
        self.back_track_ids[0,:] = 1
        self.back_track_ids[:,0] = 2
        if not (i_grnd_truth[0] == i_test_word[0]):
            self.cost[0,0] = self.subst_penalty
            self.back_track_ids[0,0] = 3
            self.cost[0,1:] += self.subst_penalty
            self.cost[1:,0] += self.subst_penalty
        else:
            self.back_track_ids[0,0] = 0
        #Compute the cost matrix
        for x in range(1, len(i_grnd_truth)):
            for y in range(1, len(i_test_word)):
                insert_cost = self.cost[y-1,x] +  1
                del_cost = self.cost[y, x-1]   +  1
                if i_test_word[y] == i_grnd_truth[x]:
                    subst_cost = self.cost[y-1,x-1]
                else:
                    subst_cost = self.cost[y-1,x-1] + self.subst_penalty
                local_cost = np.array([subst_cost, del_cost, insert_cost])
                best_idx = np.argmin(local_cost)
                self.cost[y,x] = local_cost[best_idx]
                if (best_idx == 0) and (not(i_test_word[y] == i_grnd_truth[x])):
                    best_idx = 3
                self.back_track_ids[y,x] = best_idx
        return self.cost[-1,-1]
    
    def getCostMatrix(self):
        return self.cost
    
    def getBackTrackIds(self):
        return self.back_track_ids
    
    def displayResults(self, i_grnd_truth, i_test_word):
        print "GRND TRUTH: ", i_grnd_truth
        print "TEST WORD:", i_test_word
        print "COST MATRIX:"
        print self.cost
        ids = {'0' : 'C',  '1':'D' , '2':'I' , '3':'S'}
        o_ids = [ids[str(self.back_track_ids[-1,-1])]]
        print "OPERATIONS APPLIED TO GRND: C=CORRECT, D=DELETE, I=INSERT, S=SUBSTITUTE"
        prev_x = self.back_track_ids.shape[1]-1
        prev_y = self.back_track_ids.shape[0]-1
        while not( (prev_x == 0) and (prev_y == 0)) :
            next_x = int(prev_x) - 1
            next_y = int(prev_y) - 1
            if o_ids[-1] == 'D':
                next_y = int(prev_y)
            elif (prev_id == 'I'):
                next_x = prev_x
            if (next_x < 0) or (next_y < 0):
                break
            prev_id = ids[str(self.back_track_ids[next_y, next_x])]
            o_ids.append(prev_id)
            prev_x = next_x
            prev_y = next_y
        o_ids.reverse()
        #All operations are performed on the ground truth
        o_string = []
        cost_check = 0
        grnd_idx = 0
        test_idx = 0
        for n in range(0, len(o_ids)):
            cur_string = [i_grnd_truth[grnd_idx], i_test_word[test_idx] , o_ids[n]]
            if o_ids[n] == 'D':
                cur_string[1] = ' '
                cost_check += 1
                grnd_idx += 1
            elif o_ids[n] == 'I':
                cost_check+=1
                test_idx += 1
            elif o_ids[n] == 'S':
                cost_check += self.subst_penalty
                grnd_idx += 1
                test_idx += 1
            elif o_ids[n] == 'C':
                grnd_idx += 1
                test_idx += 1
            if test_idx >= len(i_test_word):
                test_idx = len(i_test_word) - 1
            if grnd_idx >= len(i_grnd_truth):
                grnd_idx >= len(i_grnd_truth) - 1
            o_string.append(cur_string)
        o_string = np.array(o_string).transpose()
        print o_string
        print "MIN EDIT DIST: ", self.cost[-1,-1]
        print "COST CHECK (DIAGNOSTIC): ", cost_check
            
if __name__ ==  "__main__":
    min_dist = MinEditDistance()
    grnd_truth="zeros" #interest"
    test_word="hero" #industry"
    min_dist.compute(grnd_truth, test_word)
    min_dist.displayResults(grnd_truth,test_word)
"""
NOTE: You are only allowed to edit this file between the lines that say:
    # START EDITING HERE
    # END EDITING HERE

This file contains the base Algorithm class that all algorithms should inherit
from. Here are the method details:
    - __init__(self, num_arms, horizon): This method is called when the class
        is instantiated. Here, you can add any other member variables that you
        need in your algorithm.
    
    - give_pull(self): This method is called when the algorithm needs to
        select an arm to pull. The method should return the index of the arm
        that it wants to pull (0-indexed).
    
    - get_reward(self, arm_index, reward): This method is called just after the 
        give_pull method. The method should update the algorithm's internal
        state based on the arm that was pulled and the reward that was received.
        (The value of arm_index is the same as the one returned by give_pull.)

We have implemented the epsilon-greedy algorithm for you. You can use it as a
reference for implementing your own algorithms.
"""

import numpy as np
import math
# Hint: math.log is much faster than np.log for scalars

class Algorithm:
    def __init__(self, num_arms, horizon):
        self.num_arms = num_arms
        self.horizon = horizon
    
    def give_pull(self):
        raise NotImplementedError
    
    def get_reward(self, arm_index, reward):
        raise NotImplementedError

# Example implementation of Epsilon Greedy algorithm
class Eps_Greedy(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # Extra member variables to keep track of the state
        self.eps = 0.1
        self.counts = np.zeros(num_arms)
        self.values = np.zeros(num_arms)
    
    def give_pull(self):
        if np.random.random() < self.eps:
            return np.random.randint(self.num_arms)
        else:
            return np.argmax(self.values)
    
    def get_reward(self, arm_index, reward):
        self.counts[arm_index] += 1
        n = self.counts[arm_index]
        value = self.values[arm_index]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[arm_index] = new_value

# START EDITING HERE
# You can use this space to define any helper functions that you need
def kld(p, q):
    if p==0: return -math.log(1-q)
    elif p==1: return -math.log(q)
    else: return p*math.log(p/q) + (1-p)*math.log((1-p)/(1-q))

def search(lower, upper, p_t_i, u_t_i, rhs, thres = 1e-2):
    diff, summ = upper-lower, upper+lower
    q_t_i = summ/2
    lhs = u_t_i*kld(p_t_i,q_t_i)

    if lhs <= rhs:
        return q_t_i if diff<thres else search(q_t_i, upper, p_t_i, u_t_i, rhs)
    else:
        return lower if diff<thres else search(lower, q_t_i, p_t_i, u_t_i, rhs)
# END EDITING HERE

class UCB(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # START EDITING HERE
        self.ucb = np.zeros(num_arms)
        self.emp_mean = np.zeros(num_arms)
        self.pulls = np.zeros(num_arms)
        self.ctr = 0
        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE
        self.ctr += 1
        if self.ctr <= self.num_arms: return self.ctr-1
        return np.argmax(self.ucb)
        # END EDITING HERE
        
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        self.pulls[arm_index] += 1
        total_pulls = sum(self.pulls)
        n = self.pulls[arm_index]
        self.emp_mean[arm_index] = ((n-1)*self.emp_mean[arm_index] + reward)/n
        ucb_factor = [math.sqrt(2*math.log(total_pulls)/puls) if puls else 0 for puls in self.pulls]
        self.ucb = self.emp_mean + ucb_factor
        # END EDITING HERE


class KL_UCB(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # You can add any other variables you need here
        # START EDITING HERE
        self.klucb = np.zeros(num_arms)
        self.emp_mean = np.zeros(num_arms)
        self.pulls = np.zeros(num_arms)
        self.ctr = 0
        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE
        self.ctr += 1
        if self.ctr <= self.num_arms: return self.ctr-1
        return np.argmax(self.klucb)
        # END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        self.pulls[arm_index] += 1
        n = self.pulls[arm_index]
        self.emp_mean[arm_index] = ((n-1)*self.emp_mean[arm_index] + reward)/n

        rhs = math.log(sum(self.pulls))
        for i in range(self.num_arms):
            u_t_i = self.pulls[i]
            p_t_i = self.emp_mean[i]
            if u_t_i: self.klucb[i] = search(p_t_i, 1, p_t_i, u_t_i, rhs) if p_t_i!=1 else 1
        # END EDITING HERE

class Thompson_Sampling(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # You can add any other variables you need here
        # START EDITING HERE
        self.thomps = np.zeros(num_arms)
        self.success = np.zeros(num_arms)
        self.pulls = np.zeros(num_arms)
        self.failure = lambda x: self.pulls[x] - self.success[x]
        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE
        return np.argmax(self.thomps)   
        # END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        if reward == 1: self.success[arm_index] += 1
        self.pulls[arm_index] += 1
        self.thomps = [np.random.beta(self.success[i]+1, self.failure(i)+1) for i in range(self.num_arms)]
        # END EDITING HERE

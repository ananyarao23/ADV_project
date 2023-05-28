import sys
import numpy as np
import math
import time
from joblib import Parallel, delayed
data_file = open(sys.argv[2], "r")
lines = data_file.readlines()
l = len(lines)
for i in range(-1,0):
    lines_arr = lines[i].split()
    global gamma
    gamma = lines_arr[1]
for j in range(4):
    lines_arr = lines[j].split()
    if j==0:
        global num_states 
        num_states = lines_arr[1] #reading number of states from given file
    if j==1:
        global num_actions
        num_actions = lines_arr[1] #reading number of actions from given file
    if j==2:
        global start_state 
        start_state = lines_arr[1] #reading the start state
    else:
        global end_state
        end_state = lines_arr[1] #reading the end state
shape_arr = (num_states, num_states, num_actions) #3d array consisting of the transition probabilities from s1 to s2 
trans_prob_a = np.empty(shape_arr)                #by taking action a in index [s1][s2][a]
reward_a = np.empty(shape_arr)                    #similarly 3d array of rewards for respective indices
for i in range(4,l):
    line_arr = lines[i].split()
    trans_prob_a[line_arr[1]][line_arr[3]][line_arr[2]] = line_arr[5]
    reward_a[line_arr[1]][line_arr[3]][line_arr[2]] = line_arr[4]
value_state_curr = [0]*num_states                 #value that each state has initially
value_state_next = []                             
eps = 0
delta = 0
while(delta>=eps):
    eps = 0.01
    for i in range(num_states):
        v_a = []
        for j in range(num_actions):
            sum = 0
            for k in range(num_states):
                sum = sum + trans_prob_a[i][k][j]*(reward_a[i][k][j] + gamma*value_state_curr[k])
            v_a.append(sum)
        value = max(v_a)
        value_state_next.append(value)
    diff = []
    for x in range(num_states):
        a = value_state_next[x] - value_state_curr[x]
        diff.append(a)
    delta = max(diff)
    value_state_curr.clear()
    value_state_curr = np.copy(value_state_next)                #current values getting updated for next iteration
    value_state_next.clear()                                    #cleared the list for recording next iteartion values 
for i in range(num_states):
    print(value_state_curr[i],"\n")

    







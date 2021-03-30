#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import statistics as stat

def sign(x):
    if x > 0:
        return 1
    else:
        return -1

data_size = 200
Tau = 0.1
repeatTimes = 10000
t = 0
ans=[]

while(t < repeatTimes):
    x = np.empty([data_size]) 
    y_h_pos = np.ones([data_size]) # h(x) array for s = +1
    y_h_neg = -1 * np.ones([data_size]) # h(x) array for s = -1
    y_target = np.zeros([data_size])
    
    for i in range(data_size):
        x[i] = np.random.uniform(-1,1)
    x = sorted(x)
    thetaList = np.empty(len(x))
    thetaList[0] = -1
    for i in range(len(x) - 1):
        thetaList[i+1] = (x[i] + x[i+1])/2

    Ein = np.zeros([2,data_size]) # row 1 for s = +1, row 2 for s = -1


    #initialization

    for i in range(data_size):
        y_target[i] = np.random.choice([sign(x[i]), -1 * sign(x[i])], p = [1 - Tau, Tau])
        if y_h_pos[i] != y_target[i]:
            Ein[0,0] += 1
        if y_h_neg[i] != y_target[i]:
            Ein[1,0] += 1

    #dynamic programming

    for j in range(1,data_size):
        y_h_pos[j-1] = -1
        if y_h_pos[j-1] != y_target[j-1]:
            Ein[0,j] = Ein[0,j-1] + 1
        else:
            Ein[0,j] = Ein[0,j-1] - 1
        y_h_neg[j-1] = 1
        if y_h_neg[j-1] != y_target[j-1]:
            Ein[1,j] = Ein[1,j-1] + 1
        else:
            Ein[1,j] = Ein[1,j-1] - 1
    Ein = Ein / data_size
    index = np.where(Ein == np.amin(Ein))
    g_multi=[]
    for i in range(len(index[0])):
        if index[0][i] == 0:
            s = 1
        else:
            s = -1
        theta = thetaList[index[1][i]]
        g_multi.append(s+theta)
    g_min = np.where(g_multi == np.amin(g_multi))

    theta = thetaList[index[1][g_min]]
    
    if index[0][g_min] == 0:
        # s = 1
        Eout = (1 - 2 * Tau)*(abs(theta) / 2) + Tau
    else:
        # s = -1
        Eout = (1 - 2 *Tau)*(1 - abs(theta)/2) + Tau 
    ans.append(Eout - np.amin(Ein))
    t+=1
mean = sum(ans) / repeatTimes
median = stat.median(ans)
print("The mean is", mean)
print("The median is",median)


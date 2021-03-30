#!/usr/bin/env python
# coding: utf-8

# In[5]:


from liblinearutil import *
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures


# load data
train_dat = pd.read_csv("https://www.csie.ntu.edu.tw/~htlin/course/ml20fall/hw4/hw4_train.dat", header = None, sep=" ")
test_dat = pd.read_csv("https://www.csie.ntu.edu.tw/~htlin/course/ml20fall/hw4/hw4_test.dat", header = None, sep=" ")
x_train = train_dat.iloc[:,:6].to_numpy()
y_train = train_dat.iloc[:,6].to_numpy()
x_test = test_dat.iloc[:,:6].to_numpy()
y_test = test_dat.iloc[:,6].to_numpy()


# featrue transformation 
poly = PolynomialFeatures(2)
x_train_trans = poly.fit_transform(x_train)
x_test_trans = poly.fit_transform(x_test)

# C = 1 / 2*lambda

lam_log =np.array([-4,-2,0,2,4])
lam = np.power(np.full(len(lam_log),10.0),lam_log)
C = np.reciprocal(2*lam)


# for problem 18

D_train = x_train_trans[:120,:]
D_val = x_train_trans[120:,:]

prob = problem(y_train[:120], D_train)

Eval = np.empty(len(C))
for i in range(len(C)):
    param = parameter('-s 0 -c {} -e 0.000001'.format(C[i]))
    model = train(prob, param)
    p_labs, p_acc, p_vals = predict(y_train[120:], D_val, model)
    ACC, MSE, SCC = evaluations(y_train[120:], p_labs)
    Eval[i] = (100 - ACC) / 100
print(Eval)
print('choose log_lambda = -2')

# choose lam_log = -2, which means we choose C[1]
param = parameter('-s 0 -c {} -e 0.000001'.format(C[1]))
prob18 = problem(y_train[:120], D_train)
model = train(prob18, param)
p_labs, p_acc, p_vals = predict(y_test, x_test_trans, model)
ACC, MSE, SCC = evaluations(y_test, p_labs)
Eout = (100 - ACC) /100
print('Eout for problem 18 is',Eout)


# In[ ]:





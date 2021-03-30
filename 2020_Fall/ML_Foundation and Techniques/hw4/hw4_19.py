#!/usr/bin/env python
# coding: utf-8

# In[6]:


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


# for problem 19

# choose lam_log = -2, which means we choose C[1]
param = parameter('-s 0 -c {} -e 0.000001'.format(C[1]))
prob19 = problem(y_train, x_train_trans)
model = train(prob19, param)
p_labs, p_acc, p_vals = predict(y_test, x_test_trans, model)
ACC, MSE, SCC = evaluations(y_test, p_labs)
Eout = (100 - ACC) /100
print('Eout for problem 19 is',Eout)


# In[ ]:





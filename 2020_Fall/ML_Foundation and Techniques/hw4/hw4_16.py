#!/usr/bin/env python
# coding: utf-8

# In[2]:


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

prob = problem(y_train, x_train_trans)
Eout = np.empty(len(C))

# for Problem 16

for i in range(len(C)):
    param = parameter('-s 0 -c {} -e 0.000001'.format(C[i]))
    model = train(prob, param)
    p_labs, p_acc, p_vals = predict(y_test, x_test_trans, model)
    ACC, MSE, SCC = evaluations(y_test, p_labs)
    Eout[i] = (100 - ACC) / 100
print(Eout)
print('The best Eout comes from log_lambda =',lam_log[1])


# In[ ]:





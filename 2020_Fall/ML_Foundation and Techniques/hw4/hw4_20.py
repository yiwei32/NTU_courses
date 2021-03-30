#!/usr/bin/env python
# coding: utf-8

# In[8]:


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



# 5-fold 
k = 5
Ecv = np.empty(len(C))
for i in range(len(C)):
    Esum = 0
    for j in range(k):
        D_val = x_train_trans[j*40:(j+1)*40, :]
        y_val = y_train[j*40:(j+1)*40]
        D_train_cv = np.delete(x_train_trans, slice(j*40,(j+1)*40), 0)
        y_train_cv = np.delete(y_train, slice(j*40,(j+1)*40), 0)
        prob = problem(y_train_cv, D_train_cv)
        param = parameter('-s 0 -c {} -e 0.000001'.format(C[i]))
        model = train(prob, param)
        p_labs, p_acc, p_vals = predict(y_val, D_val, model)
        ACC, MSE, SCC = evaluations(y_val, p_labs)
        Esum += (100 - ACC) / 100
    Ecv[i] = Esum / k
print(Ecv)
print('The best log_lambda is',lam_log[1],'and the corresponding Ecv is',Ecv[1])


# In[ ]:





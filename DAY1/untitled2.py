# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 12:15:53 2019

@author: vijay
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


data=pd.read_csv('headbrain.csv')


x=data["Head Size(cm^3)"].values
y=data["Brain Weight(grams)"].values

xmean=np.mean(x)
ymean=np.mean(y)
x1=data.iloc[:,2:3]



from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x1,y)
y_pred=regressor.predict(x1)


sst=0
ssr=0
for i in range(0,len(x)):
    sst=sst+((y[i]-ymean)**2)
    ssr=ssr+((y[i]-y_pred[i])**2)
    
r2=1-(ssr/sst)
print(r2)

r3=regressor.score(x1,y)
print(r3)

    
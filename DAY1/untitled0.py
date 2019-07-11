# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 11:39:24 2019

@author: vijay
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('Salary_Data.csv')
#a = data["Head Size(cm^3)"].values
#y=data.iloc[:,3].values

x = data.iloc[:,0:1].values
y=data.iloc[:,1].values

#plt.scatter(x,y,color='red')
#%matplotlib auto



from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = .2, random_state=0)


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()     #object create

regressor.fit(x_train,y_train)
m=regressor.coef_
c=regressor.intercept_


xx=input("enter the x value:")
l=xx.split()
for i in l:
    print(regressor.predict([[float(i)]]))
    


#y75=(m*xx)+c
#y977=regressor.predict([[7.5]])

plt.scatter(x_train,y_train,color='red')
plt.scatter(x_test,regressor.predict(x_test),color='green')

plt.scatter(x_test,y_test,color='blue')
plt.plot(x_train,regressor.predict(x_train),color='blue')


from sklearn.metrics import mean_squared_error
y_pred=regressor.predict(x_test)
rmse = np.sqrt(mean_squared_error(y_test,y_pred))
rmse

sample = 0
for i in range(0,len(y_test)):
    sample += (y_test[i]-y_pred[i])**2
    
    res=sample/len(y_test)
    
import math
math.sqrt(res)
 


    




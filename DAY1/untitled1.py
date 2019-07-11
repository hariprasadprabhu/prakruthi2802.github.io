import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


data=pd.read_csv('headbrain.csv')


x=data["Head Size(cm^3)"].values
y=data["Brain Weight(grams)"].values

xmean=np.mean(x)
ymean=np.mean(y)

upper=0
lower=0
for i in range(0,len(x)):
    upper=upper+(x[i]-xmean)*(y[i]-ymean)
    lower=lower+(x[i]-xmean)**2
res1=upper/lower
print(res1)    
    

res0=ymean-(res1*xmean)
print(res0)
#we manually calculate here


x1=data.iloc[:,2:3] #we need to convert it to array 

from sklearn.linear_model import LinearRegression  #inbuilt library which will calculate regression
regressor = LinearRegression()

regressor.fit(x1,y) #FIT IS USED TO TRAIN THE MACHINE... machine will give m and c value itself
m=regressor.coef_
c=regressor.intercept_

print(m)
print(c)





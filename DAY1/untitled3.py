import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data=pd.read_csv('50_Startups.csv')

x=data.iloc[:,0:4].values
y=data.iloc[:,4].values


from sklearn.preprocessing import LabelEncoder
lEncoder = LabelEncoder()  #create object
x[:,3]=lEncoder.fit_transform(x[:,3])   #to convert the particular table  into catogorical table to represent in 0,1,2,3..

from sklearn.preprocessing import OneHotEncoder
ohEncoder=OneHotEncoder(categorical_features=[3])

x=ohEncoder.fit_transform(x).toarray()

x=x[:,1:] #delete it 1 column because in 3 we can predict by seeing 1 column


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = .2, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

regressor.fit(x_train,y_train)

y_pred=regressor.predict(x_test) #prediction values of all the items in x test

score=regressor.score(x_test,y_test)  #test values

from sklearn.metrics import mean_squared_error
mse=mean_squared_error(y_test,y_pred)**(1/2)


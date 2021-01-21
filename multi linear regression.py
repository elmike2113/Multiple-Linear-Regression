import numpy as np  #for numerical operations such as mean, median etc.
import matplotlib.pyplot as plt #for plotting graphs
import pandas as pd #for data manipulation

# Importing the dataset
dataset = pd.read_csv('D:/Strokx ML/Day 2/50_Startups.csv') # read the data from the CSV file.
X = dataset.iloc[:, :-1] #prints the whole input data i.e, prints columns upto second last
print(X)
y = dataset.iloc[:, 4] #prints 4th column

#Convert the column into categorical columns

states=pd.get_dummies(X['State'])#drop first for data maipulation
print(states)

# Drop the State coulmn
X=X.drop('State',axis=1)
print(X)

# concat the dummy variables
X=pd.concat([X,states,y],axis=1)  
print(X)
print(y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
import seaborn as sns
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()  
regressor.fit(X_train, y_train) #fitting

#plt.scatter(X=dataset['Florida','New York'], y=dataset['Profit']) #to plot use scatter plot
#plt.show()

# Predicting the Test set results
y_pred = regressor.predict(X_test)
print(y_pred)

from sklearn.metrics import r2_score  #for accuracy calculation
score=r2_score(y_test,y_pred)
print(score)


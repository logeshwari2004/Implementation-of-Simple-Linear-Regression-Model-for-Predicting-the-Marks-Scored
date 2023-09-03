# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored
## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.
## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook
## Algorithm
1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas.
## Program:
```
Program to implement the simple linear regression model for predicting the marks scored.
Developed by:LOGESHWARI.P
RegisterNumber:212221230055  
```
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset = pd.read_csv('student_scores.csv')
dataset.head()
X = dataset.iloc[:,:-1].values
X
Y = dataset.iloc[:,1].values
Y
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 1/3,random_state = 0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred = regressor.predict(X_test)
Y_pred
Y_test
plt.scatter(X_train,Y_train,color="blue")
plt.plot(X_train,regressor.predict(X_train),color="brown")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(X_test,Y_test,color="blue")
plt.plot(X_train,regressor.predict(X_train),color="brown![Screenshot (10)](https://user-images.githubusercontent.com/93427345/162911172-55fef480-7db3-4d81-b069-feb5782bb30c.png)
") 
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```

## Output:

![162911250-581850e0-b777-4fed-a90a-17d92b748969](https://github.com/logeshwari2004/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/94211349/7a654587-7aed-4bca-aeb7-0471b79b5cf9)
![162911373-41529fba-1a38-47ab-9bec-48f972e51a61](https://github.com/logeshwari2004/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/94211349/0fb67853-4e13-4e44-a3be-9d955444e569)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.

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
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt
df=pd.read_csv("student_scores.csv")
print(df.tail())
print(df.head())
df.info()
x=df.iloc[ :,:-1].values
print(x)
y=df.iloc[ :,-1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
print(x_train)
print(x_test)
print(y_train)
print(y_test)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
Y_pred=reg.predict(x_test)
print(Y_pred)
print(y_test)
a=Y_pred-y_test
print(a)
plt.scatter(x_train,y_train,color="green")
plt.plot(x_train,reg.predict(x_train),color="red")
plt.title('Training set (H vs S)')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(x_test,y_test,color="blue")
plt.plot(x_test,reg.predict(x_test),color="green")
plt.title('Testing set (H vs S)')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

```

## Output:
## df.head()
![277258313-317b36cd-bdeb-4caa-b66b-a1d5b4f1c44b](https://github.com/logeshwari2004/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/94211349/39771e71-6ac0-4e35-8c4d-32ca1fb3cc7c)
## df.tail()
![277258539-5657fbe1-b98e-4838-9855-b8809af4aba2](https://github.com/logeshwari2004/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/94211349/6c3e0d8e-7f9e-44e1-8901-b96d0511d467)
## Array value of X
![277258823-5cf6eac7-9b71-4358-9732-dcc0f3e856ff](https://github.com/logeshwari2004/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/94211349/c12a593f-d5b9-4758-8899-716ba6d51547)
## Array value of Y
![277258943-b1ccbeef-e935-433f-9f18-4f3912e5b605](https://github.com/logeshwari2004/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/94211349/bfc09d6e-d800-4759-81af-669410ab12a6)
## Array of y prediction
![277260592-5b1eb4af-68fd-4530-a9c9-59f799f1befa](https://github.com/logeshwari2004/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/94211349/fffec7ea-0082-4f0a-b4ff-3125012c34c6)
## Array values of Y test
![277260685-60816dbe-0442-441e-960a-e50e56bbcd4e](https://github.com/logeshwari2004/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/94211349/a59f219b-7137-4f9b-b413-3732c26b20a4)
## Training Set Grap
![266405293-8e9bcfca-9a88-479b-b772-05f8efc7829d](https://github.com/logeshwari2004/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/94211349/9e952b64-667b-4f0d-a04a-c83304cc0be2)
## Test Set Graph
![266405418-1ba0c0f7-b89f-4bcf-9103-d7fd3993c2ec](https://github.com/logeshwari2004/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/94211349/a049d61b-a34e-4d1f-b30d-bc7923a249e9)
## Values of MSE, MAE and RMSE
![277259536-f750da60-89df-4459-ac05-732303189c50](https://github.com/logeshwari2004/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/94211349/cda186c3-c297-4237-84e7-30961db45b22)
## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.

# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load California housing data, select features and targets, and split into training and testing sets.
2.Scale both X (features) and Y (targets) using StandardScaler.
3.Use SGDRegressor wrapped in MultiOutputRegressor to train on the scaled training data.
4.Predict on test data, inverse transform the results, and calculate the mean squared error.

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: RAMYA S
RegisterNumber:  212224040268
*/
```
```
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
dataset = fetch_california_housing()
df=pd.DataFrame(dataset.data, columns=dataset.feature_names)
df['HousingPrice']=dataset.target
print(df.head())
```
```
X = df.drop(columns=['AveOccup', 'HousingPrice'])
Y = df[['AveOccup', 'HousingPrice']]
X_train, X_test, Y_train, Y_test=train_test_split(X, Y, test_size=0.2, random_state=42)
scaler_X=StandardScaler()
scaler_Y=StandardScaler()
X_train=scaler_X.fit_transform(X_train)
X_test=scaler_X.transform(X_test)
Y_train=scaler_Y.fit_transform (Y_train)
Y_test=scaler_Y.transform(Y_test)
sgd=SGDRegressor(max_iter=1000, tol=1e-3)
print(X_train)
print(X_test)
print(Y_train)
print(Y_test)
```
```
multi_output_sgd = MultiOutputRegressor (sgd)
multi_output_sgd.fit(X_train, Y_train)
Y_pred=multi_output_sgd.predict(X_test)
Y_pred = scaler_Y.inverse_transform(Y_pred)
Y_test=scaler_Y.inverse_transform(Y_test)
mse=mean_squared_error(Y_test, Y_pred)
print("Mean Squared Error:", mse)
print("\nPredictions: \n", Y_pred[:5])
```

## Output:
![multivariate linear regression model for predicting the price of the house and number of occupants in the house](sam.png)
![image](https://github.com/user-attachments/assets/7076dc28-5a6b-4336-b0d9-4dfa30e1e3cb)
![image](https://github.com/user-attachments/assets/5d8a23c5-a777-42d0-9f69-43171b2b66b9)
![image](https://github.com/user-attachments/assets/b28d1975-930a-48b8-9b08-0e39dfe4b5d5)
![image](https://github.com/user-attachments/assets/d1f66e04-79ce-44a7-8601-42c0c99141d6)

## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.

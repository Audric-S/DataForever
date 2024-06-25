import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def perform_regression(X, y, algo='Linear Regression', **kwargs):
    if algo == 'Linear Regression':
        max_iter = kwargs.pop('max_iter', None)  # Récupère max_iter s'il est passé
        model = LinearRegression(max_iter=max_iter, **kwargs)
    elif algo == 'Decision Tree Regressor':
        model = DecisionTreeRegressor(**kwargs)
    
    model.fit(X, y)
    y_pred = model.predict(X)
    
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mse)
    
    return mse, r2, mae, rmse

def perform_classification(X, y, algo, max_iter=None, max_depth=None):
    if algo == 'Logistic Regression':
        model = LogisticRegression(max_iter=max_iter)
    elif algo == 'Decision Tree Classifier':
        model = DecisionTreeClassifier(max_depth=max_depth)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return classification_report(y_test, y_pred)

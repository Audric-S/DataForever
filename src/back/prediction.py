import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error
from sklearn.preprocessing import LabelEncoder

def perform_regression(X, y, algo, max_iter=None, max_depth=None):
    if algo == 'Linear Regression':
        model = LinearRegression()
    elif algo == 'Decision Tree Regressor':
        model = DecisionTreeRegressor(max_depth=max_depth)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return mean_squared_error(y_test, y_pred)

def perform_classification(X, y, algo, max_iter=None, max_depth=None):
    if algo == 'Logistic Regression':
        model = LogisticRegression(max_iter=max_iter)
    elif algo == 'Decision Tree Classifier':
        model = DecisionTreeClassifier(max_depth=max_depth)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return classification_report(y_test, y_pred)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer

veri = pd.read_csv('linear_model.csv')

veri.x = (veri.x - veri.x.min()) / (veri.x.max() - veri.x.min())
veri.y = (veri.y - veri.y.min()) / (veri.y.max() - veri.y.min())

X = veri.x.values.reshape(-1,1).astype(float)
y = veri.y.values.reshape(-1,1).astype(float)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("y = %0.2f"%model.coef_+"x+%0.2f"%model.intercept_)

from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import median_absolute_error, r2_score

print("R^2-score:",r2_score(y_test, y_pred))
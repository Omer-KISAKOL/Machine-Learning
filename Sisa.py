import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as sp
from sklearn.tree import DecisionTreeClassifier as dtc
from sklearn.metrics import accuracy_score, confusion_matrix


np.random.seed(146)

df = pd.read_csv("SISA.txt")

df.fillna(0, inplace = True)

features = df.iloc[:,2:35]

classes = df.Hospitalized

X_Train, X_Test, y_Train, y_Test = sp(features, classes, random_state = 0, test_size = 0.28)

model = dtc()
model.fit(X_Train, y_Train)
y_Pred = model.predict(X_Test)

print(accuracy_score(y_Test, y_Pred))

conf_mat = confusion_matrix(y_Test, y_Pred)

plt.figure(figsize=(8,4))
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(conf_mat, annot = True, fmt = 'd')

plt.ylabel('Gerçek')
plt.xlabel('Tahmin')
plt.show()


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

veri = pd.read_excel('iris.xlsx')
features = veri.iloc[:, 0:4]
classes = veri.iloc[:, -1]

from sklearn.model_selection import train_test_split as sp

X_train, X_test, y_train, y_test = sp(features, classes, random_state = 0, test_size = 0.32)

from sklearn.neighbors import KNeighborsClassifier as knn

model = knn(n_neighbors = 3, metric = 'euclidean')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score, confusion_matrix

print(accuracy_score(y_test, y_pred))

conf_mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(conf_mat, annot = True, fmt = 'd')

plt.ylabel('Gerçek')
plt.xlabel('Tahmin')
plt.show()
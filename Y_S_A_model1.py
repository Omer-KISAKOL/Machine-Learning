import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense 
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split as sp
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.python.distribute.merge_call_interim import maybe_merge_call

def createModel():
    model = Sequential()
    model.add(Dense(100))
    model.add(Dense(3, activation='sigmoid'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = createModel()

df = pd.read_csv("iris.data")

features = df.iloc[:, 0:3]
classes = df.iloc[:,-1]

encoder = LabelEncoder

classes = encoder
classes = to_categorical

X_Train, X_Test, y_Train, y_Test = sp(features, classes, test_size = 0.3)

model.fit(X_Train, y_Train, epochs=100)
y_Pred = model.predict(X_Test)

print(accuracy_score(y_Test, y_Pred))

conf_mat = confusion_matrix(y_Test, y_Pred)

plt.figure(figsize=(8,4))
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(conf_mat, annot = True, fmt = 'd')

plt.ylabel('Gerçek')
plt.xlabel('Tahmin')
plt.show()

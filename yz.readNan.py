import numpy as np
import pandas as pd

np.random.seed(145)
veriSeti = pd.read_csv('SISA.csv')

#print(veriSeti.head(3))

#Veri setindeki 'nan' değerlerini değiştirmek için:
veriSeti.fillna(0, inplace = True)

features = veriSeti.iloc[:,2:36]

classes = veriSeti.Hospitalized

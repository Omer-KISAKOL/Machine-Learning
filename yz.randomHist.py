import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

np.random.seed(0)
values = np.random.randn(100) 
s = pd.Series(values)

s.plot(kind = 'hist', title = 'Normally distributed random values') 

plt.show()
print(s.describe())

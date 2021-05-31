import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

df = pd.DataFrame({'A':[1,2,1,4,3,5,2,3,4,1,], 'B':[12,14,11,16,18,18,22,13,21,17], 
                   'C':['a', 'a', 'b', 'a', 'b', 'c', 'b', 'a', 'b', 'a']}, index=(1,2,3,4,5,6,7,8,9,10))

print(df)

print(df.describe())
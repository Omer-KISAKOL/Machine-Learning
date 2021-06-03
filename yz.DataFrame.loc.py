import pandas as pd

veriSeti = pd.DataFrame(columns=['A','B','C'])

veriSeti.loc[0,'A'] = 1
veriSeti.loc[1] = [2,3,4]
veriSeti.loc[2] = {'A':3, 'B':9, 'C':8}
veriSeti.loc[0] = [-1,0,1] #üzerine yama işlemi

print(veriSeti)
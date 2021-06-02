import pandas as pd

vs1 = pd.DataFrame(columns = ['A','B'])

vs1.loc[0] = {'A':'a1','B':'b1'}
vs1.loc[1] = {'A':'a2','B':'b2'} 


vs2 = pd.DataFrame(columns = ['B','C'])

vs2.loc[0] = {'B':'b1', 'C':'c1'}

print(vs1,'\n')
print(vs2,'\n')

vs = vs1.append(vs2, ignore_index = True)
print(vs,'\n')

vs.fillna(0, inplace=True)
print(vs)
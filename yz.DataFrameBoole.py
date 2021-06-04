#Boole endeksi olan bir DataFrame'e erişim

import pandas as pd

df = pd.DataFrame({"color": ['red', 'blue', 'red', 'blue']},
 index=[True, False, True, False])


print(df,'\n')
print(df.loc[True])
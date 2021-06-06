import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

num_securities = 1000
num_periods = 1000
period_frequency = 'W'

start_date = '2000-12-31'
np.random.seed([3,1])
means = [0,0]
covariance = [[1.,5e-3],[5e-3,1.]]

#~0,005 korelasyon ile m[0] ve m[1] veri setleri üretir
m = np.random.multivariate_normal(means, covariance, (num_periods, num_securities)).T

#Zaman serisi ve kimliklerini oluşturma
ids = pd.Index(['s{:05d}'.format(s) for s in range(num_securities)])
tidx = pd.date_range(start = start_date, periods = num_periods, freq = period_frequency)

#scale down yapılıyor pozitif ortalama için 1e-7 ekleniyor
security_returns = pd.DataFrame(m[0]/25+1e-7,tidx,ids)
security_signals = pd.DataFrame(m[1], tidx, ids)

#Sinyalleri 5'te 1'lik bölümlere ayırmak için:
def qcut(s,q=5):
    labels = ['q{}'.format(i) for i in range(1,6)]
    return pd.qcut(s,q,labels = labels)

cut = security_signals.stack().groupby(level = 0).apply(qcut)

returns_cut = security_returns.stack().rename('returns') \
    .to_frame().set_index(cut, append=True) \
    .swaplevel(2,1).sort_index().squeeze() \
    .groupby(level = [0,1]).mean().unstack()    
    
    
fig = plt.figure(figsize = (15,5))
ax1 = plt.subplot2grid((1,3),(0,0))
ax2 = plt.subplot2grid((1,3),(0,1))
ax3 = plt.subplot2grid((1,3),(0,2))

#Cumulative Returns
returns_cut.add(1).cumprod() \
    .plot(colormap = 'jet', ax = ax1, title = "Cumulative Returns")
leg1 = ax1.legend(loc = 'upper left', ncol = 2, prop={'size':10}, fancybox = True)
leg1.get_frame().set_alpha(.8)

#Rolling 50 Week Return
returns_cut.add(1).rolling(50).apply(lambda x: x.prod()) \
    .plot(colormap = 'jet', ax = ax2, title = "Rolling 50 Week Return" )
leg2 = ax2.legend(loc = 'upper left', ncol = 2, prop = {'size':10}, fancybox = True)
leg2.get_frame().set_alpha(.8)

#Return Distribution
returns_cut.plot.box(vert = False, ax= ax3, title = "Return Distribution")
fig.autofmt_xdate()

plt.show()

scatter_matrix(returns_cut, alpha = 0.5, figsize = (8,8), diagonal='hist')

plt.show()
















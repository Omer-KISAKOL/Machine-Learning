import matplotlib.pyplot as plt

etiketler = ['Türkiye', 'Almanya', 'Yunanistan']
veriler = [339104, 762338, 319636]
renkler = ['#ff9999', '#66b3ff', '#99ff99']
ayrıkParça = [0.1,0,0]

plt.pie(veriler, labels = etiketler, colors = renkler, explode = ayrıkParça, autopct = '%1.1f%%')
plt.axis('equal')

#merkezDaire = plt.Circle((0,0),0.7,fc = 'white')
#sekil = plt.gcf()
#sekil.gca().add_artist(merkezDaire)

plt.show()
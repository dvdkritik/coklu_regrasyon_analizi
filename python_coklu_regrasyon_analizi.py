# Gerekli Kütüphaneler Eklendi
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


girdi = pd.read_csv('sahibinden.csv')  # Veri seti eklendi
girdi.head()
girdi.shape
girdi.info()  # Veri setinin okunduğu kontrol edildi

# Ev Fiyatları İncelendi
plt.subplots(figsize=(12, 9))
sns.distplot(girdi['fiyat'], fit=stats.norm)
(mu, sigma) = stats.norm.fit(girdi['fiyat'])
plt.show()

# Olasılık durumu grafiği
fig = plt.figure()
stats.probplot(girdi["fiyat"], plot=plt)
plt.show()

# numpy kütüphanesiyle logaritmik dönüşüm yapıldı
girdi['fiyat'] = np.log1p(girdi['fiyat'])

# Normal dağılım kontrol edildi
plt.subplots(figsize=(12, 9))
sns.distplot(girdi['fiyat'], fit=stats.norm)
(mu, sigma) = stats.norm.fit(girdi['fiyat'])

# Olasılık durumu görselleştirildi
fig = plt.figure()
stats.probplot(girdi['fiyat'], plot=plt)
plt.show()

# Veriler arasındaki korelasyona bakıldı
girdi_corr = girdi.select_dtypes(include=[np.number])
girdi_corr.shape

corr =girdi_corr.corr()
plt.subplots(figsize = (20, 11))
sns.heatmap(corr, annot = True)


corr = girdi_corr.corr()
plt.subplots(figsize=(20, 11))
sns.heatmap(corr, annot=True)

# Satış fiyatı ile en iyi ilişki içerisinde olan özellik m2 olarak belirlendikten sonra aralarındaki ilişki çubuk grafik üzerinde gösterildi
girdi.m2.unique()
sns.barplot(girdi.m2, girdi.fiyat)
plt.show()

col = ['oda_sayisi', 'salon_sayisi', 'm2', 'bina_yasi', 'bulundugu_kat', 'balkon_sayisi', 'aidat', 'fiyat']
sns.set(style='ticks')
sns.pairplot(girdi[col], size=3, kind='reg')
plt.show()

# Hedef değişken ile olan ilişkiler yazdırıldı
print('Hedef değişken(fiyat) ile en iyi ilişkisi olan değişkeni bulalım')
corr = girdi.corr()
corr.sort_values(['fiyat'], ascending=False, inplace=True)
print(corr.fiyat)

y = girdi['fiyat']
del girdi['fiyat']

X = girdi.values
y = y.values

#(%80 eğitim, % 20 test)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 7)

from sklearn import linear_model
model1 = linear_model.LinearRegression()

#Verilerin modele uygulanması
model1.fit(X_train, y_train)


print("Tahmin edilen değer : " + str(model1.predict([X_test[5]])))
print("Gerçek değer : " + str(y_test[1]))
print("Doğruluk oranı :  ", model1.score(X_test, y_test)* 100)
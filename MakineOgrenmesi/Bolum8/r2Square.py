# -*- coding: utf-8 -*-

#1.kutuphanler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score


# veri yukleme
veriler = pd.read_csv('maaslar.csv')


# data frame dilimleme (slice)
x = veriler.iloc[:,1:2]
y = veriler.iloc[:,2:]


# NumPy array (dizi) dönüşümü
X = x.values
Y = y.values

# linear regression
# doğrusal model oluşturma
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x,y)

print('Linear R2 değeri')
print(r2_score(Y,lin_reg.predict(X)))


# polynomial regression
# 2.dereceden polinom
# doğrusal olmayan (nonlinear model) oluşturma
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
x_poly = poly_reg.fit_transform(X)


lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly, y)


# 4.dereceden polinom
poly_reg3 = PolynomialFeatures(degree = 4)
x_poly3 = poly_reg3.fit_transform(X)
print(x_poly)

lin_reg3 = LinearRegression()
lin_reg3.fit(x_poly3, y)


# Görselleştirme

plt.scatter(X,Y, color='red')
plt.plot(x,lin_reg.predict(X),color='blue')
plt.show()

plt.scatter(X,Y,color ='red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)),color ='blue')
plt.show()

plt.scatter(X,Y,color ='red')
plt.plot(X,lin_reg3.predict(poly_reg3.fit_transform(X)),color ='blue')
plt.show()


# tahminler
print(lin_reg.predict([[11]]))
print(lin_reg.predict([[6]]))

print(lin_reg2.predict(poly_reg.fit_transform([[6.6]])))
print(lin_reg2.predict(poly_reg.fit_transform([[11]])))

print('Polynomial R2 değeri')
print(r2_score(Y,lin_reg2.predict(poly_reg.fit_transform(X))))


# SVR (support vector)
# verilerin ölçeklenmesi
from sklearn.preprocessing import StandardScaler

sc1 = StandardScaler()
x_olcekli = sc1.fit_transform(X)

sc2 = StandardScaler()
y_olcekli = sc2.fit_transform(Y)

from sklearn.svm import SVR

svr_reg = SVR(kernel = 'rbf')
svr_reg.fit(x_olcekli, y_olcekli)

plt.scatter(x_olcekli,y_olcekli, color = 'red')
plt.plot(x_olcekli,svr_reg.predict(x_olcekli), color = 'blue')
plt.show()

print(svr_reg.predict([[11]]))
print(svr_reg.predict([[6.6]]))

print('SVR R2 değeri')
print(r2_score(y_olcekli,svr_reg.predict(x_olcekli)))


# decision tree regression (karar ağacı)
from sklearn.tree import DecisionTreeRegressor

r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(X,Y)
Z = X + 0.5
K = X - 0.4

plt.scatter(X,Y,color='red')
plt.plot(X,r_dt.predict(X),color='blue')

plt.plot(x,r_dt.predict(Z),color='green')
plt.plot(x,r_dt.predict(K),color='yellow')
plt.show()

print(r_dt.predict([[11]]))
print(r_dt.predict([[6.6]]))

print('Decision Tree R2 değeri')
print(r2_score(Y,r_dt.predict(X)))


# random forest regression (rassal orman regresyonu)

from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators=10, random_state=0)
rf_reg.fit(X,Y.ravel())

print(rf_reg.predict([[6.5]]))

plt.scatter(X,Y,color='red')
plt.plot(X,rf_reg.predict(X),color='blue')

plt.plot(X,rf_reg.predict(Z),color='green')
plt.plot(X,rf_reg.predict(K),color='yellow')



# r2 square tahmin değerlendirilmesi

print('Random forest R2 değeri')
print(r2_score(Y,rf_reg.predict(X)))

print(r2_score(Y,rf_reg.predict(K)))
print(r2_score(Y,rf_reg.predict(Z)))


print("######################## OZET ########################")

print("--------------------------------")
print('Linear R2 değeri')s
print(r2_score(Y,lin_reg.predict(X)))
print("--------------------------------")
print('Polynomial R2 değeri')
print(r2_score(Y,lin_reg2.predict(poly_reg.fit_transform(X))))
print("--------------------------------")
print('SVR R2 değeri')
print(r2_score(y_olcekli,svr_reg.predict(x_olcekli)))
print("--------------------------------")
print('Decision Tree R2 değeri')
print(r2_score(Y,r_dt.predict(X)))
print("--------------------------------")
print('Random forest R2 değeri')
print(r2_score(Y,rf_reg.predict(X)))
print("--------------------------------")



























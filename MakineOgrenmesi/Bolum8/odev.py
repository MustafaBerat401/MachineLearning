# -*- coding: utf-8 -*-

#1.kutuphanler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score
import statsmodels.api as sm

# veri yukleme
veriler = pd.read_csv('maaslar_yeni.csv')


# data frame dilimleme (slice)
x = veriler.iloc[:,2:5]
y = veriler.iloc[:,5:]


# NumPy array (dizi) dönüşümü
X = x.values
Y = y.values

print(veriler.corr())

# linear regression
# doğrusal model oluşturma
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)

print("LİNEAR OLS")
model = sm.OLS(lin_reg.predict(X),X)
print(model.fit().summary())

print("Linear R2 degeri:")
print(r2_score(Y,lin_reg.predict((X))))
      
# polynomial regression
# 2.dereceden polinom
# doğrusal olmayan (nonlinear model) oluşturma
from sklearn.preprocessing import PolynomialFeatures


# 4.dereceden polinom
poly_reg3 = PolynomialFeatures(degree = 4)
x_poly3 = poly_reg3.fit_transform(X)
print(x_poly3)

lin_reg3 = LinearRegression()
lin_reg3.fit(x_poly3, y)



# tahminler



print("POLY OLS")
model2 = sm.OLS(lin_reg3.predict(poly_reg3.fit_transform(X)),X)
print(model2.fit().summary())

print("Polynomial R2 degeri:")
print(r2_score(Y,lin_reg3.predict(poly_reg3.fit_transform(X))))



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


print("SVR OLS")
model3 = sm.OLS(svr_reg.predict(x_olcekli),x_olcekli)
print(model3.fit().summary())

print("SVR R2 degeri:")
print(r2_score(y_olcekli, svr_reg.predict(x_olcekli)))


# decision tree regression (karar ağacı)
from sklearn.tree import DecisionTreeRegressor

r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(X,Y)


print("DECİSİON TREE OLS")
model4 = sm.OLS(r_dt.predict(X),X)
print(model4.fit().summary())

print("Decision tree R2 degeri:")
print(r2_score(Y, r_dt.predict(X)))




# random forest (rassal orman)

from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators=10, random_state=0)
rf_reg.fit(X,Y.ravel())



print("RANDOM FOREST OLS")
model5 = sm.OLS(rf_reg.predict(X),X)
print(model5.fit().summary())

print("Random forest R2 degeri:")
print(r2_score(Y, rf_reg.predict(X)))





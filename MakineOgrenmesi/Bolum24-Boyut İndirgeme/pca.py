# -*- coding: utf-8 -*-

# Kütüphaneler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# veri kümesi
veriler = pd.read_csv('Wine.csv')
X = veriler.iloc[:,0:13].values
y = veriler.iloc[:,13].values

# eğitim ve test kümelerinin bölünmesi
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Ölçekleme
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# PCA

from sklearn.decomposition import PCA

pca = PCA(n_components=2)

X_train2 = pca.fit_transform(X_train)
X_test2 = pca.transform(X_test)

# PCA dönüşümünden önce gelen Logistic Regression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)

# PCA dönüşümünden sonra gelen Logistic Regression
classifier2 = LogisticRegression(random_state=0)
classifier2.fit(X_train2,y_train)

# tahminler
y_pred = classifier.predict(X_test)

y_pred2 = classifier2.predict(X_test2)

from sklearn.metrics import confusion_matrix

# actual / PCA olmadan çıkan sonuç
cm = confusion_matrix(y_test, y_pred)
print('gerçek / PCAsiz')
print(cm)

# actual / PCA sonrası çıkan sonuç
cm2 = confusion_matrix(y_test, y_pred2)
print('gerçek / pca ile')
print(cm2)

# PCA sonrası / PCA öncesi
print('pcasiz / pcali')
cm3 = confusion_matrix(y_pred, y_pred2)
print(cm3)





















